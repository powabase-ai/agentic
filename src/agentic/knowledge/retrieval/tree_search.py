"""
TreeSearch retrieval algorithm.

Uses LLM reasoning over lightweight ToC structures (titles + summaries)
to identify relevant sections, returning SelectedNode references that
the search service uses for targeted text retrieval from the DB.

Only compatible with the "page_index" indexing strategy.
"""

import json
import logging
import re
from dataclasses import dataclass

from agentic.knowledge.model_config import PAGEINDEX_RETRIEVAL_MODEL
from agentic.llm.routing import maybe_route_through_responses, reasoning_call_kwargs

logger = logging.getLogger(__name__)


@dataclass
class SelectedNode:
    """A node selected by the LLM during tree-search reasoning.

    Contains enough metadata for the search service to fetch the
    corresponding section text from page_index_nodes and build
    RetrievedItem objects.
    """

    toc_id: str  # UUID of the page_index_toc record
    node_id: str  # "0001" — plain node ID within document
    doc_prefix: str  # "d0" — index into toc_records list
    title: str
    doc_name: str | None
    doc_description: str | None
    source_id: str | None
    knowledge_base_id: str | None
    rank: int  # 0-based relevance rank within document
    doc_rank: int = 0  # 0-based document relevance rank from Stage 1


def _build_doc_summary(toc_record: dict, doc_index: int) -> str:
    """Build a compact document summary for Stage 1 screening.

    Includes doc_name, doc_description, and top-level section titles only.
    """
    doc_name = toc_record.get("doc_name", "Unknown Document")
    doc_desc = toc_record.get("doc_description", "")
    structure = toc_record.get("structure", [])

    # Extract only top-level titles
    top_titles = [node.get("title", "") for node in structure if isinstance(node, dict)]

    lines = [f"[d{doc_index}] {doc_name}"]
    if doc_desc:
        lines.append(f"  Description: {doc_desc}")
    if top_titles:
        lines.append(f"  Sections: {', '.join(top_titles)}")
    return "\n".join(lines)


def _extract_doc_ids_from_response(response_text: str) -> list[str]:
    """Parse LLM response to extract document IDs like 'd0', 'd2'."""
    # Try JSON parsing first
    try:
        data = json.loads(response_text)
        if isinstance(data, list):
            return [str(did) for did in data]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: find d{N} patterns
    doc_ids = re.findall(r"\b(d\d+)\b", response_text)
    return list(dict.fromkeys(doc_ids))  # Deduplicate preserving order


def _build_condensed_view(toc_record: dict, id_prefix: str = "") -> str:
    """Build a condensed view of a ToC showing titles + summaries only.

    This omits full text content to keep the prompt small enough for the
    LLM to reason over the entire document structure at once.

    Args:
        toc_record: A ToC record dict with 'structure', 'doc_name', etc.
        id_prefix: Optional prefix for node IDs (e.g. "d0:") to make them
            globally unique across multiple documents.
    """
    structure = toc_record.get("structure", [])
    doc_name = toc_record.get("doc_name", "Unknown Document")

    lines = [f"Document: {doc_name}"]
    doc_desc = toc_record.get("doc_description")
    if doc_desc:
        lines.append(f"Description: {doc_desc}")
    lines.append("")

    def _walk(nodes, depth=0):
        if not isinstance(nodes, list):
            return
        for node in nodes:
            indent = "  " * depth
            title = node.get("title", "Untitled")
            node_id = node.get("node_id", "?")
            summary = node.get("summary") or node.get("prefix_summary") or ""

            lines.append(f"{indent}[{id_prefix}{node_id}] {title}")
            if summary:
                lines.append(f"{indent}  Summary: {summary}")

            children = node.get("nodes")
            if children:
                _walk(children, depth + 1)

    _walk(structure)
    return "\n".join(lines)


def _collect_valid_node_ids(toc_map: dict[str, dict]) -> set[str]:
    """Walk all ToC structures and collect valid node IDs (both prefixed and plain).

    Returns a set containing both "d0:0001" prefixed forms and plain "0001"
    forms so that either format from the LLM response can be validated.
    """
    valid: set[str] = set()

    def _walk(nodes: list, prefix: str) -> None:
        if not isinstance(nodes, list):
            return
        for node in nodes:
            nid = node.get("node_id")
            if nid:
                valid.add(f"{prefix}:{nid}")
                valid.add(nid)
            children = node.get("nodes")
            if children:
                _walk(children, prefix)

    for doc_prefix, toc_record in toc_map.items():
        _walk(toc_record.get("structure", []), doc_prefix)

    return valid


def _extract_node_ids_from_response(
    response_text: str,
    valid_ids: set[str] | None = None,
) -> list[str]:
    """Parse LLM response to extract node IDs.

    Supports both prefixed IDs (e.g. "d0:0001") for multi-document KBs
    and plain IDs (e.g. "0001") for single-document KBs.

    Args:
        response_text: Raw LLM response text.
        valid_ids: Optional set of known-valid node IDs. When provided,
            results are filtered to only include IDs present in this set.
            This prevents the plain 4-digit regex from matching years,
            page numbers, or other numbers in the LLM response.
    """
    # Try JSON parsing first
    try:
        data = json.loads(response_text)
        if isinstance(data, list):
            result = [str(nid) for nid in data]
            if valid_ids is not None:
                result = [nid for nid in result if nid in valid_ids]
            return result
        if isinstance(data, dict) and "node_ids" in data:
            result = [str(nid) for nid in data["node_ids"]]
            if valid_ids is not None:
                result = [nid for nid in result if nid in valid_ids]
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: find prefixed IDs first (d0:0001), then plain 4-digit IDs
    prefixed = re.findall(r"\b(d\d+:\d{4})\b", response_text)
    if prefixed:
        result = list(dict.fromkeys(prefixed))
        if valid_ids is not None:
            result = [nid for nid in result if nid in valid_ids]
        return result

    plain = re.findall(r"\b(\d{4})\b", response_text)
    result = list(dict.fromkeys(plain))  # Deduplicate preserving order
    if valid_ids is not None:
        result = [nid for nid in result if nid in valid_ids]
    return result


def _get_node_title(structure: list, node_id: str) -> str:
    """Find a node by ID in the structure and return its title."""

    def _find(nodes):
        if not isinstance(nodes, list):
            return None
        for node in nodes:
            if node.get("node_id") == node_id:
                return node.get("title", "")
            children = node.get("nodes")
            if children:
                found = _find(children)
                if found is not None:
                    return found
        return None

    return _find(structure) or ""


class TreeSearchAlgorithm:
    """Retrieval algorithm that uses LLM reasoning over tree structures.

    Instead of vector similarity, this algorithm:
    1. Presents the condensed ToC (titles + summaries) to an LLM
    2. Asks the LLM which sections are relevant to the query
    3. Returns SelectedNode references (no text lookup)

    The search service handles fetching actual section text from the DB
    and building RetrievedItem objects.

    Supports multiple documents per knowledge base by prefixing node IDs
    with a document index (e.g. "d0:0001", "d1:0003") so the LLM can
    unambiguously reference nodes across different documents.
    """

    name = "tree_search"

    async def select_documents(
        self,
        query: str,
        toc_records: list[dict],
        config: dict | None = None,
    ) -> list[tuple[int, dict]]:
        """Stage 1: Identify which documents are relevant to the query.

        Args:
            query: User search query
            toc_records: All ToC records from the KB
            config: Optional config with retrieval_model

        Returns:
            List of (doc_index, toc_record) tuples for relevant documents,
            ordered by relevance.
        """
        config = config or {}
        retrieval_model = config.get("retrieval_model", PAGEINDEX_RETRIEVAL_MODEL)
        reasoning_effort = config.get("retrieval_reasoning_effort")
        routed_model = maybe_route_through_responses(retrieval_model, reasoning_effort)
        reasoning_kwargs = reasoning_call_kwargs(reasoning_effort, routed_model)

        # Build compact summaries
        summaries = []
        for i, toc_record in enumerate(toc_records):
            summaries.append(_build_doc_summary(toc_record, i))

        all_summaries = "\n\n".join(summaries)

        prompt = f"""You are a document retrieval assistant. You are given a list of documents with their names, descriptions, and top-level section titles.

Your task: identify which documents are likely to contain information relevant to the user's query. Return ONLY the document IDs of relevant documents.

Documents:
{all_summaries}

User Query: {query}

IMPORTANT: Return ONLY a JSON array of document ID strings. Example: ["d0", "d2"]
If all documents seem relevant, include all of them. If none seem relevant, return all of them."""

        import litellm

        try:
            response = await litellm.acompletion(
                model=routed_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                drop_params=True,
                num_retries=3,
                **reasoning_kwargs,
            )

            response_text = response.choices[0].message.content or ""
            doc_ids = _extract_doc_ids_from_response(response_text)

            logger.info(
                f"Document selection: {len(doc_ids)} of {len(toc_records)} "
                f"documents selected: {doc_ids}"
            )

            # Map doc IDs back to (index, toc_record) pairs
            results = []
            for did in doc_ids:
                match = re.match(r"d(\d+)", did)
                if match:
                    idx = int(match.group(1))
                    if 0 <= idx < len(toc_records):
                        results.append((idx, toc_records[idx]))

            if results:
                return results

        except Exception as e:
            logger.warning(f"Document selection failed, using all documents: {e}")

        # Fallback: return all documents
        return [(i, rec) for i, rec in enumerate(toc_records)]

    async def select_nodes(
        self,
        query: str,
        toc_records: list[dict],
        config: dict | None = None,
        doc_rank: int = 0,
    ) -> list[SelectedNode]:
        """Identify relevant sections using LLM reasoning over ToC structures.

        Args:
            query: The user's search query
            toc_records: List of ToC record dicts from the DB, each with:
                - id: UUID of the toc record
                - structure: JSONB tree hierarchy (metadata only)
                - doc_name, doc_description, source_id, knowledge_base_id
            config: Optional config dict with:
                - retrieval_model: LLM model (default: PAGEINDEX_RETRIEVAL_MODEL)
                - top_k: Max nodes to select (default: 5)

        Returns:
            List of SelectedNode objects with toc_id + node_id for DB lookup
        """
        if not toc_records:
            logger.warning("No ToC records provided for tree_search")
            return []

        config = config or {}
        retrieval_model = config.get("retrieval_model", PAGEINDEX_RETRIEVAL_MODEL)
        reasoning_effort = config.get("retrieval_reasoning_effort")
        routed_model = maybe_route_through_responses(retrieval_model, reasoning_effort)
        reasoning_kwargs = reasoning_call_kwargs(reasoning_effort, routed_model)
        top_k = config.get("top_k", 5)

        # Build condensed views with document-scoped node ID prefixes.
        condensed_parts = []
        toc_map: dict[str, dict] = {}  # "d0" -> toc_record
        for i, toc_record in enumerate(toc_records):
            doc_prefix = f"d{i}"
            toc_map[doc_prefix] = toc_record
            condensed = _build_condensed_view(toc_record, id_prefix=f"{doc_prefix}:")
            condensed_parts.append(condensed)

        full_condensed = "\n\n---\n\n".join(condensed_parts)

        # Ask LLM to identify relevant node IDs
        prompt = f"""You are a document retrieval assistant. You are given hierarchical document structures with section titles, node IDs, and summaries. The knowledge base may contain multiple documents.

Your task: identify which sections are most relevant to the user's query. Return the node IDs of the most relevant sections (up to {top_k} sections). Sections can come from one or multiple documents.

Node IDs are prefixed with a document identifier (e.g. "d0:0001" means node 0001 from the first document, "d1:0003" means node 0003 from the second document).

IMPORTANT: Return ONLY a JSON array of node_id strings including their document prefix. Example: ["d0:0001", "d1:0005", "d0:0012"]

Document Structure:
{full_condensed}

User Query: {query}

Return the most relevant node IDs as a JSON array:"""

        import litellm

        try:
            response = await litellm.acompletion(
                model=routed_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                drop_params=True,
                num_retries=3,
                **reasoning_kwargs,
            )
            response_text = response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"Node selection LLM call failed: {e}")
            return []

        valid_ids = _collect_valid_node_ids(toc_map)
        selected_ids = _extract_node_ids_from_response(response_text, valid_ids=valid_ids)

        logger.info(
            f"TreeSearch selected {len(selected_ids)} nodes: {selected_ids}"
        )

        # Build SelectedNode objects from the LLM's response
        results: list[SelectedNode] = []
        for rank, prefixed_id in enumerate(selected_ids[:top_k]):
            if ":" in prefixed_id:
                doc_prefix, node_id = prefixed_id.split(":", 1)
                toc_record = toc_map.get(doc_prefix)
                if not toc_record:
                    logger.warning(f"Unknown document prefix: {doc_prefix}")
                    continue
            else:
                # Unprefixed fallback (single-doc): use first toc
                doc_prefix = "d0"
                node_id = prefixed_id
                toc_record = toc_map.get(doc_prefix)
                if not toc_record:
                    continue

            title = _get_node_title(
                toc_record.get("structure", []), node_id
            )

            results.append(SelectedNode(
                toc_id=toc_record["id"],
                node_id=node_id,
                doc_prefix=doc_prefix,
                title=title,
                doc_name=toc_record.get("doc_name"),
                doc_description=toc_record.get("doc_description"),
                source_id=toc_record.get("source_id"),
                knowledge_base_id=toc_record.get("knowledge_base_id"),
                rank=rank,
                doc_rank=doc_rank,
            ))

        return results
