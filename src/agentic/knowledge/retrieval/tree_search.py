"""
TreeSearch retrieval algorithm.

Uses LLM reasoning over a condensed tree structure (titles + summaries)
to identify relevant sections, then extracts full text from those nodes.

Only compatible with the "page_index" indexing strategy.
"""

import json
import logging
import re
from typing import TYPE_CHECKING

from agentic.knowledge.retrieval.base import RetrievalAlgorithm

if TYPE_CHECKING:
    from agentic.knowledge.models import RetrievalConfig, RetrievedChunk
    from agentic.knowledge.store import KnowledgeStore

logger = logging.getLogger(__name__)


def _build_condensed_view(tree_data: dict) -> str:
    """Build a condensed view of a tree showing titles + summaries only.

    This omits full text content to keep the prompt small enough for the
    LLM to reason over the entire document structure at once.
    """
    structure = tree_data.get("structure", [])
    doc_name = tree_data.get("doc_name", "Unknown Document")

    lines = [f"Document: {doc_name}"]
    doc_desc = tree_data.get("doc_description")
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

            lines.append(f"{indent}[{node_id}] {title}")
            if summary:
                lines.append(f"{indent}  Summary: {summary}")

            children = node.get("nodes")
            if children:
                _walk(children, depth + 1)

    _walk(structure)
    return "\n".join(lines)


def _extract_node_ids_from_response(response_text: str) -> list[str]:
    """Parse LLM response to extract node IDs."""
    # Try JSON parsing first
    try:
        data = json.loads(response_text)
        if isinstance(data, list):
            return [str(nid) for nid in data]
        if isinstance(data, dict) and "node_ids" in data:
            return [str(nid) for nid in data["node_ids"]]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: find 4-digit zero-padded IDs in the text
    ids = re.findall(r"\b(\d{4})\b", response_text)
    return list(dict.fromkeys(ids))  # Deduplicate preserving order


def _get_node_text(tree_data: dict, node_id: str) -> dict | None:
    """Find a node by ID and return its details."""
    structure = tree_data.get("structure", [])

    def _find(nodes):
        if not isinstance(nodes, list):
            return None
        for node in nodes:
            if node.get("node_id") == node_id:
                return node
            children = node.get("nodes")
            if children:
                found = _find(children)
                if found:
                    return found
        return None

    return _find(structure)


class TreeSearchAlgorithm(RetrievalAlgorithm):
    """Retrieval algorithm that uses LLM reasoning over tree structures.

    Instead of vector similarity, this algorithm:
    1. Presents the condensed tree (titles + summaries) to an LLM
    2. Asks the LLM which sections are relevant to the query
    3. Extracts full text from those sections
    4. Returns results as RetrievedChunk objects

    Requires tree data to be passed via config.extra["trees"].
    """

    name = "tree_search"

    async def retrieve(
        self,
        query: str,
        store: "KnowledgeStore",
        config: "RetrievalConfig",
    ) -> list["RetrievedChunk"]:
        """Retrieve relevant sections using LLM tree reasoning.

        Args:
            query: The user's search query
            store: Not used for tree search (trees passed via config)
            config: RetrievalConfig with extra keys:
                - trees: List of tree dicts from the database
                - retrieval_model: LLM model for retrieval reasoning
                    (default: "gpt-4o-mini")

        Returns:
            List of RetrievedChunk objects with relevant sections
        """
        from agentic.knowledge.models import RetrievedChunk

        trees = config.extra.get("trees", [])
        if not trees:
            logger.warning("No trees provided for tree_search")
            return []

        retrieval_model = config.extra.get("retrieval_model", "gpt-4o-mini")
        top_k = config.top_k

        # Build condensed views from all trees
        condensed_parts = []
        for i, tree_record in enumerate(trees):
            tree_data = tree_record.get("tree_data", {})
            condensed = _build_condensed_view(tree_data)
            condensed_parts.append(condensed)

        full_condensed = "\n\n---\n\n".join(condensed_parts)

        # Ask LLM to identify relevant node IDs
        prompt = f"""You are a document retrieval assistant. You are given a hierarchical document structure with section titles, node IDs, and summaries.

Your task: identify which sections are most relevant to the user's query. Return the node IDs of the most relevant sections (up to {top_k} sections).

IMPORTANT: Return ONLY a JSON array of node_id strings. Example: ["0001", "0005", "0012"]

Document Structure:
{full_condensed}

User Query: {query}

Return the most relevant node IDs as a JSON array:"""

        import litellm

        response = await litellm.acompletion(
            model=retrieval_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        response_text = response.choices[0].message.content or ""
        selected_ids = _extract_node_ids_from_response(response_text)

        logger.info(
            f"TreeSearch selected {len(selected_ids)} nodes: {selected_ids}"
        )

        # Extract full text from selected nodes
        results: list[RetrievedChunk] = []
        for rank, node_id in enumerate(selected_ids[:top_k]):
            for tree_record in trees:
                tree_data = tree_record.get("tree_data", {})
                node = _get_node_text(tree_data, node_id)
                if node:
                    text = node.get("text", "")
                    if not text:
                        # Use summary as fallback if no text
                        text = (
                            node.get("summary")
                            or node.get("prefix_summary")
                            or ""
                        )

                    if text:
                        results.append(
                            RetrievedChunk(
                                chunk_id=f"tree-{node_id}",
                                text=text,
                                score=1.0 - (rank * 0.05),  # Rank-based score
                                source_id=tree_record.get("source_id"),
                                knowledge_base_id=tree_record.get(
                                    "knowledge_base_id"
                                ),
                                meta={
                                    "node_id": node_id,
                                    "title": node.get("title", ""),
                                    "retrieval_method": "tree_search",
                                },
                            )
                        )
                    break  # Found the node, move to next ID

        return results
