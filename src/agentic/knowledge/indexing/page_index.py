"""
PageIndex indexing algorithm — Agentic platform wrapper.

This module is the public entry point for the PageIndex indexing strategy.
It wraps the internal _pageindex_lib pipeline and adapts its output into the
Agentic platform's IndexResult format.

The indexer accepts a markdown document and produces two complementary outputs
designed for a two-phase retrieval system:

1. **toc_structure** — A lightweight, metadata-only tree (titles, node_ids,
   summaries, parent/child relationships). Contains NO section text. This is
   given to an LLM in Phase 1 so it can reason over the entire document
   outline cheaply and identify which sections are relevant to a query.

2. **sections** — A flat list of SectionData objects, each carrying the full
   text of one tree node plus its depth, parent pointer, and metadata. These
   are stored as individual database rows and fetched selectively in Phase 2
   after the LLM picks the relevant node_ids from the ToC.

Two pipelines are supported, selected automatically based on input:

- **Markdown pipeline** (`md_to_tree`): For documents that arrive as a single
  markdown string. Parses `# headers` to build the initial tree, then uses
  LLM calls to split oversized leaf nodes into sub-sections.

- **Page-aware pipeline** (`md_to_tree_from_pages`): For documents that arrive
  with per-page text arrays (e.g. from PDF extractors). Delegates to the
  reference PageIndex `tree_parser()` which performs full ToC detection,
  verification, correction, and fallback splitting. After tree_parser builds
  the structure, text is assigned from page ranges, parent text is trimmed to
  avoid duplication, and small siblings are merged.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agentic.knowledge.indexing.base import IndexingAlgorithm
from agentic.knowledge.models import IndexingConfig, IndexResult

logger = logging.getLogger(__name__)


@dataclass
class SectionData:
    """Data for a single section/node extracted from the tree.

    Each SectionData corresponds to one node in the hierarchical tree and is
    stored as an individual row in the `page_index_nodes` database table.
    During retrieval, only the sections whose node_ids were selected by the
    LLM (from the toc_structure) are fetched.

    Attributes:
        node_id: Unique identifier within the document (e.g. "0001", "0002").
            Assigned sequentially in pre-order traversal by write_node_id().
        title: Section heading text (from markdown header or LLM-inferred).
        text: Full text content of this section. For leaf nodes this is the
            complete section text. For parent nodes in the markdown pipeline
            this includes all descendant text (overlapping). For parent nodes
            in the page-aware pipeline this is trimmed to only the intro
            portion before the first child (no overlap).
        depth: Nesting depth in the tree (0 = root level).
        parent_node_id: node_id of this section's parent, or None for roots.
            Enables reconstructing the hierarchy during retrieval.
        line_num: Source line number (markdown pipeline) or page number
            (page-aware pipeline) where this section starts. Used for
            citation references.
        meta: Additional metadata dict. May contain:
            - "summary" / "prefix_summary": LLM-generated summaries
            - "start_page" / "end_page": Page range (page-aware pipeline only)
    """

    node_id: str
    title: str
    text: str
    depth: int
    parent_node_id: str | None
    line_num: int | None
    meta: dict = field(default_factory=dict)


def _count_nodes(nodes: list) -> int:
    """Recursively count the total number of nodes in a tree structure.

    Walks all nodes and their nested children (via the "nodes" key) to
    produce a total count. Used for indexing statistics.

    Args:
        nodes: List of tree node dicts, each potentially containing a
            "nodes" key with child node lists.

    Returns:
        Total number of nodes across all levels of the tree.
    """
    count = 0
    if isinstance(nodes, list):
        for node in nodes:
            count += 1
            if "nodes" in node:
                count += _count_nodes(node["nodes"])
    return count


def _extract_sections_from_structure(structure: list) -> list[SectionData]:
    """Split the combined tree into a metadata-only ToC and a flat sections list.

    This is the critical separation step that enables the two-phase retrieval
    design. It walks the tree in pre-order and for each node:

    1. **Pops** the "text" field out of the node dict (mutating it in-place).
       After this, the tree structure only contains metadata — titles, node_ids,
       summaries, line numbers, and child pointers. This becomes toc_structure.

    2. **Creates** a SectionData object capturing the popped text along with
       the node's position in the tree (depth, parent_node_id). These become
       the sections list stored as individual database rows.

    Also removes "text_token_count" if present (an artifact from the optional
    thinning step) since it's not needed in the final output.

    For the page-aware pipeline, also preserves start_index/end_index as
    start_page/end_page in the SectionData meta dict, and uses start_index
    as line_num when no explicit line_num exists (since page-aware nodes
    track page numbers, not line numbers).

    Args:
        structure: The tree structure list (mutated in-place to remove text).

    Returns:
        Flat list of SectionData objects in pre-order traversal order.
    """
    sections: list[SectionData] = []

    def _walk(nodes: list, depth: int = 0, parent_node_id: str | None = None) -> None:
        if not isinstance(nodes, list):
            return
        for node in nodes:
            nid = node.get("node_id")

            # Pop text out of the tree node — after this the tree is text-free.
            # This is the key operation that separates toc_structure from sections.
            text = node.pop("text", None)

            # Clean up thinning artifact if present
            node.pop("text_token_count", None)

            if nid:
                # Build metadata dict from optional fields
                meta = {}
                if node.get("summary"):
                    meta["summary"] = node["summary"]
                if node.get("prefix_summary"):
                    meta["prefix_summary"] = node["prefix_summary"]

                # Preserve page-range metadata from the page-aware pipeline.
                # These are stored in meta so retrieval can show page citations.
                start_page = node.get("start_index")
                end_page = node.get("end_index")
                if start_page is not None:
                    meta["start_page"] = start_page
                if end_page is not None:
                    meta["end_page"] = end_page

                # Determine line_num: prefer explicit line_num (markdown pipeline),
                # fall back to start_index/page number (page-aware pipeline).
                line_num = node.get("line_num")
                if line_num is None and start_page is not None:
                    line_num = start_page

                sections.append(SectionData(
                    node_id=nid,
                    title=node.get("title", ""),
                    text=text or "",
                    depth=depth,
                    parent_node_id=parent_node_id,
                    line_num=line_num,
                    meta=meta,
                ))

            # Recurse into children, incrementing depth and passing this
            # node's ID as the parent for child sections.
            children = node.get("nodes")
            if children:
                _walk(children, depth + 1, nid)

    _walk(structure)
    return sections


@dataclass
class PageIndexResult(IndexResult):
    """Complete result of PageIndex indexing, ready for database storage.

    This is the final output returned by PageIndexAlgorithm.aindex(). It
    contains both halves of the two-phase retrieval data:

    Attributes:
        toc_structure: The hierarchical tree with text stripped out. Each node
            contains only: title, node_id, summary or prefix_summary, line_num,
            and child nodes. This entire tree is given to the LLM in Phase 1
            retrieval so it can reason about document structure and select
            relevant node_ids.
        sections: Flat list of SectionData objects, one per tree node. Each
            carries the full section text plus tree position metadata. Stored
            as individual database rows and fetched selectively in Phase 2.
        doc_name: Human-readable document name (from source_name config or
            inferred from filename).
        doc_description: Optional LLM-generated description of the entire
            document. Only populated when if_add_doc_description="yes".
        strategy_name: Always "page_index" — identifies which indexing
            strategy produced this result.
        stats: Dictionary of indexing statistics:
            - node_count: Total nodes in the tree
            - section_count: Number of SectionData objects (same as node_count)
            - total_chars: Character count of the input content
            - model: LLM model used for tree building and summaries
    """

    toc_structure: list = field(default_factory=list)
    sections: list[SectionData] = field(default_factory=list)
    doc_name: str | None = None
    doc_description: str | None = None
    strategy_name: str = "page_index"
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def node_count(self) -> int:
        """Count total nodes in the tree."""
        return _count_nodes(self.toc_structure)


class PageIndexAlgorithm(IndexingAlgorithm):
    """Indexing algorithm that builds a hierarchical tree from markdown.

    This is the Agentic platform adapter that bridges the IndexingAlgorithm
    interface with the internal _pageindex_lib pipeline. It:

    1. Reads configuration from IndexingConfig.extra
    2. Routes to the appropriate pipeline (markdown or page-aware)
    3. Calls _extract_sections_from_structure() to separate the tree into
       toc_structure + sections
    4. Packages everything into a PageIndexResult

    The algorithm is stateless — all configuration comes from the IndexingConfig
    passed to each index/aindex call.
    """

    name = "page_index"

    def index(
        self,
        content: str,
        config: IndexingConfig,
        source_id: str | None = None,
    ) -> PageIndexResult:
        """Synchronous entry point — wraps aindex() with asyncio.run().

        Raises RuntimeError if called from within an async event loop.
        Use 'await aindex(...)' in async contexts instead.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running — safe to create one
            return asyncio.run(self.aindex(content, config, source_id))
        raise RuntimeError(
            "PageIndexAlgorithm.index() cannot be called from an async context "
            "(an event loop is already running). Use 'await aindex(...)' instead."
        )

    async def aindex(
        self,
        content: str,
        config: IndexingConfig,
        source_id: str | None = None,
    ) -> PageIndexResult:
        """Main async entry point — builds a tree structure from markdown content.

        This method orchestrates the full indexing pipeline:

        1. Extract configuration from config.extra
        2. Route to the correct pipeline based on whether page_texts is provided
        3. Both pipelines return {"doc_name": ..., "structure": [...]} where
           structure is a tree with text included in each node
        4. Call _extract_sections_from_structure() to split tree → toc + sections
        5. Package into PageIndexResult

        Args:
            content: The full markdown text content of the document.
            config: IndexingConfig with extra options dict:
                - source_name (str): Display name for the document
                - model (str): LLM model for tree building
                - if_add_node_summary ("yes"/"no"): Generate per-node summaries
                - if_thinning (bool): Merge small subtrees (markdown pipeline)
                - min_token_threshold (int): Thinning threshold in tokens
                - summary_token_threshold (int): Below this, text IS the summary
                - split_large_sections (bool): LLM-split oversized leaf nodes
                - max_node_tokens (int): Token threshold for leaf splitting
                - page_texts (list[str]): Per-page text array — if provided and
                  len >= 2, routes to the page-aware pipeline instead
                - toc_check_page_num (int): Pages to scan for ToC (page-aware)
                - max_page_num_each_node (int): Max pages/node before splitting
                - max_token_num_each_node (int): Max tokens/node before splitting
            source_id: Optional identifier for the source (passed through to result).

        Returns:
            PageIndexResult with toc_structure (metadata-only tree) and
            sections (list of SectionData with full text).
        """
        # Lazy import to avoid circular dependencies and speed up module load.
        # These are the two pipeline entry points in the _pageindex_lib package.
        from agentic.knowledge.indexing._pageindex_lib.page_index_md import (
            md_to_tree,
            md_to_tree_from_pages,
        )
        from agentic.knowledge.model_config import (
            PAGEINDEX_INDEXING_MODEL,
            PAGEINDEX_TOC_CHECK_PAGE_NUM,
            PAGEINDEX_MAX_PAGE_NUM_EACH_NODE,
            PAGEINDEX_MAX_TOKEN_NUM_EACH_NODE,
            PAGEINDEX_MAX_NODE_TOKENS,
            PAGEINDEX_MIN_TOKEN_THRESHOLD,
            PAGEINDEX_SUMMARY_TOKEN_THRESHOLD,
        )

        # --- Extract configuration from the extra dict ---
        extra = config.extra or {}
        source_name = extra.get("source_name")
        model = extra.get("model", PAGEINDEX_INDEXING_MODEL)
        if_add_node_summary = extra.get("if_add_node_summary", "yes")
        if_thinning = extra.get("if_thinning", False)
        min_token_threshold = extra.get("min_token_threshold", PAGEINDEX_MIN_TOKEN_THRESHOLD)
        summary_token_threshold = extra.get("summary_token_threshold", PAGEINDEX_SUMMARY_TOKEN_THRESHOLD)
        split_large_sections = extra.get("split_large_sections", True)
        max_node_tokens = extra.get("max_node_tokens", PAGEINDEX_MAX_NODE_TOKENS)

        # page_texts: if provided by PDF extractors, triggers the page-aware pipeline.
        # This is a list of strings where each element is the extracted text of one page.
        page_texts = extra.get("page_texts")

        # Page-aware pipeline config (only used when page_texts is provided)
        toc_check_page_num = extra.get("toc_check_page_num", PAGEINDEX_TOC_CHECK_PAGE_NUM)
        max_page_num_each_node = extra.get("max_page_num_each_node", PAGEINDEX_MAX_PAGE_NUM_EACH_NODE)
        max_token_num_each_node = extra.get("max_token_num_each_node", PAGEINDEX_MAX_TOKEN_NUM_EACH_NODE)

        # --- Route to the correct pipeline ---
        if page_texts and len(page_texts) >= 2:
            # PAGE-AWARE PIPELINE: Used for PDF documents where per-page text
            # arrays are available. Delegates to tree_parser() from the reference
            # PageIndex library which performs:
            #   1. ToC detection (scans first N pages for table of contents)
            #   2. ToC verification (checks if detected ToC matches actual content)
            #   3. ToC correction (fixes page numbers using LLM)
            #   4. Fallback (if no ToC found, uses LLM to infer structure)
            #   5. Large-node splitting (recursively breaks oversized nodes)
            # After tree_parser returns the structure, this pipeline adds text
            # from page ranges, trims parent text, and merges small siblings.
            doc_name = source_name or "Untitled"
            logger.info(
                f"Running PageIndex page-aware pipeline: model={model}, "
                f"pages={len(page_texts)}, summary={if_add_node_summary}"
            )

            tree_result = await md_to_tree_from_pages(
                page_texts=page_texts,
                doc_name=doc_name,
                model=model,
                if_add_node_summary=if_add_node_summary,
                if_add_node_text="yes",
                summary_token_threshold=summary_token_threshold,
                if_split_large_sections=split_large_sections,
                max_node_tokens=max_node_tokens,
                toc_check_page_num=toc_check_page_num,
                max_page_num_each_node=max_page_num_each_node,
                max_token_num_each_node=max_token_num_each_node,
            )
        else:
            # MARKDOWN PIPELINE: Used for documents that arrive as a single
            # markdown string (the typical case since all documents are parsed
            # to markdown before indexing). Pipeline steps:
            #   1. Regex-scan for # headers to build initial flat node list
            #   2. Assign text slices (each header's line to the next header's line)
            #   3. Build nested tree from header levels (stack-based)
            #   4. LLM-split oversized leaf nodes into sub-sections
            #   5. Generate per-node summaries via concurrent LLM calls
            logger.info(
                f"Running PageIndex md_to_tree: model={model}, "
                f"summary={if_add_node_summary}"
            )

            # Always request text (if_add_node_text="yes") because we need
            # section text for the SectionData objects. Summaries are
            # generated from the text before we split them apart.
            tree_result = await md_to_tree(
                md_content=content,
                doc_name=source_name or "Untitled",
                model=model,
                if_add_node_summary=if_add_node_summary,
                if_add_node_text="yes",
                if_thinning=if_thinning,
                min_token_threshold=min_token_threshold,
                summary_token_threshold=summary_token_threshold,
                if_split_large_sections=split_large_sections,
                max_node_tokens=max_node_tokens,
            )

        # --- Post-pipeline: split tree into toc_structure + sections ---

        # Both pipelines return the same format:
        # {"doc_name": "...", "structure": [...tree with text...]}
        structure = tree_result.get("structure", [])
        doc_description = tree_result.get("doc_description")

        # Prefer the explicit source_name from config over the filename-derived
        # doc_name from the pipeline (which would be the temp file name for
        # the markdown pipeline).
        doc_name = source_name or tree_result.get("doc_name")

        # THE KEY STEP: Walk the tree and pop text out of every node.
        # After this call:
        #   - 'structure' is mutated to be text-free (becomes toc_structure)
        #   - 'sections' is a flat list of SectionData with the popped text
        sections = _extract_sections_from_structure(structure)

        node_count = _count_nodes(structure)

        result = PageIndexResult(
            toc_structure=structure,
            sections=sections,
            doc_name=doc_name,
            doc_description=doc_description,
            strategy_name="page_index",
            source_id=source_id,
            indexed_at=datetime.now(),
            stats={
                "node_count": node_count,
                "section_count": len(sections),
                "total_chars": len(content),
                "model": model,
            },
        )

        logger.info(
            f"PageIndex complete: {node_count} nodes in tree, "
            f"{len(sections)} sections extracted"
        )
        return result
