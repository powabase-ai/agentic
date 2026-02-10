"""
PageIndex indexing algorithm.

Builds a hierarchical tree structure from markdown content using
LLM-powered analysis. The tree can then be used for tree-based
retrieval (tree_search) where an LLM reasons over the structure
to find relevant sections.
"""

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agentic.knowledge.indexing.base import IndexingAlgorithm
from agentic.knowledge.models import IndexingConfig, IndexResult

logger = logging.getLogger(__name__)


@dataclass
class PageIndexResult(IndexResult):
    """Result of PageIndex indexing - contains a tree structure.

    Attributes:
        tree_data: Full tree dict (titles, node_ids, summaries, text)
        doc_name: Name of the source document
        doc_description: LLM-generated document description (if enabled)
        strategy_name: Always "page_index"
        stats: Indexing statistics (node_count, etc.)
    """

    tree_data: dict = field(default_factory=dict)
    doc_name: str | None = None
    doc_description: str | None = None
    strategy_name: str = "page_index"
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def node_count(self) -> int:
        """Count total nodes in the tree."""

        def _count(nodes):
            count = 0
            if isinstance(nodes, list):
                for node in nodes:
                    count += 1
                    if "nodes" in node:
                        count += _count(node["nodes"])
            return count

        structure = self.tree_data.get("structure", [])
        return _count(structure)


class PageIndexAlgorithm(IndexingAlgorithm):
    """Indexing algorithm that builds a hierarchical tree from markdown.

    Uses the vendored PageIndex library to parse markdown headers into
    a tree structure with optional node summaries and text content.

    The resulting tree is stored as JSONB and used by TreeSearchAlgorithm
    for LLM-reasoning-based retrieval.
    """

    name = "page_index"

    def index(
        self,
        content: str,
        config: IndexingConfig,
        source_id: str | None = None,
    ) -> PageIndexResult:
        """Synchronous indexing - delegates to async."""
        return asyncio.run(self.aindex(content, config, source_id))

    async def aindex(
        self,
        content: str,
        config: IndexingConfig,
        source_id: str | None = None,
    ) -> PageIndexResult:
        """Build a tree structure from markdown content.

        Args:
            content: Markdown text content
            config: IndexingConfig with extra options:
                - model: LLM model for tree building (default: "gpt-4o-2024-11-20")
                - if_add_node_summary: "yes"/"no" (default: "yes")
                - if_add_node_text: "yes"/"no" (default: "yes")
                - if_thinning: bool (default: False)
                - min_token_threshold: int (default: 5000)
                - summary_token_threshold: int (default: 200)
            source_id: Optional source identifier

        Returns:
            PageIndexResult with the tree data
        """
        from agentic.knowledge.indexing._pageindex_lib.page_index_md import (
            md_to_tree,
        )

        extra = config.extra or {}
        model = extra.get("model", "gpt-4o-2024-11-20")
        if_add_node_summary = extra.get("if_add_node_summary", "yes")
        if_add_node_text = extra.get("if_add_node_text", "yes")
        if_thinning = extra.get("if_thinning", False)
        min_token_threshold = extra.get("min_token_threshold", 5000)
        summary_token_threshold = extra.get("summary_token_threshold", 200)

        # Write content to a temp .md file (md_to_tree expects a file path)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".md")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info(
                f"Running PageIndex md_to_tree: model={model}, "
                f"summary={if_add_node_summary}, text={if_add_node_text}"
            )

            tree_result = await md_to_tree(
                md_path=tmp_path,
                model=model,
                if_add_node_summary=if_add_node_summary,
                if_add_node_text=if_add_node_text,
                if_thinning=if_thinning,
                min_token_threshold=min_token_threshold,
                summary_token_threshold=summary_token_threshold,
            )
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        doc_name = tree_result.get("doc_name")
        doc_description = tree_result.get("doc_description")
        structure = tree_result.get("structure", [])

        # Count nodes
        def _count_nodes(nodes):
            count = 0
            if isinstance(nodes, list):
                for node in nodes:
                    count += 1
                    if "nodes" in node:
                        count += _count_nodes(node["nodes"])
            return count

        node_count = _count_nodes(structure)

        result = PageIndexResult(
            tree_data=tree_result,
            doc_name=doc_name,
            doc_description=doc_description,
            strategy_name="page_index",
            source_id=source_id,
            indexed_at=datetime.now(),
            stats={
                "node_count": node_count,
                "total_chars": len(content),
                "model": model,
            },
        )

        logger.info(f"PageIndex complete: {node_count} nodes in tree")
        return result
