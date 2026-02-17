"""Vendored PageIndex library for tree-based document indexing.

This is a vendored copy of the PageIndex library (https://github.com/...)
providing tree-based document structure extraction.

Key exports:
- md_to_tree: Build a hierarchical tree from markdown content
- md_to_tree_from_pages: Build a hierarchical tree from per-page text (PDF pipeline)
- page_index: Build a hierarchical tree from PDF documents (future use)
"""

from .page_index import *  # noqa: F401, F403
from .page_index_md import md_to_tree, md_to_tree_from_pages

__all__ = ["md_to_tree", "md_to_tree_from_pages"]
