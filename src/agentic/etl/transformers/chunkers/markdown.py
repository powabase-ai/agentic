"""Markdown-aware chunker"""

from typing import Dict, List

from .base import BaseChunker, Chunk
from ...registry import ChunkerFactory


@ChunkerFactory.register("markdown")
class MarkdownChunker(BaseChunker):
    """Markdown header and length splitter
    
    Reference: doc_extract/splitters.py MarkdownHeaderAndLengthSplitter for algorithm
    TODO: Port markdown header + length splitting logic from reference
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # TODO: Implement markdown splitting logic
        raise NotImplementedError("Markdown chunker not yet implemented")
    
    def chunk(self, text: str, metadata: Dict = None) -> List[Chunk]:
        """Split markdown text preserving headers"""
        # TODO: Port from doc_extract/splitters.py
        raise NotImplementedError("Markdown chunker not yet implemented")

