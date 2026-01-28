"""
Chunking module - strategies for splitting text into chunks.

Chunking strategies determine how large documents are split into smaller
pieces suitable for embedding and retrieval.

Built-in strategies:
- FixedSizeChunking: Simple character-based chunking with overlap
- RecursiveChunking: Smart chunking at natural boundaries (recommended)

Example:
    >>> from agentic.knowledge.chunking import RecursiveChunking
    >>>
    >>> chunker = RecursiveChunking(chunk_size=500, overlap=50)
    >>> chunks = chunker.chunk("Long document text...")
    >>> for chunk in chunks:
    ...     print(f"Chunk {chunk.index}: {len(chunk.text)} chars")
"""

from agentic.knowledge.chunking.base import ChunkingStrategy
from agentic.knowledge.chunking.fixed import FixedSizeChunking
from agentic.knowledge.chunking.recursive import RecursiveChunking
from agentic.knowledge.models import TextChunk

__all__ = [
    "ChunkingStrategy",
    "TextChunk",
    "FixedSizeChunking",
    "RecursiveChunking",
]
