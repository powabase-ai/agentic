"""
Chunking module - strategies for splitting text into chunks.

Chunking strategies determine how large documents are split into smaller
pieces suitable for embedding and retrieval. Chunk sizes and overlap are
measured in **tokens** (cl100k_base encoding).

Built-in strategies:
- FixedSizeChunking: Simple token-based chunking with overlap
- RecursiveChunking: Smart chunking at natural boundaries (recommended)
- MarkdownHeaderChunking: Split by Markdown headers, then by length

Example:
    >>> from agentic.knowledge.chunking import RecursiveChunking
    >>>
    >>> chunker = RecursiveChunking(chunk_size=2000, overlap=50)
    >>> chunks = chunker.chunk("Long document text...")
    >>> for chunk in chunks:
    ...     print(f"Chunk {chunk.index}: {len(chunk.text)} chars")
"""

from agentic.knowledge.chunking.base import ChunkingStrategy
from agentic.knowledge.chunking.fixed import FixedSizeChunking
from agentic.knowledge.chunking.markdown_header import MarkdownHeaderChunking
from agentic.knowledge.chunking.recursive import RecursiveChunking
from agentic.knowledge.models import TextChunk

__all__ = [
    "ChunkingStrategy",
    "TextChunk",
    "FixedSizeChunking",
    "MarkdownHeaderChunking",
    "RecursiveChunking",
]
