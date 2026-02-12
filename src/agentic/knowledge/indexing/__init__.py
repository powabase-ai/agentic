"""
Indexing module - algorithms for preparing content for retrieval.

Indexing algorithms transform raw content into indexed artifacts that can
be efficiently searched at runtime. Each strategy produces different artifacts:

Built-in algorithms:
- ChunkAndEmbedAlgorithm: Standard RAG (chunks + embeddings)

Future algorithms:
- graph: Entities + relations (GraphRAG)
- summary_tree: Hierarchical summaries

Example:
    >>> from agentic.knowledge.indexing import ChunkAndEmbedAlgorithm
    >>> from agentic.knowledge.chunking import RecursiveChunking
    >>> from agentic.knowledge.embedder import OpenAIEmbedder
    >>>
    >>> indexer = ChunkAndEmbedAlgorithm(
    ...     chunker=RecursiveChunking(chunk_size=500),
    ...     embedder=OpenAIEmbedder(),
    ... )
    >>> result = indexer.index("Long document content...")
    >>> print(f"Created {len(result.chunks)} chunks with embeddings")
"""

from agentic.knowledge.indexing.base import IndexingAlgorithm
from agentic.knowledge.indexing.chunk_embed import (
    ChunkAndEmbedAlgorithm,
    ChunkData,
    ChunkEmbedResult,
)
from agentic.knowledge.indexing.page_index import (
    PageIndexAlgorithm,
    PageIndexResult,
    SectionData,
)

__all__ = [
    "IndexingAlgorithm",
    "ChunkAndEmbedAlgorithm",
    "ChunkData",
    "ChunkEmbedResult",
    "PageIndexAlgorithm",
    "PageIndexResult",
    "SectionData",
]
