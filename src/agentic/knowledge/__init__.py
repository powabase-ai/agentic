"""
Knowledge module - abstractions for indexing content and retrieving context.

The Knowledge module provides a clean separation between:
- **Indexing**: How content is prepared for retrieval (chunking, embedding, etc.)
- **Retrieval**: How content is queried at runtime (vector search, hybrid, etc.)

Core Abstractions:
    - IndexingAlgorithm: Base class for indexing strategies
    - RetrievalAlgorithm: Base class for retrieval methods
    - ChunkingStrategy: Base class for text chunking
    - Embedder: Base class for embedding providers
    - KnowledgeStore: Interface for storage (implemented by platforms)

Built-in Implementations:
    - ChunkAndEmbedAlgorithm: Standard RAG indexing
    - VectorSearchAlgorithm: Semantic similarity retrieval
    - RecursiveChunking: Smart text chunking
    - MarkdownHeaderChunking: Markdown header + length chunking
    - OpenAIEmbedder: OpenAI embeddings via LiteLLM

Example:
    >>> from agentic.knowledge import (
    ...     ChunkAndEmbedAlgorithm,
    ...     VectorSearchAlgorithm,
    ...     RecursiveChunking,
    ...     OpenAIEmbedder,
    ... )
    >>>
    >>> # Index content
    >>> indexer = ChunkAndEmbedAlgorithm(
    ...     chunker=RecursiveChunking(chunk_size=500),
    ...     embedder=OpenAIEmbedder(),
    ... )
    >>> result = indexer.index("Long document content...")
    >>>
    >>> # Retrieve at runtime
    >>> retriever = VectorSearchAlgorithm(embedder=OpenAIEmbedder())
    >>> chunks = await retriever.aretrieve(query, store, config)

See Also:
    - docs/CONCEPTS.md for detailed explanations
    - agentic-platform for KnowledgeStore implementations
"""

# Base classes
# Chunking
from agentic.knowledge.chunking import (
    ChunkingStrategy,
    FixedSizeChunking,
    MarkdownHeaderChunking,
    RecursiveChunking,
)

# Embedder
from agentic.knowledge.embedder import (
    CohereEmbedder,
    Embedder,
    LiteLLMEmbedder,
    OpenAIEmbedder,
    VoyageEmbedder,
)

# Reranker
from agentic.knowledge.reranker import (
    CohereReranker,
    JinaReranker,
    LiteLLMReranker,
    Reranker,
    RerankResult,
    VoyageReranker,
    ZeroEntropyReranker,
)

# Indexing
from agentic.knowledge.indexing import (
    ChunkAndEmbedAlgorithm,
    ChunkData,
    ChunkEmbedResult,
    IndexingAlgorithm,
)
from agentic.knowledge.models import (
    IndexingConfig,
    IndexResult,
    RerankerConfig,
    RetrievalConfig,
    RetrievedItem,
    TextChunk,
)

# Retrieval
from agentic.knowledge.retrieval import (
    FullTextSearchAlgorithm,
    HybridSearchAlgorithm,
    RetrievalAlgorithm,
    VectorSearchAlgorithm,
    reciprocal_rank_fusion,
)
from agentic.knowledge.store import KnowledgeStore

__all__ = [
    # Store interface
    "KnowledgeStore",
    # Models
    "IndexResult",
    "RetrievedItem",
    "IndexingConfig",
    "RetrievalConfig",
    "RerankerConfig",
    "TextChunk",
    # Indexing
    "IndexingAlgorithm",
    "ChunkAndEmbedAlgorithm",
    "ChunkData",
    "ChunkEmbedResult",
    # Retrieval
    "RetrievalAlgorithm",
    "VectorSearchAlgorithm",
    "FullTextSearchAlgorithm",
    "HybridSearchAlgorithm",
    "reciprocal_rank_fusion",
    # Chunking
    "ChunkingStrategy",
    "FixedSizeChunking",
    "MarkdownHeaderChunking",
    "RecursiveChunking",
    # Embedders
    "Embedder",
    "LiteLLMEmbedder",
    "OpenAIEmbedder",
    "CohereEmbedder",
    "VoyageEmbedder",
    # Rerankers
    "Reranker",
    "RerankResult",
    "LiteLLMReranker",
    "CohereReranker",
    "VoyageReranker",
    "JinaReranker",
    "ZeroEntropyReranker",
]
