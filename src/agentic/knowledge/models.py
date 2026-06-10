"""
Shared models for the knowledge module.

These models are used across indexing, retrieval, and storage operations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from agentic.knowledge.model_config import (
    CHUNK_EMBED_DEFAULT_CHUNK_SIZE,
    CHUNK_EMBED_DEFAULT_OVERLAP,
    CHUNK_EMBED_EMBEDDING_MODEL,
    RERANKER_CANDIDATE_COUNT,
    RERANKER_DEFAULT_MODEL,
)


class TextChunk(BaseModel):
    """
    A chunk of text produced by a chunking strategy.

    Attributes:
        text: The text content of this chunk
        index: Position of this chunk in the sequence (0-indexed)
        start_char: Starting character offset in the original text
        end_char: Ending character offset in the original text
        source_id: Optional identifier of the source document
        metadata: Additional metadata (e.g., page number, section)
    """

    text: str
    index: int
    start_char: int
    end_char: int
    source_id: str | None = None
    metadata: dict[str, Any] = {}

    def __len__(self) -> int:
        """Return the length of the text in characters."""
        return len(self.text)

    def __repr__(self) -> str:
        text_preview = self.text[:30] + "..." if len(self.text) > 30 else self.text
        return f"TextChunk(index={self.index}, len={len(self.text)}, text={text_preview!r})"


@dataclass
class IndexingConfig:
    """Configuration for indexing operations.

    Attributes:
        strategy: Name of the indexing strategy (e.g., "chunk_embed", "graph")
        chunk_size: Target size for text chunks (in characters or tokens)
        overlap: Overlap between consecutive chunks
        embedding_model: Model identifier for embeddings (e.g., "text-embedding-3-small")
        extra: Additional strategy-specific configuration
    """

    strategy: str = "chunk_embed"
    chunk_size: int = CHUNK_EMBED_DEFAULT_CHUNK_SIZE
    overlap: int = CHUNK_EMBED_DEFAULT_OVERLAP
    embedding_model: str = CHUNK_EMBED_EMBEDDING_MODEL
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankerConfig:
    """Configuration for a reranking step after initial retrieval.

    Attributes:
        model: Reranker model identifier (e.g., "cohere/rerank-english-v3.0")
        candidate_count: Number of candidates to fetch in Stage 1 before reranking
        api_key: Optional API key override
        api_base: Optional custom API endpoint (for self-hosted VLLM, etc.)
        provider_params: Additional provider-specific parameters
    """

    model: str = RERANKER_DEFAULT_MODEL
    candidate_count: int = RERANKER_CANDIDATE_COUNT
    api_key: str | None = None
    api_base: str | None = None
    provider_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalConfig:
    """Configuration for retrieval operations.

    Attributes:
        method: Name of the retrieval method (e.g., "vector_search", "hybrid")
        top_k: Maximum number of results to return
        similarity_threshold: Minimum similarity score (0.0 to 1.0)
        filter_metadata: Optional metadata filters for narrowing results
        reranker: Optional reranker config for post-retrieval re-scoring
        min_per_source: Diversity floor — guarantee each matched source contributes
            at least this many items before any source gets extra (None/0 disables)
        max_per_source: Hard cap — at most this many items from any one source
            (None/0 disables)
        extra: Additional method-specific configuration
    """

    method: str = "vector_search"
    top_k: int = 5
    similarity_threshold: float = 0.0
    filter_metadata: dict[str, Any] | None = None
    reranker: RerankerConfig | None = None
    min_per_source: int | None = None
    max_per_source: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexResult:
    """Base result from an indexing operation.

    Each indexing strategy returns a specialized subclass with
    strategy-specific artifacts (e.g., ChunkEmbedResult with chunks).

    Attributes:
        strategy_name: Name of the indexing strategy used
        source_id: Identifier of the source that was indexed
        indexed_at: Timestamp of indexing
        stats: Statistics about the indexing operation
    """

    strategy_name: str
    source_id: str | None = None
    indexed_at: datetime = field(default_factory=datetime.now)
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedItem:
    """An item retrieved from a knowledge base.

    Attributes:
        item_id: Unique identifier for this item
        text: The text content of the item
        score: Relevance/similarity score (higher is better)
        source_id: Identifier of the source this item came from
        knowledge_base_id: Identifier of the knowledge base
        meta: Additional metadata (page number, section, etc.)
    """

    item_id: str
    text: str
    score: float
    source_id: str | None = None
    knowledge_base_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"RetrievedItem(score={self.score:.3f}, text={text_preview!r})"
