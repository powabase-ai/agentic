"""
IndexingAlgorithm - base class for indexing strategies.

Indexing algorithms transform content into indexed artifacts (chunks, entities, etc.)
without performing storage operations. The caller is responsible for storing
the results.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic.knowledge.models import IndexingConfig, IndexResult


class IndexingAlgorithm(ABC):
    """
    Base class for indexing algorithms.

    An IndexingAlgorithm transforms raw content into indexed artifacts
    suitable for retrieval. It returns pure data without performing
    any storage operations.

    Key Design Principles:
        1. **Pure functions**: index() returns data, doesn't write to DB
        2. **Strategy pattern**: Different algorithms for different use cases
        3. **Configurable**: Behavior controlled via IndexingConfig

    Subclasses:
        - ChunkAndEmbedAlgorithm: Standard RAG chunking + embedding
        - GraphIndexAlgorithm: Entity extraction + relationships (future)
        - SummaryTreeAlgorithm: Hierarchical summaries (future)

    Example:
        >>> class ChunkAndEmbedAlgorithm(IndexingAlgorithm):
        ...     name = "chunk_embed"
        ...
        ...     def index(self, content: str, config: IndexingConfig) -> ChunkEmbedResult:
        ...         chunks = self.chunker.chunk(content, config.chunk_size)
        ...         for chunk in chunks:
        ...             chunk.embedding = self.embedder.embed(chunk.text)
        ...         return ChunkEmbedResult(chunks=chunks)
    """

    # Subclasses must define this
    name: str = "base"

    @abstractmethod
    def index(self, content: str, config: "IndexingConfig") -> "IndexResult":
        """
        Index content and return the result.

        This method transforms raw content into indexed artifacts without
        performing any storage operations. The caller (typically the host
        application) is responsible for storing the results.

        Args:
            content: The text content to index
            config: Configuration for the indexing operation

        Returns:
            IndexResult subclass with strategy-specific artifacts
            (e.g., ChunkEmbedResult with chunks)

        Example:
            >>> result = algorithm.index("Long document text...", config)
            >>> # Caller stores: store.store_chunks(result.chunks)
        """
        ...

    async def aindex(self, content: str, config: "IndexingConfig") -> "IndexResult":
        """
        Async version of index().

        Default implementation calls sync index(). Override for true async
        implementations (e.g., when using async embedding APIs).

        Args:
            content: The text content to index
            config: Configuration for the indexing operation

        Returns:
            IndexResult subclass with strategy-specific artifacts
        """
        return self.index(content, config)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
