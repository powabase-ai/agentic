"""
RetrievalAlgorithm - base class for retrieval methods.

Retrieval algorithms query the knowledge store and return relevant context.
They handle query processing, store querying, and result post-processing.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic.knowledge.models import RetrievalConfig, RetrievedItem
    from agentic.knowledge.store import KnowledgeStore


class RetrievalAlgorithm(ABC):
    """
    Base class for retrieval algorithms.

    A RetrievalAlgorithm queries the knowledge store and returns relevant
    chunks/context for a given query. It handles:

    1. Query processing (e.g., embedding the query)
    2. Store interaction (via KnowledgeStore interface)
    3. Result post-processing (filtering, reranking, etc.)

    Key Design Principles:
        1. **Async-first**: retrieve() is async for non-blocking I/O
        2. **Store agnostic**: Uses KnowledgeStore interface
        3. **Configurable**: Behavior controlled via RetrievalConfig

    Subclasses:
        - VectorSearchAlgorithm: Semantic similarity search
        - HybridSearchAlgorithm: Vector + full-text with fusion (future)
        - GraphTraversalAlgorithm: Entity graph navigation (future)

    Example:
        >>> class VectorSearchAlgorithm(RetrievalAlgorithm):
        ...     name = "vector_search"
        ...
        ...     async def retrieve(self, query, store, config):
        ...         embedding = await self.embedder.aembed(query)
        ...         results = await store.vector_search(embedding, config.top_k)
        ...         return self._apply_threshold(results, config.similarity_threshold)
    """

    # Subclasses must define this
    name: str = "base"

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        store: "KnowledgeStore",
        config: "RetrievalConfig",
    ) -> list["RetrievedItem"]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: The user's query string
            store: KnowledgeStore implementation to query
            config: Configuration for the retrieval operation

        Returns:
            List of RetrievedItem objects, sorted by relevance

        Example:
            >>> chunks = await retriever.retrieve(
            ...     query="What is the return policy?",
            ...     store=pg_vector_store,
            ...     config=RetrievalConfig(top_k=5, similarity_threshold=0.7),
            ... )
            >>> for chunk in chunks:
            ...     print(f"[{chunk.score:.2f}] {chunk.text[:100]}...")
        """
        ...

    def _apply_threshold(
        self,
        chunks: list["RetrievedItem"],
        threshold: float,
    ) -> list["RetrievedItem"]:
        """
        Filter chunks below similarity threshold.

        Helper method for subclasses to filter results.

        Args:
            chunks: List of retrieved chunks
            threshold: Minimum similarity score (0.0 to 1.0)

        Returns:
            Filtered list of chunks above threshold
        """
        if threshold <= 0:
            return chunks
        return [c for c in chunks if c.score >= threshold]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
