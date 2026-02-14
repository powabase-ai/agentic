"""
VectorSearch Retrieval Algorithm - semantic similarity search.

This algorithm:
1. Embeds the query using the same embedder as indexing
2. Performs vector similarity search against the knowledge store
3. Returns relevant chunks ranked by similarity score
"""

import logging

from agentic.knowledge.embedder.base import Embedder
from agentic.knowledge.embedder.litellm import OpenAIEmbedder
from agentic.knowledge.models import RetrievalConfig, RetrievedItem
from agentic.knowledge.retrieval.base import RetrievalAlgorithm
from agentic.knowledge.store import KnowledgeStore

logger = logging.getLogger(__name__)


class VectorSearchAlgorithm(RetrievalAlgorithm):
    """
    Vector similarity search retrieval.

    This is the standard retrieval method for RAG systems. It embeds the query
    and finds the most similar chunks using vector similarity (cosine, etc.).

    Args:
        embedder: Embedder to use for query embedding (should match indexing embedder)

    Example:
        >>> from agentic.knowledge.retrieval import VectorSearchAlgorithm
        >>> from agentic.knowledge.embedder import OpenAIEmbedder
        >>>
        >>> algo = VectorSearchAlgorithm(embedder=OpenAIEmbedder())
        >>> results = await algo.retrieve(
        ...     query="What is machine learning?",
        ...     store=my_knowledge_store,
        ...     config=RetrievalConfig(top_k=5),
        ... )
        >>> for chunk in results:
        ...     print(f"Score: {chunk.score:.3f} - {chunk.text[:100]}...")
    """

    name = "vector_search"

    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or OpenAIEmbedder()

    def retrieve(
        self,
        query: str,
        store: KnowledgeStore,
        config: RetrievalConfig | None = None,
    ) -> list[RetrievedItem]:
        """
        Retrieve relevant chunks using vector similarity search.

        Args:
            query: Search query
            store: Knowledge store to search
            config: Optional retrieval configuration

        Returns:
            List of retrieved chunks, sorted by relevance
        """
        config = config or RetrievalConfig()

        # Step 1: Embed the query
        query_embedding = self.embedder.embed(query)

        logger.debug(f"Query embedded, searching with top_k={config.top_k}")

        # Step 2: Vector search in store
        raw_results = store.vector_search(
            query_embedding=query_embedding,
            top_k=config.top_k,
            filter_metadata=config.filter_metadata,
        )

        # Step 3: Apply similarity threshold
        filtered_results = self._apply_threshold(
            raw_results, config.similarity_threshold
        )

        logger.info(
            f"Retrieved {len(filtered_results)}/{len(raw_results)} chunks "
            f"(threshold: {config.similarity_threshold})"
        )

        return filtered_results

    async def aretrieve(
        self,
        query: str,
        store: KnowledgeStore,
        config: RetrievalConfig | None = None,
    ) -> list[RetrievedItem]:
        """
        Async version of retrieve.

        Uses async embedding for better concurrency.
        """
        config = config or RetrievalConfig()

        # Step 1: Async embed the query
        query_embedding = await self.embedder.aembed(query)

        # Step 2: Async vector search
        raw_results = await store.avector_search(
            query_embedding=query_embedding,
            top_k=config.top_k,
            filter_metadata=config.filter_metadata,
        )

        # Step 3: Apply threshold
        filtered_results = self._apply_threshold(
            raw_results, config.similarity_threshold
        )

        logger.info(f"Retrieved {len(filtered_results)}/{len(raw_results)} chunks")

        return filtered_results


class HybridSearchAlgorithm(RetrievalAlgorithm):
    """
    Hybrid search combining vector similarity with keyword matching.

    This provides better results when queries contain specific keywords
    that should be matched exactly.

    Note: This is a placeholder for future implementation.
    """

    name = "hybrid_search"

    def __init__(
        self,
        embedder: Embedder | None = None,
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7,
    ):
        self.embedder = embedder or OpenAIEmbedder()
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight

    def retrieve(
        self,
        query: str,
        store: KnowledgeStore,
        config: RetrievalConfig | None = None,
    ) -> list[RetrievedItem]:
        """
        Retrieve using hybrid search.

        Currently falls back to pure vector search.
        Full hybrid implementation requires store support for BM25/FTS.
        """
        # TODO: Implement true hybrid search when stores support it
        # For now, fall back to vector search
        vector_algo = VectorSearchAlgorithm(embedder=self.embedder)
        return vector_algo.retrieve(query, store, config)

    async def aretrieve(
        self,
        query: str,
        store: KnowledgeStore,
        config: RetrievalConfig | None = None,
    ) -> list[RetrievedItem]:
        """Async hybrid search."""
        vector_algo = VectorSearchAlgorithm(embedder=self.embedder)
        return await vector_algo.aretrieve(query, store, config)
