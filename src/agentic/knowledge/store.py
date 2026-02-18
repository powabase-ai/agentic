"""
KnowledgeStore - abstract interface for knowledge storage operations.

This interface is implemented by platforms (e.g., agentic-platform with pgvector).
The agentic framework defines the contract; platforms provide the implementation.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic.knowledge.models import RetrievedItem


class KnowledgeStore(ABC):
    """
    Abstract interface for knowledge storage operations.

    This interface defines how knowledge artifacts are stored and retrieved.
    Platform implementations (e.g., PgVectorKnowledgeStore) provide the
    actual storage logic.

    The separation allows the agentic framework to define algorithms without
    being coupled to specific storage backends.

    Example Implementation:
        >>> class PgVectorKnowledgeStore(KnowledgeStore):
        ...     def __init__(self, db_session, tenant_schema: str, kb_id: str):
        ...         self.session = db_session
        ...         self.schema = tenant_schema
        ...         self.kb_id = kb_id
        ...
        ...     async def vector_search(self, embedding, top_k):
        ...         # SELECT * FROM {schema}.chunks
        ...         # WHERE knowledge_base_id = {kb_id}
        ...         # ORDER BY embedding <=> {embedding}
        ...         # LIMIT {top_k}
        ...         ...
    """

    @abstractmethod
    async def vector_search(
        self,
        embedding: list[float],
        top_k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list["RetrievedItem"]:
        """
        Search for similar chunks using vector similarity.

        Args:
            embedding: Query embedding vector
            top_k: Maximum number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of RetrievedItem objects, sorted by similarity (highest first)
        """
        ...

    @abstractmethod
    async def full_text_search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list["RetrievedItem"]:
        """
        Search for chunks using full-text search.

        Args:
            query: Text query string
            top_k: Maximum number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of RetrievedItem objects, sorted by relevance
        """
        ...

    async def hybrid_search(
        self,
        query: str,
        embedding: list[float],
        top_k: int = 5,
        vector_weight: float = 0.7,
        filter_metadata: dict | None = None,
    ) -> list["RetrievedItem"]:
        """
        Combine vector similarity and full-text search using Reciprocal Rank Fusion.

        Default implementation runs both searches and fuses with RRF.
        Subclasses may override for optimized single-query implementations.

        Args:
            query: Text query for full-text search
            embedding: Query embedding vector for vector search
            top_k: Maximum number of results to return
            vector_weight: Weight for vector results in RRF (keyword weight = 1 - vector_weight)
            filter_metadata: Optional metadata filters

        Returns:
            List of RetrievedItem objects, sorted by fused relevance
        """
        from agentic.knowledge.retrieval.fusion import reciprocal_rank_fusion

        fetch_count = top_k * 2
        vector_results = await self.vector_search(
            embedding=embedding,
            top_k=fetch_count,
            filter_metadata=filter_metadata,
        )
        text_results = await self.full_text_search(
            query=query,
            top_k=fetch_count,
            filter_metadata=filter_metadata,
        )

        keyword_weight = 1.0 - vector_weight
        return reciprocal_rank_fusion(
            result_lists=[vector_results, text_results],
            weights=[vector_weight, keyword_weight],
            top_k=top_k,
        )

    @abstractmethod
    async def store_chunks(
        self,
        indexed_source_id: str,
        chunks: list[dict],
    ) -> int:
        """
        Store indexed chunks in the knowledge base.

        Args:
            indexed_source_id: ID of the IndexedSource record
            chunks: List of chunk data dicts with text, embedding, metadata

        Returns:
            Number of chunks stored
        """
        ...

    @abstractmethod
    async def delete_chunks(
        self,
        indexed_source_id: str,
    ) -> int:
        """
        Delete all chunks for an indexed source.

        Args:
            indexed_source_id: ID of the IndexedSource record

        Returns:
            Number of chunks deleted
        """
        ...

    @abstractmethod
    async def get_chunk_count(
        self,
        knowledge_base_id: str | None = None,
        source_id: str | None = None,
    ) -> int:
        """
        Count chunks in the store.

        Args:
            knowledge_base_id: Filter by knowledge base (optional)
            source_id: Filter by source (optional)

        Returns:
            Number of chunks matching the filters
        """
        ...
