"""
KnowledgeStore - abstract interface for knowledge storage operations.

This interface is implemented by platforms (e.g., agentic-platform with pgvector).
The agentic framework defines the contract; platforms provide the implementation.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic.knowledge.models import RetrievedChunk


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
    ) -> list["RetrievedChunk"]:
        """
        Search for similar chunks using vector similarity.

        Args:
            embedding: Query embedding vector
            top_k: Maximum number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of RetrievedChunk objects, sorted by similarity (highest first)
        """
        ...

    @abstractmethod
    async def full_text_search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list["RetrievedChunk"]:
        """
        Search for chunks using full-text search.

        Args:
            query: Text query string
            top_k: Maximum number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of RetrievedChunk objects, sorted by relevance
        """
        ...

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
