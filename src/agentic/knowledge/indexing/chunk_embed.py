"""
ChunkAndEmbed Indexing Algorithm - the standard RAG indexing strategy.

This algorithm:
1. Splits content into chunks using a ChunkingStrategy
2. Generates embeddings for each chunk
3. Returns chunks with embeddings ready for vector storage
"""

import logging
from dataclasses import dataclass, field

from agentic.knowledge.chunking.base import ChunkingStrategy
from agentic.knowledge.chunking.recursive import RecursiveChunking
from agentic.knowledge.embedder.base import Embedder
from agentic.knowledge.embedder.litellm import OpenAIEmbedder
from agentic.knowledge.indexing.base import IndexingAlgorithm
from agentic.knowledge.models import IndexingConfig, IndexResult, TextChunk

logger = logging.getLogger(__name__)


class ChunkData(TextChunk):
    """
    Extended chunk with embedding data.

    Inherits from TextChunk (Pydantic model) and adds the embedding vector.
    """

    embedding: Optional[list[float]] = None
    tokens: Optional[int] = None


@dataclass
class ChunkEmbedResult:
    """
    Result of ChunkAndEmbed indexing.

    Contains the list of chunks with their embeddings.
    Implements the same interface as IndexResult but standalone to avoid
    dataclass inheritance issues.
    """

    chunks: list[ChunkData] = field(default_factory=list)
    source_id: str | None = None
    strategy_name: str = "chunk_embed"
    stats: dict = field(default_factory=dict)

    @property
    def artifact_count(self) -> int:
        """Number of chunks created."""
        return len(self.chunks)


class ChunkAndEmbedAlgorithm(IndexingAlgorithm):
    """
    Standard RAG indexing: chunk + embed.

    This is the most common indexing approach for RAG systems.
    It splits documents into chunks and generates embeddings for each.

    Args:
        chunker: Chunking strategy to use (default: RecursiveChunking)
        embedder: Embedder to use (default: OpenAIEmbedder)

    Example:
        >>> from agentic.knowledge.indexing import ChunkAndEmbedAlgorithm
        >>> from agentic.knowledge.chunking import RecursiveChunking
        >>> from agentic.knowledge.embedder import OpenAIEmbedder
        >>>
        >>> algo = ChunkAndEmbedAlgorithm(
        ...     chunker=RecursiveChunking(chunk_size=500),
        ...     embedder=OpenAIEmbedder(),
        ... )
        >>> result = algo.index("Long document content...")
        >>> print(f"Created {len(result.chunks)} chunks")
    """

    name = "chunk_embed"

    def __init__(
        self,
        chunker: ChunkingStrategy | None = None,
        embedder: Embedder | None = None,
    ):
        self.chunker = chunker or RecursiveChunking()
        self.embedder = embedder or OpenAIEmbedder()

    def index(
        self,
        content: str,
        config: IndexingConfig | None = None,
        source_id: str | None = None,
    ) -> ChunkEmbedResult:
        """
        Index content by chunking and embedding.

        Args:
            content: Text content to index
            config: Optional indexing configuration
            source_id: Optional source identifier for metadata

        Returns:
            ChunkEmbedResult with chunks and embeddings
        """
        config = config or IndexingConfig()

        # Apply config to chunker if needed
        if config.chunk_size and hasattr(self.chunker, "chunk_size"):
            self.chunker.chunk_size = config.chunk_size
        if config.overlap is not None and hasattr(self.chunker, "overlap"):
            self.chunker.overlap = config.overlap

        # Step 1: Chunk the content
        text_chunks = self.chunker.chunk(content, source_id=source_id)

        if not text_chunks:
            logger.warning("No chunks generated from content")
            return ChunkEmbedResult(chunks=[], source_id=source_id)

        logger.debug(f"Generated {len(text_chunks)} chunks")

        # Step 2: Generate embeddings
        chunk_texts = [chunk.text for chunk in text_chunks]
        embeddings = self.embedder.embed_batch(chunk_texts)

        # Step 3: Combine chunks with embeddings
        chunk_data_list: list[ChunkData] = []
        for chunk, embedding in zip(text_chunks, embeddings, strict=True):
            chunk_data = ChunkData(
                text=chunk.text,
                index=chunk.index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                source_id=source_id,
                metadata=chunk.metadata,
                embedding=embedding,
            )
            chunk_data_list.append(chunk_data)

        logger.info(
            f"Indexed content: {len(chunk_data_list)} chunks, "
            f"embedding dim: {len(embeddings[0]) if embeddings else 0}"
        )

        return ChunkEmbedResult(
            chunks=chunk_data_list,
            source_id=source_id,
        )

    async def aindex(
        self,
        content: str,
        config: IndexingConfig | None = None,
        source_id: str | None = None,
    ) -> ChunkEmbedResult:
        """
        Async version of index.

        Uses async embedding for better concurrency.
        """
        config = config or IndexingConfig()

        # Apply config to chunker if needed
        if config.chunk_size and hasattr(self.chunker, "chunk_size"):
            self.chunker.chunk_size = config.chunk_size
        if config.overlap is not None and hasattr(self.chunker, "overlap"):
            self.chunker.overlap = config.overlap

        # Step 1: Chunk (sync - usually fast)
        text_chunks = self.chunker.chunk(content, source_id=source_id)

        if not text_chunks:
            return ChunkEmbedResult(chunks=[], source_id=source_id)

        # Step 2: Async embed
        chunk_texts = [chunk.text for chunk in text_chunks]
        embeddings = await self.embedder.aembed_batch(chunk_texts)

        # Step 3: Combine
        chunk_data_list: list[ChunkData] = []
        for chunk, embedding in zip(text_chunks, embeddings, strict=True):
            chunk_data = ChunkData(
                text=chunk.text,
                index=chunk.index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                source_id=source_id,
                metadata=chunk.metadata,
                embedding=embedding,
            )
            chunk_data_list.append(chunk_data)

        return ChunkEmbedResult(
            chunks=chunk_data_list,
            source_id=source_id,
        )
