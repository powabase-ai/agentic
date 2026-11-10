"""
ChunkAndEmbed Indexing Algorithm - the standard RAG indexing strategy.

This algorithm:
1. Splits content into chunks using a ChunkingStrategy
2. Generates embeddings for each chunk
3. Returns chunks with embeddings ready for vector storage
"""

import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from agentic.knowledge.chunking.base import ChunkingStrategy
from agentic.knowledge.chunking.recursive import RecursiveChunking
from agentic.knowledge.embedder.base import Embedder
from agentic.knowledge.embedder.litellm import OpenAIEmbedder
from agentic.knowledge.indexing.base import IndexingAlgorithm
from agentic.knowledge.models import IndexingConfig, TextChunk

logger = logging.getLogger(__name__)

# Below this share of the input's non-whitespace text, a chunker has probably
# dropped content (e.g. #46). Not exactly 1.0 even when nothing is lost:
# MarkdownHeaderChunking strips header lines and horizontal rules by design.
COVERAGE_WARNING_THRESHOLD = 0.9

_WHITESPACE = re.compile(r"\s+")


def _non_whitespace_len(text: str) -> int:
    return len(_WHITESPACE.sub("", text))


def chunk_coverage_ratio(content: str, chunks: Sequence[TextChunk]) -> float:
    """Share of the non-whitespace characters of ``content`` inside some chunk's span.

    Spans are ``[start_char, end_char)`` offsets into ``content``; overlapping
    spans are merged so overlap is not counted twice. Content with no
    non-whitespace text counts as fully covered.
    """
    total = _non_whitespace_len(content)
    if total == 0:
        return 1.0

    covered = 0
    span_start = span_end = 0
    for start, end in sorted((c.start_char, c.end_char) for c in chunks):
        if start > span_end:
            covered += _non_whitespace_len(content[span_start:span_end])
            span_start = start
        span_end = max(span_end, end)
    covered += _non_whitespace_len(content[span_start:span_end])

    return min(covered / total, 1.0)


def _coverage_stats(
    content: str, chunks: Sequence[TextChunk], source_id: str | None
) -> dict:
    ratio = chunk_coverage_ratio(content, chunks)
    if ratio < COVERAGE_WARNING_THRESHOLD:
        logger.warning(
            "chunk_coverage_low source_id=%s coverage_ratio=%.3f chunks=%d "
            "content_chars=%d max_end_char=%d",
            source_id,
            ratio,
            len(chunks),
            len(content),
            max((c.end_char for c in chunks), default=0),
        )
    return {"coverage_ratio": round(ratio, 4)}


class ChunkData(TextChunk):
    """
    Extended chunk with embedding data.

    Inherits from TextChunk (Pydantic model) and adds the embedding vector.
    """

    embedding: list[float] | None = None
    tokens: int | None = None


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
        stats = _coverage_stats(content, text_chunks, source_id)

        if not text_chunks:
            logger.warning("No chunks generated from content")
            return ChunkEmbedResult(chunks=[], source_id=source_id, stats=stats)

        logger.debug(f"Generated {len(text_chunks)} chunks")

        # Step 2: Generate embeddings
        chunk_texts = [chunk.text for chunk in text_chunks]
        embeddings = self.embedder.embed_batch(chunk_texts)

        # Step 3: Combine chunks with embeddings
        from agentic.knowledge.chunking.token_utils import count_tokens

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
                tokens=count_tokens(chunk.text),
            )
            chunk_data_list.append(chunk_data)

        logger.info(
            f"Indexed content: {len(chunk_data_list)} chunks, "
            f"embedding dim: {len(embeddings[0]) if embeddings else 0}"
        )

        return ChunkEmbedResult(
            chunks=chunk_data_list,
            source_id=source_id,
            stats=stats,
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
        t_chunk = time.monotonic()
        text_chunks = self.chunker.chunk(content, source_id=source_id)
        chunk_secs = time.monotonic() - t_chunk
        stats = _coverage_stats(content, text_chunks, source_id)

        if not text_chunks:
            logger.info(
                "chunk_embed_timing chunk_secs=%.2f embed_secs=0 chunks=0",
                chunk_secs,
            )
            return ChunkEmbedResult(chunks=[], source_id=source_id, stats=stats)

        # Step 2: Async embed
        t_embed = time.monotonic()
        chunk_texts = [chunk.text for chunk in text_chunks]
        embeddings = await self.embedder.aembed_batch(chunk_texts)
        embed_secs = time.monotonic() - t_embed

        logger.info(
            "chunk_embed_timing chunk_secs=%.2f embed_secs=%.2f chunks=%d",
            chunk_secs,
            embed_secs,
            len(text_chunks),
        )

        # Step 3: Combine
        from agentic.knowledge.chunking.token_utils import count_tokens

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
                tokens=count_tokens(chunk.text),
            )
            chunk_data_list.append(chunk_data)

        return ChunkEmbedResult(
            chunks=chunk_data_list,
            source_id=source_id,
            stats=stats,
        )
