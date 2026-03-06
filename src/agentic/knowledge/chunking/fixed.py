"""
FixedSizeChunking - split text into fixed-size chunks with overlap.

This is the simplest chunking strategy, useful when you don't need
to respect document structure. Good for general-purpose RAG.

Adapted from agno's FixedSizeChunking pattern.
"""

import re

from agentic.knowledge.chunking.base import ChunkingStrategy
from agentic.knowledge.model_config import (
    CHUNK_EMBED_DEFAULT_CHUNK_SIZE,
    CHUNK_EMBED_DEFAULT_OVERLAP,
)
from agentic.knowledge.models import TextChunk


class FixedSizeChunking(ChunkingStrategy):
    """
    Chunking strategy that splits text into fixed-size chunks with optional overlap.

    This strategy:
    - Splits by token count
    - Avoids splitting words in the middle
    - Supports overlap for context continuity

    Args:
        chunk_size: Maximum tokens per chunk (default: 2000)
        overlap: Number of overlapping tokens between chunks (default: 50)

    Example:
        >>> chunker = FixedSizeChunking(chunk_size=2000, overlap=50)
        >>> chunks = chunker.chunk("Long document text...")
        >>> for chunk in chunks:
        ...     print(f"Chunk {chunk.index}: {len(chunk.text)} chars")
    """

    name = "fixed_size"

    def __init__(
        self,
        chunk_size: int = CHUNK_EMBED_DEFAULT_CHUNK_SIZE,
        overlap: int = CHUNK_EMBED_DEFAULT_OVERLAP,
    ):
        if overlap >= chunk_size:
            raise ValueError(
                f"Overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            )

        super().__init__(chunk_size=chunk_size, overlap=overlap)

    def chunk(
        self,
        text: str,
        source_id: str | None = None,
    ) -> list[TextChunk]:
        """
        Split text into fixed-size chunks.

        Chunk size and overlap are measured in tokens. Word-boundary
        adjustment happens in character space and is snapped back to
        the nearest token boundary.

        Args:
            text: The text content to chunk
            source_id: Optional source identifier for chunk metadata

        Returns:
            List of TextChunk objects
        """
        # Clean text
        content = self._clean_text(text)
        if not content:
            return []

        from agentic.knowledge.chunking.token_utils import (
            build_token_char_map,
            char_to_token_pos,
        )

        char_map = build_token_char_map(content)
        total_tokens = len(char_map) - 1

        if total_tokens <= self.chunk_size:
            return [
                TextChunk(
                    text=content,
                    index=0,
                    start_char=0,
                    end_char=len(content),
                    source_id=source_id,
                    metadata={"strategy": self.name, "chunk_size": total_tokens},
                )
            ]

        chunks: list[TextChunk] = []
        chunk_index = 0
        token_start = 0

        while token_start + self.overlap < total_tokens:
            token_end = min(token_start + self.chunk_size, total_tokens)
            char_start = char_map[token_start]
            char_end = char_map[token_end]

            # Avoid splitting words - find last whitespace in char space
            if token_end < total_tokens:
                original_char_end = char_end
                while char_end > char_start and content[char_end] not in " \n\r\t":
                    char_end -= 1

                if char_end == char_start:
                    char_end = original_char_end

                # Snap adjusted position back to token boundary
                token_end = char_to_token_pos(char_map, char_end)
                if token_end <= token_start:
                    token_end = min(
                        token_start + max(1, self.chunk_size // 10), total_tokens
                    )
                char_end = char_map[token_end]

            chunk_text = content[char_start:char_end].strip()

            if chunk_text:
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        index=chunk_index,
                        start_char=char_start,
                        end_char=char_end,
                        source_id=source_id,
                        metadata={
                            "strategy": self.name,
                            "chunk_size": token_end - token_start,
                        },
                    )
                )
                chunk_index += 1

            # Move to next position with overlap
            new_token_start = max(token_start + 1, token_end - self.overlap)
            token_start = new_token_start

        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean text by normalizing whitespace."""
        # Replace multiple newlines with single
        cleaned = re.sub(r"\n+", "\n", text)
        # Replace multiple spaces with single
        cleaned = re.sub(r" +", " ", cleaned)
        return cleaned.strip()
