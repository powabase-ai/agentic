"""
RecursiveChunking - split text by finding natural break points.

This is the most common chunking strategy for RAG, as it produces
higher quality chunks by respecting sentence and paragraph boundaries.

Adapted from agno's RecursiveChunking and langchain's RecursiveCharacterTextSplitter.
"""

import re
import warnings

from agentic.knowledge.chunking.base import ChunkingStrategy
from agentic.knowledge.model_config import (
    CHUNK_EMBED_DEFAULT_CHUNK_SIZE,
    CHUNK_EMBED_DEFAULT_OVERLAP,
)
from agentic.knowledge.models import TextChunk


class RecursiveChunking(ChunkingStrategy):
    """
    Chunking strategy that recursively splits text by finding natural break points.

    This strategy:
    - Tries to split at paragraph boundaries (double newlines)
    - Falls back to sentence boundaries (periods, newlines)
    - Then word boundaries (spaces)
    - Supports overlap for context continuity

    This is the recommended strategy for most RAG applications.

    Args:
        chunk_size: Maximum tokens per chunk (default: 2000)
        overlap: Number of overlapping tokens between chunks (default: 50)
        separators: List of separators to try, in order of preference

    Example:
        >>> chunker = RecursiveChunking(chunk_size=2000, overlap=50)
        >>> chunks = chunker.chunk("Long document with paragraphs...")
        >>> for chunk in chunks:
        ...     print(f"Chunk {chunk.index}: {len(chunk.text)} chars")
    """

    name = "recursive"

    # Default separators in order of preference
    DEFAULT_SEPARATORS = [
        "\n\n",  # Paragraph breaks
        "\n",  # Line breaks
        ". ",  # Sentence ends
        "! ",  # Exclamation
        "? ",  # Question
        "; ",  # Semicolon
        ", ",  # Comma
        " ",  # Word boundary
        "",  # Character boundary (last resort)
    ]

    def __init__(
        self,
        chunk_size: int = CHUNK_EMBED_DEFAULT_CHUNK_SIZE,
        overlap: int = CHUNK_EMBED_DEFAULT_OVERLAP,
        separators: list[str] | None = None,
    ):
        if overlap >= chunk_size:
            raise ValueError(
                f"Overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            )

        if overlap > chunk_size * 0.5:
            warnings.warn(
                f"High overlap: {overlap} > 50% of chunk_size ({chunk_size}). "
                "This may cause slow processing and redundant chunks.",
                RuntimeWarning,
                stacklevel=2,
            )

        super().__init__(chunk_size=chunk_size, overlap=overlap)
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(
        self,
        text: str,
        source_id: str | None = None,
    ) -> list[TextChunk]:
        """
        Split text into chunks using recursive splitting at natural boundaries.

        Chunk size and overlap are measured in tokens. Natural break points
        (paragraph, sentence, word boundaries) are found in character space
        and snapped back to the nearest token boundary.

        Args:
            text: The text content to chunk
            source_id: Optional source identifier for chunk metadata

        Returns:
            List of TextChunk objects
        """
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

        while token_start < total_tokens:
            token_end = min(token_start + self.chunk_size, total_tokens)
            char_start = char_map[token_start]
            char_end = char_map[token_end]

            # Find natural break point in character space
            if token_end < total_tokens:
                break_char = self._find_break_point(content, char_start, char_end)
                # Snap to nearest token boundary (<= break_char)
                token_end = char_to_token_pos(char_map, break_char)
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

            # Calculate next start position with overlap
            new_token_start = token_end - self.overlap
            if new_token_start <= token_start:
                new_token_start = min(
                    total_tokens, token_start + max(1, self.chunk_size // 10)
                )
            token_start = new_token_start

        return chunks

    def _find_break_point(self, content: str, start: int, end: int) -> int:
        """
        Find the best break point within the chunk range.

        Tries separators in order of preference.
        """
        chunk = content[start:end]

        for sep in self.separators:
            if not sep:
                # Empty separator = character boundary, use original end
                return end

            # Find last occurrence of separator in chunk
            last_sep = chunk.rfind(sep)
            if last_sep != -1:
                # Found separator, adjust end to include it
                return start + last_sep + len(sep)

        # No separator found, use original end
        return end

    def _clean_text(self, text: str) -> str:
        """Clean text by normalizing excessive whitespace."""
        # Replace 3+ newlines with 2
        cleaned = re.sub(r"\n{3,}", "\n\n", text)
        # Replace multiple spaces (but preserve single newlines)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        return cleaned.strip()
