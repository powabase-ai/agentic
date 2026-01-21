"""
RecursiveChunking - split text by finding natural break points.

This is the most common chunking strategy for RAG, as it produces
higher quality chunks by respecting sentence and paragraph boundaries.

Adapted from agno's RecursiveChunking and langchain's RecursiveCharacterTextSplitter.
"""

import re
import warnings
from typing import Optional

from agentic.knowledge.chunking.base import ChunkingStrategy
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
        chunk_size: Maximum characters per chunk (default: 1000)
        overlap: Number of overlapping characters between chunks (default: 200)
        separators: List of separators to try, in order of preference
    
    Example:
        >>> chunker = RecursiveChunking(chunk_size=500, overlap=50)
        >>> chunks = chunker.chunk("Long document with paragraphs...")
        >>> for chunk in chunks:
        ...     print(f"Chunk {chunk.index}: {len(chunk.text)} chars")
    """
    
    name = "recursive"
    
    # Default separators in order of preference
    DEFAULT_SEPARATORS = [
        "\n\n",  # Paragraph breaks
        "\n",    # Line breaks  
        ". ",    # Sentence ends
        "! ",    # Exclamation
        "? ",    # Question
        "; ",    # Semicolon
        ", ",    # Comma
        " ",     # Word boundary
        "",      # Character boundary (last resort)
    ]
    
    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        separators: Optional[list[str]] = None,
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
            )
        
        super().__init__(chunk_size=chunk_size, overlap=overlap)
        self.separators = separators or self.DEFAULT_SEPARATORS
    
    def chunk(
        self,
        text: str,
        source_id: Optional[str] = None,
    ) -> list[TextChunk]:
        """
        Split text into chunks using recursive splitting at natural boundaries.
        
        Args:
            text: The text content to chunk
            source_id: Optional source identifier for chunk metadata
        
        Returns:
            List of TextChunk objects
        """
        content = self._clean_text(text)
        content_length = len(content)
        
        if content_length == 0:
            return []
        
        if content_length <= self.chunk_size:
            return [
                TextChunk(
                    text=content,
                    index=0,
                    start_char=0,
                    end_char=content_length,
                    source_id=source_id,
                    metadata={"strategy": self.name},
                )
            ]
        
        chunks: list[TextChunk] = []
        chunk_index = 0
        start = 0
        
        while start < content_length:
            end = min(start + self.chunk_size, content_length)
            
            # If not at end, find best break point
            if end < content_length:
                end = self._find_break_point(content, start, end)
            
            chunk_text = content[start:end].strip()
            
            if chunk_text:
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        index=chunk_index,
                        start_char=start,
                        end_char=end,
                        source_id=source_id,
                        metadata={
                            "strategy": self.name,
                            "chunk_size": len(chunk_text),
                        },
                    )
                )
                chunk_index += 1
            
            # Calculate next start position with overlap
            new_start = end - self.overlap
            
            # Prevent infinite loop
            if new_start <= start:
                new_start = min(
                    content_length,
                    start + max(1, self.chunk_size // 10)
                )
            
            start = new_start
        
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
