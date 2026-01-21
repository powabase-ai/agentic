"""
FixedSizeChunking - split text into fixed-size chunks with overlap.

This is the simplest chunking strategy, useful when you don't need
to respect document structure. Good for general-purpose RAG.

Adapted from agno's FixedSizeChunking pattern.
"""

import re
from typing import Optional

from agentic.knowledge.chunking.base import ChunkingStrategy
from agentic.knowledge.models import TextChunk


class FixedSizeChunking(ChunkingStrategy):
    """
    Chunking strategy that splits text into fixed-size chunks with optional overlap.
    
    This strategy:
    - Splits by character count (not tokens)
    - Avoids splitting words in the middle
    - Supports overlap for context continuity
    
    Args:
        chunk_size: Maximum characters per chunk (default: 1000)
        overlap: Number of overlapping characters between chunks (default: 200)
    
    Example:
        >>> chunker = FixedSizeChunking(chunk_size=500, overlap=50)
        >>> chunks = chunker.chunk("Long document text...")
        >>> for chunk in chunks:
        ...     print(f"Chunk {chunk.index}: {len(chunk.text)} chars")
    """
    
    name = "fixed_size"
    
    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
    ):
        if overlap >= chunk_size:
            raise ValueError(
                f"Overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            )
        
        super().__init__(chunk_size=chunk_size, overlap=overlap)
    
    def chunk(
        self,
        text: str,
        source_id: Optional[str] = None,
    ) -> list[TextChunk]:
        """
        Split text into fixed-size chunks.
        
        Args:
            text: The text content to chunk
            source_id: Optional source identifier for chunk metadata
        
        Returns:
            List of TextChunk objects
        """
        # Clean text
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
        
        while start + self.overlap < content_length:
            end = min(start + self.chunk_size, content_length)
            
            # Avoid splitting words - find last whitespace
            if end < content_length:
                # Look for whitespace to break at
                original_end = end
                while end > start and content[end] not in " \n\r\t":
                    end -= 1
                
                # If we couldn't find whitespace, use original end
                if end == start:
                    end = original_end
            
            chunk_text = content[start:end].strip()
            
            if chunk_text:  # Only add non-empty chunks
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
            
            # Move to next position with overlap
            # Ensure we always advance to prevent infinite loops
            new_start = max(start + 1, end - self.overlap)
            start = new_start
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean text by normalizing whitespace."""
        # Replace multiple newlines with single
        cleaned = re.sub(r"\n+", "\n", text)
        # Replace multiple spaces with single
        cleaned = re.sub(r" +", " ", cleaned)
        return cleaned.strip()
