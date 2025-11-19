"""Base chunker class"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ...base import BaseTransformer


@dataclass
class Chunk:
    """Represents a text chunk"""
    text: str
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseChunker(BaseTransformer):
    """Base class for text chunkers"""
    
    @abstractmethod
    def chunk(self, text: str, metadata: Dict = None) -> List[Chunk]:
        """Split text into chunks"""
        pass
    
    def transform(self, extraction: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Transform interface - wraps chunk"""
        text = extraction.get("text", "")
        metadata = extraction.get("metadata", {})
        chunks = self.chunk(text, metadata)
        
        # Convert Chunk objects to dicts
        return [
            {
                "text": chunk.text,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]

