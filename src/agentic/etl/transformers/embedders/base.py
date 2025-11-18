"""Base embedder class"""

from abc import ABC, abstractmethod
from typing import List

from ...base import BaseTransformer


class BaseEmbedder(BaseTransformer):
    """Base class for embedding generators"""
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        pass
    
    def transform(self, texts: List[str]) -> List[List[float]]:
        """Transform interface - wraps embed"""
        return self.embed(texts)

