"""Base extractor class"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

from ...base import BaseTransformer


@dataclass
class ExtractionResult:
    """Simple extraction result"""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseExtractor(BaseTransformer):
    """Base class for text extractors"""
    
    @abstractmethod
    def extract(self, file_path: str) -> ExtractionResult:
        """Extract text from file"""
        pass
    
    def transform(self, file_path: str) -> Dict[str, Any]:
        """Transform interface - wraps extract"""
        result = self.extract(file_path)
        return {
            "text": result.text,
            "metadata": result.metadata,
        }

