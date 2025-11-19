"""Abstract base classes for ETL components"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseConnector(ABC):
    """Base class for source connectors"""
    
    @abstractmethod
    def load(self, source: str) -> str:
        """Load document from source and return local file path"""
        pass


class BaseTransformer(ABC):
    """Base class for transformers"""
    
    @abstractmethod
    def transform(self, data: Any) -> Any:
        """Transform input data"""
        pass


class BaseSink(ABC):
    """Base class for storage sinks"""
    
    @abstractmethod
    def store(self, *args, **kwargs) -> Any:
        """Store processed data"""
        pass

