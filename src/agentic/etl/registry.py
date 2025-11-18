"""Plugin registry system for ETL components"""

from typing import Dict, Type, Any
from .base import BaseConnector, BaseTransformer, BaseSink


class ConnectorRegistry:
    """Registry for connectors"""
    _connectors: Dict[str, Type[BaseConnector]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a connector"""
        def decorator(connector_class: Type[BaseConnector]):
            cls._connectors[name] = connector_class
            return connector_class
        return decorator
    
    @classmethod
    def create(cls, name: str, config: Dict[str, Any]) -> BaseConnector:
        """Create a connector instance"""
        if name not in cls._connectors:
            raise ValueError(f"Unknown connector: {name}")
        return cls._connectors[name](**config)


class ExtractorRegistry:
    """Registry for extractors"""
    _extractors: Dict[str, Type[BaseTransformer]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register an extractor"""
        def decorator(extractor_class: Type[BaseTransformer]):
            cls._extractors[name] = extractor_class
            return extractor_class
        return decorator
    
    @classmethod
    def create(cls, name: str, config: Dict[str, Any]) -> BaseTransformer:
        """Create an extractor instance"""
        if name == "auto":
            # Auto-detect based on file extension
            file_path = config.get("file_path", "")
            if not file_path:
                # Try to get from other config keys
                file_path = config.get("source", "") or config.get("path", "")
            
            ext = file_path.split(".")[-1].lower() if "." in file_path else ""
            if ext == "pdf":
                name = "pdf"
            elif ext in ["docx", "doc"]:
                name = "docx"
            elif ext in ["jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff"]:
                name = "image"
            elif ext in ["txt", "md", "tex", "rst"]:
                name = "text"
            else:
                name = "text"  # Default fallback
        
        if name not in cls._extractors:
            raise ValueError(f"Unknown extractor: {name}. Available: {list(cls._extractors.keys())}")
        return cls._extractors[name](**config)


class ChunkerFactory:
    """Factory for chunkers"""
    _chunkers: Dict[str, Type[BaseTransformer]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a chunker"""
        def decorator(chunker_class: Type[BaseTransformer]):
            cls._chunkers[name] = chunker_class
            return chunker_class
        return decorator
    
    @classmethod
    def create(cls, strategy: str, **kwargs) -> BaseTransformer:
        """Create a chunker instance"""
        if strategy not in cls._chunkers:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        return cls._chunkers[strategy](**kwargs)


class EmbedderFactory:
    """Factory for embedders"""
    _embedders: Dict[str, Type[BaseTransformer]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register an embedder"""
        def decorator(embedder_class: Type[BaseTransformer]):
            cls._embedders[name] = embedder_class
            return embedder_class
        return decorator
    
    @classmethod
    def create(cls, provider: str, **kwargs) -> BaseTransformer:
        """Create an embedder instance"""
        if provider not in cls._embedders:
            raise ValueError(f"Unknown embedding provider: {provider}")
        return cls._embedders[provider](**kwargs)


class SinkRegistry:
    """Registry for sinks"""
    _sinks: Dict[str, Type[BaseSink]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a sink"""
        def decorator(sink_class: Type[BaseSink]):
            cls._sinks[name] = sink_class
            return sink_class
        return decorator
    
    @classmethod
    def create(cls, name: str, config: Dict[str, Any]) -> BaseSink:
        """Create a sink instance"""
        if name not in cls._sinks:
            raise ValueError(f"Unknown sink: {name}")
        return cls._sinks[name](**config)

