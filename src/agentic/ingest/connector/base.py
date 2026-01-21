"""
Connector - base class for content source connectors.

Connectors handle how content enters the system from various sources.
"""

from abc import ABC, abstractmethod
from typing import Any

from agentic.ingest.models import ContentItem, RawContent


class Connector(ABC):
    """
    Base class for content source connectors.
    
    A Connector provides access to content from an external source.
    It handles authentication, listing available content, and fetching
    raw content bytes.
    
    Key Design Principles:
        1. **Async-first**: All I/O operations are async
        2. **Two-phase fetch**: list_items() → fetch() pattern for browsable sources
        3. **Config-based auth**: connect() receives credentials/config
    
    Subclasses:
        - FileUploadConnector: Direct bytes passthrough (simplest)
        - S3Connector: AWS S3 bucket access (future)
        - WebCrawlerConnector: Fetch web pages (future)
    
    Example:
        >>> class S3Connector(Connector):
        ...     name = "s3"
        ...     
        ...     async def connect(self, config):
        ...         self.bucket = config["bucket"]
        ...         self.client = boto3.client("s3", ...)
        ...     
        ...     async def list_items(self):
        ...         objects = self.client.list_objects_v2(Bucket=self.bucket)
        ...         return [ContentItem(uri=obj["Key"], ...) for obj in objects]
        ...     
        ...     async def fetch(self, item):
        ...         response = self.client.get_object(Bucket=self.bucket, Key=item.uri)
        ...         return RawContent(content=response["Body"].read(), ...)
    """
    
    # Subclasses must define this
    name: str = "base"
    
    async def connect(self, config: dict[str, Any]) -> None:
        """
        Initialize the connector with configuration.
        
        Override this for connectors that need authentication or setup.
        The default implementation does nothing (for simple connectors).
        
        Args:
            config: Connector-specific configuration (credentials, paths, etc.)
        
        Example:
            >>> await connector.connect({
            ...     "bucket": "my-bucket",
            ...     "access_key": "...",
            ...     "secret_key": "...",
            ... })
        """
        pass  # Default: no setup needed
    
    async def list_items(self) -> list[ContentItem]:
        """
        List available content items.
        
        For browsable sources (S3, file systems), returns all available items.
        For non-browsable sources (uploads), this may return an empty list.
        
        Returns:
            List of ContentItem objects that can be fetched
        
        Example:
            >>> items = await connector.list_items()
            >>> for item in items:
            ...     print(f"{item.name}: {item.size_bytes} bytes")
        """
        return []  # Default: no listing (e.g., for upload connectors)
    
    @abstractmethod
    async def fetch(self, item: ContentItem) -> RawContent:
        """
        Fetch raw content for a specific item.
        
        Args:
            item: The ContentItem to fetch (from list_items or constructed)
        
        Returns:
            RawContent containing the bytes and metadata
        
        Example:
            >>> item = ContentItem(uri="path/to/file.pdf", name="file.pdf")
            >>> raw = await connector.fetch(item)
            >>> print(f"Fetched {len(raw.content)} bytes")
        """
        ...
    
    async def disconnect(self) -> None:
        """
        Clean up connector resources.
        
        Override for connectors that need cleanup (close connections, etc.).
        """
        pass  # Default: no cleanup needed
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
