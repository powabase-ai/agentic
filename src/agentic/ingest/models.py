"""
Models for the content ingestion module.

These models define the data structures for content as it flows through
the ingestion pipeline: RawContent → Extractor → ExtractionResult (with Derivatives)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class ContentItem:
    """
    A reference to content available from a connector.
    
    Connectors return ContentItems when listing available content.
    These can then be fetched to get RawContent.
    
    Attributes:
        uri: Unique identifier/path for this item
        name: Human-readable name (e.g., filename)
        mime_type: Content type if known
        size_bytes: Size in bytes if known
        modified_at: Last modification time if known
        metadata: Additional connector-specific metadata
    """
    uri: str
    name: str
    mime_type: str | None = None
    size_bytes: int | None = None
    modified_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawContent:
    """
    Raw content fetched from a connector, ready for extraction.
    
    This is the input to extractors. It contains the raw bytes and
    metadata about the content source.
    
    Attributes:
        content: Raw bytes of the content
        mime_type: MIME type (e.g., "application/pdf", "text/plain")
        source_uri: URI identifying where this came from
        filename: Original filename if available
        size_bytes: Size of content in bytes
        metadata: Additional metadata (source-specific)
        fetched_at: When the content was fetched
    
    Example:
        >>> raw = RawContent(
        ...     content=b"PDF binary data...",
        ...     mime_type="application/pdf",
        ...     source_uri="upload://report.pdf",
        ...     filename="report.pdf",
        ... )
    """
    content: bytes
    mime_type: str
    source_uri: str
    filename: str | None = None
    size_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    fetched_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if self.size_bytes is None:
            self.size_bytes = len(self.content)


# Derivative types - what extractors produce
DerivativeType = Literal["text", "markdown", "html", "image", "table", "metadata"]


@dataclass
class Derivative:
    """
    A piece of extracted content from a source.
    
    Extractors produce one or more derivatives from raw content.
    For example, a PDF might produce:
    - A "text" derivative with plain text
    - Multiple "image" derivatives for embedded images
    - A "metadata" derivative with document properties
    
    Attributes:
        type: Kind of derivative ("text", "markdown", "image", etc.)
        content: The extracted content (string for text, bytes for binary)
        format: Specific format (e.g., "plain", "utf-8", "png")
        page: Page number if applicable (1-indexed)
        metadata: Additional extraction metadata
    
    Example:
        >>> # Text derivative
        >>> text_deriv = Derivative(
        ...     type="text",
        ...     content="Extracted text content...",
        ...     format="plain",
        ... )
        >>> 
        >>> # Image derivative
        >>> img_deriv = Derivative(
        ...     type="image",
        ...     content=b"PNG bytes...",
        ...     format="png",
        ...     page=3,
        ...     metadata={"width": 800, "height": 600},
        ... )
    """
    type: DerivativeType
    content: str | bytes
    format: str = "plain"
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def is_text(self) -> bool:
        """Check if this is a text-based derivative."""
        return self.type in ("text", "markdown", "html")
    
    def get_text(self) -> str:
        """Get content as string, decoding bytes if needed."""
        if isinstance(self.content, bytes):
            return self.content.decode("utf-8", errors="replace")
        return self.content


@dataclass
class ExtractionResult:
    """
    Result of extracting content from a source.
    
    Contains all derivatives produced by an extractor, along with
    metadata about the extraction process.
    
    Attributes:
        source_uri: URI of the source that was extracted
        mime_type: MIME type of the source
        derivatives: List of extracted derivatives
        auto_metadata: Metadata automatically extracted (title, author, etc.)
        extraction_method: Name of the extractor/method used
        extracted_at: When extraction completed
        stats: Extraction statistics (pages processed, time taken, etc.)
    
    Example:
        >>> result = ExtractionResult(
        ...     source_uri="upload://doc.pdf",
        ...     mime_type="application/pdf",
        ...     derivatives=[text_deriv, img_deriv],
        ...     auto_metadata={"title": "My Document", "pages": 10},
        ...     extraction_method="pdf",
        ... )
        >>> 
        >>> # Get primary text content
        >>> text = result.get_primary_text()
    """
    source_uri: str
    mime_type: str
    derivatives: list[Derivative]
    auto_metadata: dict[str, Any] = field(default_factory=dict)
    extraction_method: str = "unknown"
    extracted_at: datetime = field(default_factory=datetime.now)
    stats: dict[str, Any] = field(default_factory=dict)
    
    def get_primary_text(self) -> str | None:
        """
        Get the primary text content from derivatives.
        
        Looks for text/markdown derivatives in order of preference.
        
        Returns:
            The text content, or None if no text derivative exists
        """
        # Prefer markdown over plain text
        for dtype in ("markdown", "text", "html"):
            for deriv in self.derivatives:
                if deriv.type == dtype:
                    return deriv.get_text()
        return None
    
    def get_all_text(self) -> str:
        """
        Concatenate all text content from derivatives.
        
        Returns:
            All text content joined with newlines
        """
        texts = []
        for deriv in self.derivatives:
            if deriv.is_text():
                texts.append(deriv.get_text())
        return "\n\n".join(texts)
    
    def get_derivatives_by_type(self, dtype: DerivativeType) -> list[Derivative]:
        """Get all derivatives of a specific type."""
        return [d for d in self.derivatives if d.type == dtype]
