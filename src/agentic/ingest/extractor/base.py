"""
Extractor - base class for content extractors.

Extractors process raw content into derivatives (text, images, etc.)
"""

from abc import ABC, abstractmethod

from agentic.ingest.models import RawContent, ExtractionResult


class Extractor(ABC):
    """
    Base class for content extractors.
    
    An Extractor transforms raw content (bytes) into structured derivatives
    that can be indexed and retrieved. It handles a specific set of MIME
    types and produces appropriate derivatives.
    
    Key Design Principles:
        1. **Async-first**: extract() is async for non-blocking I/O
        2. **MIME-based selection**: supported_types defines what it handles
        3. **Multiple derivatives**: Can produce text, images, metadata, etc.
    
    Subclasses:
        - TextExtractor: Plain text, markdown
        - HTMLExtractor: HTML with tag stripping
        - PDFExtractor: PDF documents (text + images)
        - DocxExtractor: Word documents
    
    Example:
        >>> class PDFExtractor(Extractor):
        ...     name = "pdf"
        ...     supported_types = ["application/pdf"]
        ...     
        ...     async def extract(self, raw: RawContent) -> ExtractionResult:
        ...         # Extract text from PDF
        ...         text = extract_pdf_text(raw.content)
        ...         return ExtractionResult(
        ...             source_uri=raw.source_uri,
        ...             mime_type=raw.mime_type,
        ...             derivatives=[Derivative(type="text", content=text)],
        ...             extraction_method=self.name,
        ...         )
    """
    
    # Subclasses must define these
    name: str = "base"
    supported_types: list[str] = []  # MIME types this extractor handles
    
    def supports(self, mime_type: str) -> bool:
        """
        Check if this extractor supports a MIME type.
        
        Default implementation checks against supported_types list.
        Subclasses can override for more complex matching (wildcards, etc.).
        
        Args:
            mime_type: The MIME type to check (e.g., "application/pdf")
        
        Returns:
            True if this extractor can handle the MIME type
        
        Example:
            >>> extractor = PDFExtractor()
            >>> extractor.supports("application/pdf")
            True
            >>> extractor.supports("text/plain")
            False
        """
        return mime_type in self.supported_types
    
    @abstractmethod
    async def extract(self, raw: RawContent) -> ExtractionResult:
        """
        Extract content from raw bytes.
        
        Args:
            raw: RawContent containing bytes and metadata
        
        Returns:
            ExtractionResult with derivatives and metadata
        
        Raises:
            ExtractionError: If extraction fails
        
        Example:
            >>> raw = RawContent(content=b"...", mime_type="application/pdf", ...)
            >>> result = await extractor.extract(raw)
            >>> text = result.get_primary_text()
        """
        ...
    
    def __repr__(self) -> str:
        types = ", ".join(self.supported_types[:3])
        if len(self.supported_types) > 3:
            types += ", ..."
        return f"{self.__class__.__name__}(name={self.name!r}, types=[{types}])"


class ExtractionError(Exception):
    """
    Error during content extraction.
    
    Attributes:
        message: Description of the error
        extractor_name: Name of the extractor that failed
        source_uri: URI of the content being extracted
        cause: Original exception if any
    """
    
    def __init__(
        self,
        message: str,
        extractor_name: str | None = None,
        source_uri: str | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.extractor_name = extractor_name
        self.source_uri = source_uri
        self.cause = cause
    
    def __str__(self) -> str:
        parts = [self.message]
        if self.extractor_name:
            parts.append(f"extractor={self.extractor_name}")
        if self.source_uri:
            parts.append(f"source={self.source_uri}")
        return " | ".join(parts)
