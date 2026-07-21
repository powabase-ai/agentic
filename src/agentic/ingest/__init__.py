"""
Content Ingestion module - how content enters and is processed.

The Ingest module provides abstractions for:
- **Connectors**: How content enters the system (file upload, S3, web crawl)
- **Extractors**: How raw content is processed into derivatives (PDF → text)
- **Registry**: Auto-select extractors based on MIME type

This module is designed for extensibility - users can register custom
connectors and extractors to handle new content types.

Example:
    >>> from agentic.ingest import FileUploadConnector, ExtractorRegistry
    >>>
    >>> # Upload a file
    >>> connector = FileUploadConnector()
    >>> raw = await connector.fetch_bytes(file_bytes, "report.pdf", "application/pdf")
    >>>
    >>> # Extract content
    >>> registry = ExtractorRegistry.default()
    >>> extractor = registry.get_extractor(raw.mime_type)
    >>> result = await extractor.extract(raw)
    >>>
    >>> print(result.derivatives[0].content)  # Extracted text

See Also:
    - docs/CONCEPTS.md for detailed explanations
    - powabase-ai for storage integration
"""

from agentic.ingest.connector import Connector, FileUploadConnector
from agentic.ingest.extractor import (
    DocxExtractor,
    ExtractionError,
    Extractor,
    ExtractorRegistry,
    HTMLExtractor,
    ImageExtractor,
    PDFExtractor,
    PptxExtractor,
    TextExtractor,
    XlsxExtractor,
)
from agentic.ingest.models import ContentItem, Derivative, ExtractionResult, RawContent

__all__ = [
    # Models
    "RawContent",
    "Derivative",
    "ExtractionResult",
    "ContentItem",
    # Connectors
    "Connector",
    "FileUploadConnector",
    # Extractors
    "Extractor",
    "ExtractionError",
    "ExtractorRegistry",
    "TextExtractor",
    "HTMLExtractor",
    "PDFExtractor",
    "DocxExtractor",
    "PptxExtractor",
    "XlsxExtractor",
    "ImageExtractor",
]
