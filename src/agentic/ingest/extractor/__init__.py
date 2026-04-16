"""
Extractor module - how raw content is processed into derivatives.

Extractors transform raw content (bytes) into structured derivatives
(text, images, metadata) that can be indexed and retrieved.

Built-in extractors:
- TextExtractor: Plain text and markdown files
- HTMLExtractor: HTML with tag stripping
- PDFExtractor: PDF documents (fitz/pdfplumber/Mistral OCR)
- DocxExtractor: Word documents (pypandoc/python-docx)

The ExtractorRegistry provides automatic extractor selection based on
MIME type, and allows custom extractors to be registered.

Example:
    >>> from agentic.ingest.extractor import ExtractorRegistry
    >>>
    >>> # Get default registry with built-in extractors
    >>> registry = ExtractorRegistry.default()
    >>>
    >>> # Auto-select extractor based on MIME type
    >>> extractor = registry.get_extractor("application/pdf")
    >>> result = await extractor.extract(raw_content)
    >>>
    >>> # Register custom extractor
    >>> registry.register(MyCustomExtractor())
"""

from agentic.ingest.extractor.base import ExtractionError, Extractor
from agentic.ingest.extractor.docx import DocxExtractor
from agentic.ingest.extractor.html import HTMLExtractor
from agentic.ingest.extractor.image import ImageExtractor
from agentic.ingest.extractor.pdf import PDFExtractor
from agentic.ingest.extractor.pptx import PptxExtractor
from agentic.ingest.extractor.registry import ExtractorRegistry
from agentic.ingest.extractor.text import TextExtractor
from agentic.ingest.extractor.txt import TxtExtractor
from agentic.ingest.extractor.xlsx import XlsxExtractor

__all__ = [
    "Extractor",
    "ExtractionError",
    "ExtractorRegistry",
    # Built-in extractors
    "TextExtractor",
    "HTMLExtractor",
    "PDFExtractor",
    "DocxExtractor",
    "PptxExtractor",
    "XlsxExtractor",
    "ImageExtractor",
    "TxtExtractor",
]
