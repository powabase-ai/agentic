"""
TextExtractor - simple passthrough for text-based files.

Handles plain text, markdown, and other text formats.
"""

from agentic.ingest.extractor.base import ExtractionError, Extractor
from agentic.ingest.models import Derivative, ExtractionResult, RawContent


class TextExtractor(Extractor):
    """
    Extractor for plain text files.

    This is the simplest extractor - it decodes bytes as text with
    optional encoding detection and produces a single text derivative.

    Supported types:
        - text/plain (.txt)
        - text/markdown (.md)
        - text/x-python, text/x-java, etc. (source code)
        - text/* (any text type)

    Example:
        >>> extractor = TextExtractor()
        >>> raw = RawContent(
        ...     content=b"Hello, world!",
        ...     mime_type="text/plain",
        ...     source_uri="upload://hello.txt",
        ... )
        >>> result = await extractor.extract(raw)
        >>> print(result.get_primary_text())
        "Hello, world!"
    """

    name = "text"
    supported_types = [
        "text/plain",
        "text/markdown",
        "text/x-markdown",
        "text/csv",
        "text/html",  # Fallback, HTMLExtractor preferred
        "text/xml",
        "text/json",
        "application/json",
        "text/x-python",
        "text/x-java",
        "text/javascript",
        "application/javascript",
        "text/*",  # Wildcard for any text type
    ]

    def __init__(self, default_encoding: str = "utf-8"):
        """
        Initialize the text extractor.

        Args:
            default_encoding: Encoding to use if detection fails
        """
        self.default_encoding = default_encoding

    async def extract(self, raw: RawContent) -> ExtractionResult:
        """
        Extract text content from raw bytes.

        Args:
            raw: RawContent with text bytes

        Returns:
            ExtractionResult with a single text derivative
        """
        try:
            # Try UTF-8 first (most common)
            text = raw.content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                # Fall back to latin-1 (never fails, but may be wrong)
                text = raw.content.decode("latin-1")
            except Exception as e:
                raise ExtractionError(
                    f"Failed to decode text content: {e}",
                    extractor_name=self.name,
                    source_uri=raw.source_uri,
                    cause=e,
                ) from e

        # Determine format from MIME type — always produce "text" derivative
        # so frontend and indexing pipeline work uniformly
        format_name = "markdown" if "markdown" in raw.mime_type else "plain"
        deriv_type = "text"

        return ExtractionResult(
            source_uri=raw.source_uri,
            mime_type=raw.mime_type,
            derivatives=[
                Derivative(
                    type=deriv_type,
                    content=text,
                    format=format_name,
                    metadata={
                        "encoding": "utf-8",
                        "char_count": len(text),
                        "line_count": text.count("\n") + 1,
                    },
                ),
            ],
            auto_metadata={
                "char_count": len(text),
                "line_count": text.count("\n") + 1,
            },
            extraction_method=self.name,
            stats={
                "bytes_processed": len(raw.content),
            },
        )
