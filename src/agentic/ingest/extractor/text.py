"""
TextExtractor - simple passthrough for text-based files.

Handles plain text, markdown, and other text formats.
For markdown files, also produces page_text derivatives via native splitting.
"""

from agentic.ingest.extractor.base import ExtractionError, Extractor
from agentic.ingest.models import Derivative, ExtractionResult, RawContent


_MARKDOWN_MIMES = {"text/markdown", "text/x-markdown"}


class TextExtractor(Extractor):
    """
    Extractor for plain text files.

    This is the simplest extractor - it decodes bytes as text with
    optional encoding detection and produces a single text derivative.

    For markdown MIME types, also produces page_text derivatives using
    heading-aware page splitting.

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
            ExtractionResult with text derivative (and page_text derivatives
            for markdown files)
        """
        try:
            text = raw.content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.content.decode("latin-1")
            except Exception as e:
                raise ExtractionError(
                    f"Failed to decode text content: {e}",
                    extractor_name=self.name,
                    source_uri=raw.source_uri,
                    cause=e,
                ) from e

        is_markdown = raw.mime_type in _MARKDOWN_MIMES
        format_name = "markdown" if is_markdown else "plain"

        extraction_method = self.name
        auto_metadata: dict = {
            "char_count": len(text),
            "line_count": text.count("\n") + 1,
        }

        # For markdown files, produce page_text derivatives via native splitting
        if is_markdown:
            from agentic.ingest.extractor.page_splitter import (
                split_text_into_pages,
            )

            pages = split_text_into_pages(
                text, target_tokens=1500, heading_aware=True
            )
            content = "\n\n".join(pages)

            derivatives = [
                Derivative(
                    type="text",
                    content=content,
                    format=format_name,
                    metadata={
                        "encoding": "utf-8",
                        "char_count": len(content),
                        "line_count": content.count("\n") + 1,
                    },
                ),
            ]
            for i, page_text in enumerate(pages, start=1):
                derivatives.append(
                    Derivative(
                        type="page_text",
                        content=page_text,
                        format="plain",
                        page=i,
                    )
                )
            extraction_method = "markdown-native"
            auto_metadata["page_count"] = len(pages)
            auto_metadata["char_count"] = len(content)
            auto_metadata["line_count"] = content.count("\n") + 1
            auto_metadata["extraction_method"] = extraction_method
        else:
            derivatives = [
                Derivative(
                    type="text",
                    content=text,
                    format=format_name,
                    metadata={
                        "encoding": "utf-8",
                        "char_count": len(text),
                        "line_count": text.count("\n") + 1,
                    },
                ),
            ]

        return ExtractionResult(
            source_uri=raw.source_uri,
            mime_type=raw.mime_type,
            derivatives=derivatives,
            auto_metadata=auto_metadata,
            extraction_method=extraction_method,
            stats={
                "bytes_processed": len(raw.content),
            },
        )
