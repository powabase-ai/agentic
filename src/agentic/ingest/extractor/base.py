"""
Extractor - base class for content extractors.

Extractors process raw content into derivatives (text, images, etc.)
"""

import logging
import pickle
import re
from abc import ABC, abstractmethod

from agentic.ingest.models import ExtractionResult, RawContent

_logger = logging.getLogger(__name__)


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


class PageImageSinkError(Exception):
    """The host's page-image sink raised while taking a rendered page.

    Raised by an extractor that was given ``raw.metadata["page_image_sink"]``
    when that callable fails. It ends the whole extraction: the extractor does
    not retry the method or move on to another one, because every method
    delivers its pages to the same sink and an OCR method pays for every page
    it sends first. Whether to try again is the caller's decision.

    Deliberately not an ``ExtractionError``, which callers may treat as a
    permanent verdict on the document: a sink that cannot store a page is
    usually a storage outage, not a bad document.

    Attributes:
        page: 1-indexed page the sink was given when it failed
        cause: the exception the sink raised (also ``__cause__``)
    """

    def __init__(self, page: int | None, cause: BaseException):
        super().__init__(
            f"page image sink failed on page {page}: {_describe_exception(cause)}"
        )
        self.page = page
        self.cause = cause
        self.__cause__ = cause

    def __reduce__(self):
        # BaseException pickles as ``type(self)(*self.args)``, and args holds
        # only the message; rebuild from the constructor's own arguments so
        # the error survives a trip through a task queue or process pool.
        # The cause is the host's exception and may not survive that trip
        # itself (a constructor taking other arguments, a lock); then it
        # travels as text, so the error stays reportable.
        cause = self.cause
        try:
            pickle.loads(pickle.dumps(cause))
        except Exception:
            cause = RuntimeError(_describe_exception(cause))
        return (_rebuild_page_image_sink_error, (type(self), self.page, cause, self.args))


def _describe_exception(exc: BaseException) -> str:
    """``Type: message``, even for an exception whose ``__str__`` raises."""
    try:
        text = str(exc)
    except Exception:
        text = "<exception message could not be read>"
    return f"{type(exc).__name__}: {text}"


def _rebuild_page_image_sink_error(cls, page, cause, args):
    error = cls(page, cause)
    error.args = args
    return error


def replace_image_annotations(
    markdown: str,
    annotations: list[tuple[str, str | None]],
) -> str:
    """Replace image placeholders in markdown with annotation text.

    Provider-agnostic: any OCR extractor can call this with a list of
    (image_id, annotation_text) pairs extracted from its response.

    Handles multiple placeholder patterns:
      - ![id](id)       — bare id as URL (Mistral default)
      - ![id](https://) — full URL variant

    Args:
        markdown: The OCR markdown containing image placeholders.
        annotations: List of (image_id, annotation_text) tuples.

    Returns:
        Markdown with image placeholders replaced by annotations.
    """
    for image_id, annotation in annotations:
        if not annotation:
            _logger.debug(
                "Image %s has no annotation, skipping replacement", image_id
            )
            continue

        # Try exact id-as-URL match first (fast path): ![id](id)
        exact = f"![{image_id}]({image_id})"
        if exact in markdown:
            markdown = markdown.replace(
                exact, f"![{image_id}]\n**{annotation}**"
            )
            continue

        # Fallback: match ![id](anything) via regex
        pattern = re.escape(f"![{image_id}]") + r"\([^)]*\)"
        replacement = f"![{image_id}]\n**{annotation}**"
        markdown = re.sub(pattern, lambda _: replacement, markdown)

    return markdown
