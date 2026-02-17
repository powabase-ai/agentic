"""
ExtractorRegistry - auto-select extractors based on MIME type.

The registry pattern allows:
1. Built-in extractors for common file types
2. Custom extractors that override or extend built-ins
3. Automatic selection based on MIME type
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic.ingest.extractor.base import Extractor


class ExtractorRegistry:
    """
    Registry for content extractors with MIME type-based selection.

    The registry maintains a mapping from MIME types to extractors.
    When multiple extractors support the same type, the most recently
    registered one takes precedence.

    Example:
        >>> registry = ExtractorRegistry()
        >>>
        >>> # Register extractors
        >>> registry.register(TextExtractor())
        >>> registry.register(PDFExtractor())
        >>>
        >>> # Get extractor by MIME type
        >>> extractor = registry.get_extractor("application/pdf")
        >>> result = await extractor.extract(raw_content)
        >>>
        >>> # Override built-in with custom
        >>> registry.register(MyBetterPDFExtractor())  # Takes precedence
    """

    def __init__(self):
        """Initialize an empty registry."""
        self._extractors: dict[str, Extractor] = {}
        self._by_name: dict[str, Extractor] = {}

    def register(self, extractor: "Extractor") -> None:
        """
        Register an extractor for its supported MIME types.

        If an extractor for a MIME type already exists, this one
        takes precedence (useful for overriding built-ins).

        Args:
            extractor: The extractor to register

        Example:
            >>> registry.register(PDFExtractor())
        """
        # Store by name
        self._by_name[extractor.name] = extractor

        # Map each supported MIME type to this extractor
        for mime_type in extractor.supported_types:
            self._extractors[mime_type] = extractor

    def get_extractor(self, mime_type: str) -> "Extractor":
        """
        Get an extractor for a MIME type.

        Args:
            mime_type: The MIME type to find an extractor for

        Returns:
            The registered extractor for this type

        Raises:
            KeyError: If no extractor is registered for this type

        Example:
            >>> extractor = registry.get_extractor("application/pdf")
        """
        if mime_type in self._extractors:
            return self._extractors[mime_type]

        # Try without parameters (e.g., "text/plain; charset=utf-8" → "text/plain")
        base_type = mime_type.split(";")[0].strip()
        if base_type in self._extractors:
            return self._extractors[base_type]

        # Try wildcard match for type/* patterns
        type_prefix = mime_type.split("/")[0]
        wildcard = f"{type_prefix}/*"
        if wildcard in self._extractors:
            return self._extractors[wildcard]

        raise KeyError(
            f"No extractor registered for MIME type: {mime_type}. "
            f"Available types: {list(self._extractors.keys())}"
        )

    def get_by_name(self, name: str) -> "Extractor":
        """
        Get an extractor by its name.

        Args:
            name: The extractor name

        Returns:
            The registered extractor

        Raises:
            KeyError: If no extractor with this name exists
        """
        if name not in self._by_name:
            raise KeyError(
                f"No extractor with name: {name}. "
                f"Available: {list(self._by_name.keys())}"
            )
        return self._by_name[name]

    def supports(self, mime_type: str) -> bool:
        """
        Check if any registered extractor supports a MIME type.

        Args:
            mime_type: The MIME type to check

        Returns:
            True if an extractor exists for this type
        """
        try:
            self.get_extractor(mime_type)
            return True
        except KeyError:
            return False

    def list_extractors(self) -> list["Extractor"]:
        """Get all registered extractors."""
        return list(self._by_name.values())

    def list_supported_types(self) -> list[str]:
        """Get all supported MIME types."""
        return list(self._extractors.keys())

    @classmethod
    def default(cls) -> "ExtractorRegistry":
        """
        Create a registry with built-in extractors.

        This includes extractors for:
        - text/* (TextExtractor)
        - HTML, PDF, DOCX

        Environment variables used:
        - MISTRAL_API_KEY: If set, enables Mistral OCR for PDF extraction

        Returns:
            ExtractorRegistry with built-ins registered

        Example:
            >>> registry = ExtractorRegistry.default()
            >>> extractor = registry.get_extractor("text/plain")
        """
        import os

        registry = cls()

        # Import and register built-in extractors
        try:
            from agentic.ingest.extractor.text import TextExtractor

            registry.register(TextExtractor())
        except ImportError:
            pass  # Not yet implemented

        try:
            from agentic.ingest.extractor.html import HTMLExtractor

            registry.register(HTMLExtractor())
        except ImportError:
            pass  # Not yet implemented

        try:
            from agentic.ingest.extractor.pdf import PDFExtractor

            # Use Mistral OCR if API key is available
            mistral_api_key = os.getenv("MISTRAL_API_KEY")
            registry.register(PDFExtractor(mistral_api_key=mistral_api_key))
        except ImportError:
            pass  # Not yet implemented

        try:
            from agentic.ingest.extractor.docx import DocxExtractor

            registry.register(DocxExtractor())
        except ImportError:
            pass  # Not yet implemented

        try:
            from agentic.ingest.extractor.image import ImageExtractor

            mistral_api_key = os.getenv("MISTRAL_API_KEY")
            registry.register(ImageExtractor(mistral_api_key=mistral_api_key))
        except ImportError:
            pass

        return registry

    def __len__(self) -> int:
        """Return number of unique extractors."""
        return len(self._by_name)

    def __repr__(self) -> str:
        return f"ExtractorRegistry(extractors={len(self._by_name)}, types={len(self._extractors)})"
