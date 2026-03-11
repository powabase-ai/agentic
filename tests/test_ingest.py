"""
Tests for the Content Ingestion module.
"""

import pytest

from agentic.ingest import (
    ContentItem,
    Derivative,
    DocxExtractor,
    ExtractionError,
    ExtractionResult,
    Extractor,
    ExtractorRegistry,
    FileUploadConnector,
    HTMLExtractor,
    PDFExtractor,
    RawContent,
    TextExtractor,
)


class TestModels:
    """Tests for ingest data models."""

    def test_content_item(self):
        """ContentItem should store content reference."""
        item = ContentItem(
            uri="s3://bucket/file.pdf",
            name="file.pdf",
            mime_type="application/pdf",
            size_bytes=1024,
        )

        assert item.uri == "s3://bucket/file.pdf"
        assert item.name == "file.pdf"
        assert item.mime_type == "application/pdf"
        assert item.size_bytes == 1024

    def test_raw_content(self):
        """RawContent should hold bytes and metadata."""
        raw = RawContent(
            content=b"Hello, world!",
            mime_type="text/plain",
            source_uri="upload://hello.txt",
            filename="hello.txt",
        )

        assert raw.content == b"Hello, world!"
        assert raw.mime_type == "text/plain"
        assert raw.source_uri == "upload://hello.txt"
        assert raw.size_bytes == 13  # Auto-calculated
        assert raw.fetched_at is not None

    def test_derivative_text(self):
        """Derivative should hold extracted content."""
        deriv = Derivative(
            type="text",
            content="Extracted text here",
            format="plain",
            page=1,
        )

        assert deriv.type == "text"
        assert deriv.content == "Extracted text here"
        assert deriv.is_text()
        assert deriv.get_text() == "Extracted text here"

    def test_derivative_binary(self):
        """Derivative should handle binary content."""
        deriv = Derivative(
            type="image",
            content=b"PNG binary data",
            format="png",
        )

        assert deriv.type == "image"
        assert not deriv.is_text()
        # get_text() should decode bytes
        assert "PNG binary data" in deriv.get_text()

    def test_extraction_result(self):
        """ExtractionResult should contain derivatives."""
        result = ExtractionResult(
            source_uri="upload://doc.pdf",
            mime_type="application/pdf",
            derivatives=[
                Derivative(type="text", content="Page 1 text"),
                Derivative(type="text", content="Page 2 text"),
            ],
            auto_metadata={"pages": 2},
            extraction_method="pdf",
        )

        assert result.source_uri == "upload://doc.pdf"
        assert len(result.derivatives) == 2
        assert result.extraction_method == "pdf"

    def test_extraction_result_get_primary_text(self):
        """get_primary_text should return first text derivative."""
        result = ExtractionResult(
            source_uri="test",
            mime_type="text/plain",
            derivatives=[
                Derivative(type="image", content=b"..."),
                Derivative(type="text", content="Primary text"),
            ],
        )

        assert result.get_primary_text() == "Primary text"

    def test_extraction_result_get_all_text(self):
        """get_all_text should concatenate all text derivatives."""
        result = ExtractionResult(
            source_uri="test",
            mime_type="text/plain",
            derivatives=[
                Derivative(type="text", content="First"),
                Derivative(type="image", content=b"..."),
                Derivative(type="text", content="Second"),
            ],
        )

        all_text = result.get_all_text()
        assert "First" in all_text
        assert "Second" in all_text

    def test_extraction_result_get_by_type(self):
        """get_derivatives_by_type should filter correctly."""
        result = ExtractionResult(
            source_uri="test",
            mime_type="application/pdf",
            derivatives=[
                Derivative(type="text", content="Text 1"),
                Derivative(type="image", content=b"Image 1"),
                Derivative(type="text", content="Text 2"),
                Derivative(type="image", content=b"Image 2"),
            ],
        )

        text_derivs = result.get_derivatives_by_type("text")
        assert len(text_derivs) == 2

        image_derivs = result.get_derivatives_by_type("image")
        assert len(image_derivs) == 2


class TestFileUploadConnector:
    """Tests for FileUploadConnector."""

    @pytest.mark.asyncio
    async def test_fetch_bytes(self):
        """fetch_bytes should create RawContent from bytes."""
        connector = FileUploadConnector()

        raw = await connector.fetch_bytes(
            content=b"Test content",
            filename="test.txt",
            mime_type="text/plain",
        )

        assert raw.content == b"Test content"
        assert raw.mime_type == "text/plain"
        assert raw.filename == "test.txt"
        assert raw.source_uri == "upload://test.txt"

    @pytest.mark.asyncio
    async def test_fetch_bytes_guess_mime(self):
        """fetch_bytes should guess MIME type from filename."""
        connector = FileUploadConnector()

        raw = await connector.fetch_bytes(
            content=b"PDF content",
            filename="document.pdf",
        )

        assert raw.mime_type == "application/pdf"

    @pytest.mark.asyncio
    async def test_fetch_bytes_unknown_mime(self):
        """fetch_bytes should use octet-stream for unknown types."""
        connector = FileUploadConnector()

        raw = await connector.fetch_bytes(
            content=b"Unknown content",
            filename="file.zzz_unknown",  # Truly unknown extension
        )

        assert raw.mime_type == "application/octet-stream"

    def test_connector_name(self):
        """FileUploadConnector should have correct name."""
        connector = FileUploadConnector()
        assert connector.name == "file_upload"


class TestExtractor:
    """Tests for Extractor base class."""

    def test_extractor_is_abstract(self):
        """Extractor should not be instantiable."""
        with pytest.raises(TypeError, match="abstract"):
            Extractor()

    def test_extractor_subclass_must_implement_extract(self):
        """Extractor subclass must implement extract."""

        class PartialExtractor(Extractor):
            name = "partial"
            supported_types = ["test/type"]

        with pytest.raises(TypeError, match="abstract"):
            PartialExtractor()

    def test_extractor_supports(self):
        """supports() should check against supported_types."""

        class TestExtractor(Extractor):
            name = "test"
            supported_types = ["text/plain", "text/markdown"]

            async def extract(self, raw):
                pass

        ext = TestExtractor()
        assert ext.supports("text/plain")
        assert ext.supports("text/markdown")
        assert not ext.supports("application/pdf")


class TestTextExtractor:
    """Tests for TextExtractor."""

    @pytest.mark.asyncio
    async def test_extract_plain_text(self):
        """TextExtractor should handle plain text."""
        extractor = TextExtractor()
        raw = RawContent(
            content=b"Hello, world!",
            mime_type="text/plain",
            source_uri="test://hello.txt",
        )

        result = await extractor.extract(raw)

        assert result.extraction_method == "text"
        assert len(result.derivatives) == 1
        assert result.derivatives[0].type == "text"
        assert result.derivatives[0].content == "Hello, world!"

    @pytest.mark.asyncio
    async def test_extract_markdown(self):
        """TextExtractor should handle markdown."""
        extractor = TextExtractor()
        raw = RawContent(
            content=b"# Header\n\nParagraph",
            mime_type="text/markdown",
            source_uri="test://doc.md",
        )

        result = await extractor.extract(raw)

        assert result.derivatives[0].type == "text"
        assert result.derivatives[0].format == "markdown"

    @pytest.mark.asyncio
    async def test_extract_utf8(self):
        """TextExtractor should handle UTF-8."""
        extractor = TextExtractor()
        raw = RawContent(
            content="Hello, 世界! 🌍".encode(),
            mime_type="text/plain",
            source_uri="test://unicode.txt",
        )

        result = await extractor.extract(raw)

        text = result.get_primary_text()
        assert "世界" in text
        assert "🌍" in text

    @pytest.mark.asyncio
    async def test_extract_metadata(self):
        """TextExtractor should produce metadata."""
        extractor = TextExtractor()
        raw = RawContent(
            content=b"Line 1\nLine 2\nLine 3",
            mime_type="text/plain",
            source_uri="test://lines.txt",
        )

        result = await extractor.extract(raw)

        assert result.auto_metadata["line_count"] == 3
        assert result.auto_metadata["char_count"] == 20

    def test_text_extractor_supported_types(self):
        """TextExtractor should support text/* types."""
        extractor = TextExtractor()

        assert extractor.supports("text/plain")
        assert extractor.supports("text/markdown")
        assert extractor.supports("text/csv")


class TestExtractorRegistry:
    """Tests for ExtractorRegistry."""

    def test_registry_empty(self):
        """Empty registry should have no extractors."""
        registry = ExtractorRegistry()

        assert len(registry) == 0
        assert registry.list_extractors() == []

    def test_registry_register(self):
        """register should add extractor for its types."""
        registry = ExtractorRegistry()
        extractor = TextExtractor()

        registry.register(extractor)

        assert len(registry) == 1
        assert "text/plain" in registry.list_supported_types()

    def test_registry_get_extractor(self):
        """get_extractor should return correct extractor."""
        registry = ExtractorRegistry()
        extractor = TextExtractor()
        registry.register(extractor)

        result = registry.get_extractor("text/plain")

        assert result is extractor

    def test_registry_get_extractor_with_params(self):
        """get_extractor should handle MIME params."""
        registry = ExtractorRegistry()
        registry.register(TextExtractor())

        # MIME type with charset parameter
        result = registry.get_extractor("text/plain; charset=utf-8")

        assert result is not None
        assert result.name == "text"

    def test_registry_get_extractor_wildcard(self):
        """get_extractor should support wildcard types."""
        registry = ExtractorRegistry()

        # TextExtractor registers text/* wildcard
        registry.register(TextExtractor())

        # Should match via wildcard
        result = registry.get_extractor("text/x-custom")
        assert result is not None

    def test_registry_get_extractor_not_found(self):
        """get_extractor should raise KeyError for unknown type."""
        registry = ExtractorRegistry()

        with pytest.raises(KeyError, match="No extractor registered"):
            registry.get_extractor("application/pdf")

    def test_registry_get_by_name(self):
        """get_by_name should return extractor by name."""
        registry = ExtractorRegistry()
        extractor = TextExtractor()
        registry.register(extractor)

        result = registry.get_by_name("text")

        assert result is extractor

    def test_registry_supports(self):
        """supports should check if type is registered."""
        registry = ExtractorRegistry()
        registry.register(TextExtractor())

        assert registry.supports("text/plain")
        assert not registry.supports("application/pdf")

    def test_registry_override(self):
        """Later registration should override earlier."""
        registry = ExtractorRegistry()

        # Custom text extractor
        class CustomTextExtractor(Extractor):
            name = "custom_text"
            supported_types = ["text/plain"]

            async def extract(self, raw):
                pass

        registry.register(TextExtractor())
        registry.register(CustomTextExtractor())

        result = registry.get_extractor("text/plain")
        assert result.name == "custom_text"

    def test_registry_default(self):
        """default() should return registry with built-ins."""
        registry = ExtractorRegistry.default()

        # Should have at least TextExtractor
        assert registry.supports("text/plain")


class TestExtractionError:
    """Tests for ExtractionError."""

    def test_extraction_error_message(self):
        """ExtractionError should include details."""
        error = ExtractionError(
            message="Failed to parse PDF",
            extractor_name="pdf",
            source_uri="upload://doc.pdf",
        )

        error_str = str(error)
        assert "Failed to parse PDF" in error_str
        assert "pdf" in error_str
        assert "doc.pdf" in error_str


class TestHTMLExtractor:
    """Tests for HTMLExtractor."""

    @pytest.mark.asyncio
    async def test_extract_html(self):
        """HTMLExtractor should handle HTML content."""
        pytest.importorskip("bs4", reason="BeautifulSoup required")

        extractor = HTMLExtractor()
        raw = RawContent(
            content=b"<html><head><title>Test</title></head><body><h1>Hello</h1><p>World</p></body></html>",
            mime_type="text/html",
            source_uri="test://page.html",
        )

        result = await extractor.extract(raw)

        assert result.extraction_method == "html"
        text = result.get_primary_text()
        assert "Hello" in text
        assert "World" in text

    @pytest.mark.asyncio
    async def test_extract_html_strips_scripts(self):
        """HTMLExtractor should remove script tags."""
        pytest.importorskip("bs4", reason="BeautifulSoup required")

        extractor = HTMLExtractor()
        raw = RawContent(
            content=b"<html><body><script>alert('bad')</script><p>Good content</p></body></html>",
            mime_type="text/html",
            source_uri="test://page.html",
        )

        result = await extractor.extract(raw)
        text = result.get_primary_text()

        assert "Good content" in text
        assert "alert" not in text

    def test_html_extractor_supported_types(self):
        """HTMLExtractor should support HTML types."""
        extractor = HTMLExtractor()
        assert extractor.supports("text/html")
        assert extractor.supports("application/xhtml+xml")


class TestPDFExtractor:
    """Tests for PDFExtractor."""

    def test_pdf_extractor_supported_types(self):
        """PDFExtractor should support PDF type."""
        extractor = PDFExtractor()
        assert extractor.supports("application/pdf")

    def test_pdf_extractor_init(self):
        """PDFExtractor should accept configuration."""
        extractor = PDFExtractor(mistral_api_key="test-key", max_pages=100)
        assert extractor.mistral_api_key == "test-key"
        assert extractor.max_pages == 100


class TestDocxExtractor:
    """Tests for DocxExtractor."""

    def test_docx_extractor_supported_types(self):
        """DocxExtractor should support DOCX type."""
        extractor = DocxExtractor()
        assert extractor.supports(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )


class TestExtractorRegistryWithBuiltins:
    """Tests for ExtractorRegistry with built-in extractors."""

    def test_registry_default_has_text(self):
        """default() should include TextExtractor."""
        registry = ExtractorRegistry.default()
        assert registry.supports("text/plain")

        extractor = registry.get_extractor("text/plain")
        assert extractor.name == "text"

    def test_registry_default_has_html(self):
        """default() should include HTMLExtractor."""
        registry = ExtractorRegistry.default()
        assert registry.supports("text/html")

        extractor = registry.get_extractor("text/html")
        assert extractor.name == "html"

    def test_registry_default_has_pdf(self):
        """default() should include PDFExtractor."""
        registry = ExtractorRegistry.default()
        assert registry.supports("application/pdf")

        extractor = registry.get_extractor("application/pdf")
        assert extractor.name == "pdf"

    def test_registry_default_has_docx(self):
        """default() should include DocxExtractor."""
        registry = ExtractorRegistry.default()
        mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        assert registry.supports(mime)

        extractor = registry.get_extractor(mime)
        assert extractor.name == "docx"


class TestModuleImports:
    """Tests that module exports are correct."""

    def test_ingest_module_exports(self):
        """Main ingest module should export all key classes."""
        from agentic import ingest

        # Models
        assert hasattr(ingest, "RawContent")
        assert hasattr(ingest, "Derivative")
        assert hasattr(ingest, "ExtractionResult")
        assert hasattr(ingest, "ContentItem")

        # Connectors
        assert hasattr(ingest, "Connector")
        assert hasattr(ingest, "FileUploadConnector")

        # Extractors
        assert hasattr(ingest, "Extractor")
        assert hasattr(ingest, "ExtractionError")
        assert hasattr(ingest, "ExtractorRegistry")

        # Built-in extractors
        assert hasattr(ingest, "TextExtractor")
        assert hasattr(ingest, "HTMLExtractor")
        assert hasattr(ingest, "PDFExtractor")
        assert hasattr(ingest, "DocxExtractor")
