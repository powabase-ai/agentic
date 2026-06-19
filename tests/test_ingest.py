"""
Tests for the Content Ingestion module.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

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
from agentic.ingest.extractor.image import ImageExtractor
from agentic.ingest.extractor.pptx import PptxExtractor
from agentic.ingest.extractor.xlsx import XlsxExtractor


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
        """TextExtractor should support text/* types (except text/plain, handled by TxtExtractor)."""
        extractor = TextExtractor()

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
        assert "text/markdown" in registry.list_supported_types()

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


class TestPDFExtractorPaddleOCR:
    """Tests for PDFExtractor PaddleOCR-VL method."""

    def test_paddleocr_requires_api_key(self):
        """PaddleOCR should fail without API key."""
        extractor = PDFExtractor()
        raw = RawContent(
            content=b"%PDF-1.4 fake",
            mime_type="application/pdf",
            source_uri="upload://test.pdf",
            metadata={"extraction_model": "paddleocr"},
        )
        with pytest.raises(ExtractionError, match="PADDLEOCR_API_KEY"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(extractor.extract(raw))

    @pytest.mark.asyncio
    async def test_paddleocr_extraction(self):
        """PaddleOCR should produce correct derivatives from API response."""
        extractor = PDFExtractor(paddleocr_api_key="test-key")
        raw = RawContent(
            content=b"%PDF-1.4 fake",
            mime_type="application/pdf",
            source_uri="upload://test.pdf",
            metadata={"extraction_model": "paddleocr"},
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {
                "layoutParsingResults": [
                    {"markdown": {"text": "# Page 1\nContent here", "images": {}}},
                    {"markdown": {"text": "# Page 2\nMore content", "images": {}}},
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response), \
             patch.object(extractor, "_render_page_images", return_value=[]):
            result = await extractor.extract(raw)

        assert result.extraction_method == "paddleocr_vl"
        assert result.auto_metadata["page_count"] == 2

        # Check markdown derivative
        md_derivs = [d for d in result.derivatives if d.type == "markdown"]
        assert len(md_derivs) == 1
        assert "Page 1" in md_derivs[0].content
        assert "Page 2" in md_derivs[0].content

        # Check page_text derivatives
        page_derivs = [d for d in result.derivatives if d.type == "page_text"]
        assert len(page_derivs) == 2
        assert page_derivs[0].page == 1
        assert page_derivs[1].page == 2

    @pytest.mark.asyncio
    async def test_paddleocr_strips_image_refs(self):
        """PaddleOCR should strip image references from markdown."""
        extractor = PDFExtractor(paddleocr_api_key="test-key")
        raw = RawContent(
            content=b"%PDF-1.4 fake",
            mime_type="application/pdf",
            source_uri="upload://test.pdf",
            metadata={"extraction_model": "paddleocr"},
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {
                "layoutParsingResults": [
                    {
                        "markdown": {
                            "text": "Text before ![figure](https://temp-url/img1.png) and after ![chart](img1) end",
                            "images": {"img1": "https://temp-url/img1.png"},
                        }
                    },
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response), \
             patch.object(extractor, "_render_page_images", return_value=[]):
            result = await extractor.extract(raw)

        # Image refs should be stripped — both URL-based and path-based patterns
        md = result.derivatives[0].content
        assert "https://temp-url" not in md
        assert "![figure]" not in md
        assert "![chart]" not in md
        assert "Text before" in md
        assert "and after" in md
        assert "end" in md


class TestPDFExtractorLightOnOCR:
    """Tests for PDFExtractor LightOnOCR method."""

    def test_lighton_requires_api_key(self):
        """LightOnOCR should fail without API key."""
        extractor = PDFExtractor()
        raw = RawContent(
            content=b"%PDF-1.4 fake",
            mime_type="application/pdf",
            source_uri="upload://test.pdf",
            metadata={"extraction_model": "lighton"},
        )
        with pytest.raises(ExtractionError, match="LIGHTON_API_KEY"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(extractor.extract(raw))

    @pytest.mark.asyncio
    async def test_lighton_extraction(self):
        """LightOnOCR should produce correct derivatives from API response."""
        extractor = PDFExtractor(lighton_api_key="test-key")
        raw = RawContent(
            content=b"%PDF-1.4 fake",
            mime_type="application/pdf",
            source_uri="upload://test.pdf",
            metadata={"extraction_model": "lighton"},
        )

        # Mock page image derivatives (simulating _render_page_images output)
        mock_img_derivs = [
            Derivative(type="image", content=b"fake-png-1", format="png", page=1),
            Derivative(type="image", content=b"fake-png-2", format="png", page=2),
        ]

        mock_response_1 = MagicMock()
        mock_response_1.status_code = 200
        mock_response_1.json.return_value = {
            "choices": [{"message": {"content": "# Page 1\nOCR text"}}]
        }
        mock_response_1.raise_for_status = MagicMock()

        mock_response_2 = MagicMock()
        mock_response_2.status_code = 200
        mock_response_2.json.return_value = {
            "choices": [{"message": {"content": "# Page 2\nMore OCR text"}}]
        }
        mock_response_2.raise_for_status = MagicMock()

        with patch.object(extractor, "_render_page_images", return_value=mock_img_derivs), \
             patch("requests.post", side_effect=[mock_response_1, mock_response_2]):
            result = await extractor.extract(raw)

        assert result.extraction_method == "lighton_ocr"
        assert result.auto_metadata["page_count"] == 2

        # Check markdown derivative
        md_derivs = [d for d in result.derivatives if d.type == "markdown"]
        assert len(md_derivs) == 1
        assert "Page 1" in md_derivs[0].content
        assert "Page 2" in md_derivs[0].content

        # Check page_text derivatives
        page_derivs = [d for d in result.derivatives if d.type == "page_text"]
        assert len(page_derivs) == 2
        assert page_derivs[0].page == 1
        assert page_derivs[1].page == 2

        # Check image derivatives are included
        img_derivs = [d for d in result.derivatives if d.type == "image"]
        assert len(img_derivs) == 2

    @pytest.mark.asyncio
    async def test_lighton_requires_fitz_for_rendering(self):
        """LightOnOCR should fail if page rendering returns no images."""
        extractor = PDFExtractor(lighton_api_key="test-key")
        raw = RawContent(
            content=b"%PDF-1.4 fake",
            mime_type="application/pdf",
            source_uri="upload://test.pdf",
            metadata={"extraction_model": "lighton"},
        )

        with patch.object(extractor, "_render_page_images", return_value=[]):
            with pytest.raises(ExtractionError, match="PyMuPDF"):
                await extractor.extract(raw)


class TestPDFExtractorLlamaParse:
    """Tests for PDFExtractor LlamaParse method."""

    def test_llamaparse_requires_api_key(self):
        """LlamaParse should fail without API key."""
        extractor = PDFExtractor()
        raw = RawContent(
            content=b"%PDF-1.4 fake",
            mime_type="application/pdf",
            source_uri="upload://test.pdf",
            metadata={"extraction_model": "llamaparse"},
        )
        with pytest.raises(ExtractionError, match="LLAMAPARSE_API_KEY"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(extractor.extract(raw))

    @pytest.mark.asyncio
    async def test_llamaparse_extraction(self):
        """LlamaParse should upload, poll, and build derivatives from the result."""
        extractor = PDFExtractor(llamaparse_api_key="test-key")
        raw = RawContent(
            content=b"%PDF-1.4 fake",
            mime_type="application/pdf",
            source_uri="upload://test.pdf",
            metadata={"extraction_model": "llamaparse"},
        )

        mock_img_derivs = [
            Derivative(type="image", content=b"fake-png-1", format="png", page=1),
            Derivative(type="image", content=b"fake-png-2", format="png", page=2),
        ]

        upload_resp = MagicMock()
        upload_resp.status_code = 200
        upload_resp.json.return_value = {"id": "file-123"}

        parse_resp = MagicMock()
        parse_resp.status_code = 200
        parse_resp.json.return_value = {"id": "job-123", "status": "PENDING"}

        status_resp = MagicMock()
        status_resp.status_code = 200
        status_resp.json.return_value = {"job": {"status": "COMPLETED"}}

        result_resp = MagicMock()
        result_resp.status_code = 200
        result_resp.json.return_value = {
            "job": {"status": "COMPLETED"},
            "markdown": {
                "pages": [
                    {"page_number": 1, "markdown": "# Page 1\nParsed text"},
                    {"page_number": 2, "markdown": "# Page 2\nMore parsed text"},
                ]
            },
        }

        with patch.object(extractor, "_render_page_images", return_value=mock_img_derivs), \
             patch("requests.post", side_effect=[upload_resp, parse_resp]), \
             patch("requests.get", side_effect=[status_resp, result_resp]):
            result = await extractor.extract(raw)

        assert result.extraction_method == "llamaparse_ocr"
        assert result.auto_metadata["page_count"] == 2

        md_derivs = [d for d in result.derivatives if d.type == "markdown"]
        assert len(md_derivs) == 1
        assert "Page 1" in md_derivs[0].content
        assert "Page 2" in md_derivs[0].content

        page_derivs = [d for d in result.derivatives if d.type == "page_text"]
        assert len(page_derivs) == 2
        assert page_derivs[0].page == 1
        assert page_derivs[1].page == 2

        img_derivs = [d for d in result.derivatives if d.type == "image"]
        assert len(img_derivs) == 2

    @pytest.mark.asyncio
    async def test_llamaparse_job_failure_raises(self):
        """A terminal job error should raise ExtractionError (no infinite poll)."""
        extractor = PDFExtractor(llamaparse_api_key="test-key")
        raw = RawContent(
            content=b"%PDF-1.4 fake",
            mime_type="application/pdf",
            source_uri="upload://test.pdf",
            metadata={"extraction_model": "llamaparse"},
        )

        upload_resp = MagicMock()
        upload_resp.status_code = 200
        upload_resp.json.return_value = {"id": "file-err"}

        parse_resp = MagicMock()
        parse_resp.status_code = 200
        parse_resp.json.return_value = {"id": "job-err", "status": "PENDING"}

        status_resp = MagicMock()
        status_resp.status_code = 200
        status_resp.json.return_value = {"job": {"status": "FAILED"}}

        with patch("requests.post", side_effect=[upload_resp, parse_resp]), \
             patch("requests.get", return_value=status_resp):
            with pytest.raises(ExtractionError, match="LlamaParse"):
                await extractor._extract_llamaparse(raw)

    @pytest.mark.asyncio
    async def test_llamaparse_4xx_fails_fast_with_body(self):
        """A 4xx response raises ExtractionError with the body — deterministic,
        so _try_method does not retry (avoids re-uploading + re-billing)."""
        extractor = PDFExtractor(llamaparse_api_key="test-key")
        raw = RawContent(
            content=b"%PDF-1.4 fake",
            mime_type="application/pdf",
            source_uri="upload://test.pdf",
            metadata={"extraction_model": "llamaparse"},
        )

        bad_resp = MagicMock()
        bad_resp.status_code = 422
        bad_resp.text = "Unprocessable Entity: bad upload field"

        with patch("requests.post", return_value=bad_resp) as mock_post:
            with pytest.raises(ExtractionError, match="422"):
                await extractor._extract_llamaparse(raw)
        # Failed fast on the upload — no parse job was ever started.
        assert mock_post.call_count == 1

    @pytest.mark.asyncio
    async def test_llamaparse_poll_timeout(self, monkeypatch):
        """A perpetually-PENDING job gives up at LLAMAPARSE_MAX_POLL_SECONDS
        instead of blocking forever."""
        from agentic.knowledge import model_config

        monkeypatch.setattr(model_config, "LLAMAPARSE_MAX_POLL_SECONDS", 10)
        monkeypatch.setattr(model_config, "LLAMAPARSE_POLL_INTERVAL", 5)

        extractor = PDFExtractor(llamaparse_api_key="lp-key")
        raw = RawContent(
            content=b"%PDF-1.4 fake",
            mime_type="application/pdf",
            source_uri="upload://t.pdf",
            metadata={"extraction_model": "llamaparse"},
        )
        upload_resp = MagicMock()
        upload_resp.status_code = 200
        upload_resp.json.return_value = {"id": "file-1"}
        parse_resp = MagicMock()
        parse_resp.status_code = 200
        parse_resp.json.return_value = {"id": "job-1", "status": "PENDING"}
        pending = MagicMock()
        pending.status_code = 200
        pending.json.return_value = {"job": {"status": "PENDING"}}

        with patch("requests.post", side_effect=[upload_resp, parse_resp]), \
             patch("requests.get", return_value=pending), \
             patch("asyncio.sleep", new=AsyncMock()):
            with pytest.raises(ExtractionError, match="did not complete within"):
                await extractor._extract_llamaparse(raw)

    @pytest.mark.asyncio
    async def test_llamaparse_transient_5xx_does_not_resubmit_job(self):
        """A sticky 5xx during polling exhausts the internal GET retry and fails
        WITHOUT re-uploading or starting a new (billable) parse job."""
        extractor = PDFExtractor(llamaparse_api_key="lp-key")
        raw = RawContent(
            content=b"%PDF-1.4 fake",
            mime_type="application/pdf",
            source_uri="upload://t.pdf",
            metadata={"extraction_model": "llamaparse"},
        )
        upload_resp = MagicMock()
        upload_resp.status_code = 200
        upload_resp.json.return_value = {"id": "file-1"}
        parse_resp = MagicMock()
        parse_resp.status_code = 200
        parse_resp.json.return_value = {"id": "job-1", "status": "PENDING"}
        five_xx = MagicMock()
        five_xx.status_code = 503
        five_xx.text = "upstream unavailable"

        with patch("requests.post", side_effect=[upload_resp, parse_resp]) as mock_post, \
             patch("requests.get", return_value=five_xx), \
             patch("asyncio.sleep", new=AsyncMock()):
            with pytest.raises(ExtractionError, match="after retries"):
                await extractor._extract_llamaparse(raw)
        # Upload + parse-start happened exactly once each — the transient poll
        # error did NOT spawn a second billable parse job.
        assert mock_post.call_count == 2


class TestImageExtractorLlamaParse:
    """Image uploads must honour an explicit LlamaParse selection (not just PDFs)."""

    @pytest.mark.asyncio
    async def test_image_uses_llamaparse_when_selected(self):
        extractor = ImageExtractor(mistral_api_key="m-key", llamaparse_api_key="lp-key")
        raw = RawContent(
            content=b"\x89PNG fake",
            mime_type="image/png",
            source_uri="upload://pic.png",
            filename="pic.png",
            metadata={"extraction_model": "llamaparse"},
        )

        upload_resp = MagicMock()
        upload_resp.status_code = 200
        upload_resp.json.return_value = {"id": "file-1"}
        parse_resp = MagicMock()
        parse_resp.status_code = 200
        parse_resp.json.return_value = {"id": "job-1", "status": "PENDING"}
        status_resp = MagicMock()
        status_resp.status_code = 200
        status_resp.json.return_value = {"job": {"status": "COMPLETED"}}
        result_resp = MagicMock()
        result_resp.status_code = 200
        result_resp.json.return_value = {
            "markdown": {"pages": [{"page_number": 1, "markdown": "# OCR text"}]}
        }

        with patch("requests.post", side_effect=[upload_resp, parse_resp]), \
             patch("requests.get", side_effect=[status_resp, result_resp]):
            result = await extractor.extract(raw)

        assert result.extraction_method == "llamaparse_ocr"
        md = [d for d in result.derivatives if d.type == "markdown"]
        assert md and "OCR text" in md[0].content
        # The original image is preserved for image-mode retrieval.
        assert any(d.type == "image" for d in result.derivatives)

    @pytest.mark.asyncio
    async def test_image_falls_back_to_mistral_when_llamaparse_fails(self):
        extractor = ImageExtractor(mistral_api_key="m-key", llamaparse_api_key="lp-key")
        raw = RawContent(
            content=b"\x89PNG fake",
            mime_type="image/png",
            source_uri="upload://pic.png",
            metadata={"extraction_model": "llamaparse"},
        )
        sentinel = ExtractionResult(
            source_uri="upload://pic.png",
            mime_type="image/png",
            derivatives=[],
            extraction_method="mistral_ocr",
        )
        with patch.object(
            extractor, "_extract_llamaparse",
            side_effect=ExtractionError("boom", extractor_name="image", source_uri="upload://pic.png"),
        ), patch.object(extractor, "_extract_mistral", AsyncMock(return_value=sentinel)) as mock_m:
            result = await extractor.extract(raw)

        mock_m.assert_awaited_once()
        assert result.extraction_method == "mistral_ocr"
        # Fallback must be observable so the worker flips to attention_required.
        assert result.auto_metadata["requested_method"] == "llamaparse"
        assert "boom" in result.auto_metadata["fallback_reason"]


class TestPDFExtractorOpenDataLoader:
    """Tests for PDFExtractor OpenDataLoader method."""

    @pytest.mark.asyncio
    async def test_opendataloader_extraction(self):
        """OpenDataLoader should produce per-page page_text derivatives and joined markdown."""
        extractor = PDFExtractor()
        raw = RawContent(
            content=b"%PDF-1.4 fake",
            mime_type="application/pdf",
            source_uri="upload://test.pdf",
            metadata={"extraction_model": "opendataloader"},
        )

        page1 = "# Document Title\n\nFirst page content."
        page2 = "## Chapter 2\n\nSecond page content."
        page3 = "## Chapter 3\n\nThird page content."
        page_contents = [page1, page2, page3]

        # Track which page is being converted (by call order)
        convert_call_count = {"n": 0}

        def mock_convert(input_path, output_dir, format, quiet):
            import os
            assert isinstance(input_path, list), "input_path must be a list"
            assert len(input_path) == 1
            idx = convert_call_count["n"]
            convert_call_count["n"] += 1
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "page.md"), "w") as f:
                f.write(page_contents[idx])

        # Mock fitz: first call opens src_doc (with stream= kwarg),
        # subsequent calls create empty single-page docs (no args)
        mock_src_doc = MagicMock()
        mock_src_doc.__len__ = lambda self: 3

        mock_fitz = MagicMock()

        def mock_fitz_open(*args, **kwargs):
            if "stream" in kwargs:
                return mock_src_doc  # src_doc = fitz.open(stream=..., filetype="pdf")
            return MagicMock()  # single = fitz.open()  — empty doc for insert_pdf

        mock_fitz.open.side_effect = mock_fitz_open

        with patch.dict("sys.modules", {
                "opendataloader_pdf": MagicMock(convert=mock_convert),
                "fitz": mock_fitz,
             }), \
             patch.object(extractor, "_render_page_images", return_value=[
                 Derivative(type="image", content=b"fake-png", format="png", page=1),
             ]):
            result = await extractor.extract(raw)

        assert result.extraction_method == "opendataloader"
        assert result.auto_metadata["page_count"] == 3
        expected_fulltext = "\n\n".join(page_contents)
        assert result.auto_metadata["char_count"] == len(expected_fulltext)

        md_derivs = [d for d in result.derivatives if d.type == "markdown"]
        assert len(md_derivs) == 1
        assert md_derivs[0].content == expected_fulltext
        assert md_derivs[0].format == "markdown"

        page_derivs = [d for d in result.derivatives if d.type == "page_text"]
        assert len(page_derivs) == 3
        assert page_derivs[0].content == page1
        assert page_derivs[0].page == 1
        assert page_derivs[1].content == page2
        assert page_derivs[1].page == 2
        assert page_derivs[2].content == page3
        assert page_derivs[2].page == 3

        img_derivs = [d for d in result.derivatives if d.type == "image"]
        assert len(img_derivs) == 1

    @pytest.mark.asyncio
    async def test_opendataloader_import_error(self):
        """OpenDataLoader should raise ExtractionError when not installed."""
        extractor = PDFExtractor()
        raw = RawContent(
            content=b"%PDF-1.4 fake",
            mime_type="application/pdf",
            source_uri="upload://test.pdf",
            metadata={"extraction_model": "opendataloader"},
        )

        with patch.dict("sys.modules", {"opendataloader_pdf": None}):
            with pytest.raises(ExtractionError, match="opendataloader-pdf"):
                await extractor.extract(raw)


class TestPDFExtractorOCRLocalFallback:
    """Tests for OCR-to-local fallback behavior."""

    @pytest.mark.asyncio
    async def test_cloud_ocr_falls_back_to_local(self):
        """When a cloud OCR method fails, should fall back to local methods."""
        extractor = PDFExtractor(mistral_api_key="test-key")
        raw = RawContent(
            content=b"%PDF-1.4 fake",
            mime_type="application/pdf",
            source_uri="upload://test.pdf",
            metadata={"extraction_model": "mistral"},
        )

        mock_result = ExtractionResult(
            source_uri="upload://test.pdf",
            mime_type="application/pdf",
            derivatives=[Derivative(type="text", content="fallback text")],
            extraction_method="fitz",
        )

        with patch.object(extractor, "_extract_mistral", side_effect=Exception("API down")), \
             patch.object(extractor, "_extract_opendataloader", side_effect=Exception("no java")), \
             patch.object(extractor, "_extract_fitz", return_value=mock_result):
            result = await extractor.extract(raw)

        assert result.extraction_method == "fitz"

    @pytest.mark.asyncio
    async def test_local_method_no_circular_fallback(self):
        """When a local method (fitz) fails explicitly, should NOT fall back."""
        extractor = PDFExtractor()
        raw = RawContent(
            content=b"%PDF-1.4 fake",
            mime_type="application/pdf",
            source_uri="upload://test.pdf",
            metadata={"extraction_model": "fitz"},
        )

        with patch.object(extractor, "_extract_fitz", side_effect=ExtractionError(
            "fitz broken", extractor_name="pdf", source_uri="upload://test.pdf"
        )):
            with pytest.raises(ExtractionError, match="fitz broken"):
                await extractor.extract(raw)

    @pytest.mark.asyncio
    async def test_cloud_ocr_all_fallbacks_fail(self):
        """When cloud OCR and all local fallbacks fail, should raise with context."""
        extractor = PDFExtractor(mistral_api_key="test-key")
        raw = RawContent(
            content=b"%PDF-1.4 fake",
            mime_type="application/pdf",
            source_uri="upload://test.pdf",
            metadata={"extraction_model": "mistral"},
        )

        with patch.object(extractor, "_extract_mistral", side_effect=Exception("API down")), \
             patch.object(extractor, "_extract_opendataloader", side_effect=Exception("no java")), \
             patch.object(extractor, "_extract_fitz", side_effect=Exception("fitz fail")), \
             patch.object(extractor, "_extract_pdfplumber", side_effect=Exception("plumber fail")):
            with pytest.raises(ExtractionError, match="all local fallbacks failed"):
                await extractor.extract(raw)


class TestPDFExtractorAutoMode:
    """Test the auto-mode fallback chain: lighton-first, paddleocr/mistral excluded."""

    @pytest.mark.asyncio
    async def test_auto_mode_tries_lighton_first_then_skips_paddleocr_and_mistral(self):
        """Auto mode must try lighton FIRST (the default OCR head this PR ships),
        walk the local chain in order, and never reach paddleocr/mistral.

        Asserting only the negatives (paddle/mistral not called) is not enough:
        with every chain method patched-to-fail, a regression that dropped lighton
        from the chain entirely would still raise "All extraction methods failed"
        and still skip paddle/mistral — so the negative-only test would pass while
        the headline change silently regressed. We therefore record call order and
        assert lighton actually ran, and ran first.
        """
        extractor = PDFExtractor(
            paddleocr_api_key="key",
            lighton_api_key="key",
            mistral_api_key="key",
        )
        raw = RawContent(
            content=b"%PDF-1.4 fake",
            mime_type="application/pdf",
            source_uri="upload://test.pdf",
            metadata={"extraction_model": "auto"},
        )

        call_order: list[str] = []

        def _record(name):
            # Raise ExtractionError (re-raised without retry) so each method is
            # invoked exactly once and the call order is unambiguous.
            def _fail(*args, **kwargs):
                call_order.append(name)
                raise ExtractionError(
                    f"{name} fail", extractor_name="pdf", source_uri="upload://test.pdf"
                )
            return _fail

        with patch.object(extractor, "_extract_lighton", side_effect=_record("lighton")) as mock_lighton, \
             patch.object(extractor, "_extract_opendataloader", side_effect=_record("opendataloader")), \
             patch.object(extractor, "_extract_fitz", side_effect=_record("fitz")), \
             patch.object(extractor, "_extract_pdfplumber", side_effect=_record("pdfplumber")), \
             patch.object(extractor, "_extract_paddleocr") as mock_paddle, \
             patch.object(extractor, "_extract_mistral") as mock_mistral:
            with pytest.raises(ExtractionError, match="All extraction methods failed"):
                await extractor.extract(raw)

        # The change this PR ships: lighton is the head and is tried first.
        mock_lighton.assert_called_once()
        assert call_order == ["lighton", "opendataloader", "fitz", "pdfplumber"]
        # paddleocr/mistral are explicit-only — never reached in auto mode.
        mock_paddle.assert_not_called()
        mock_mistral.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_mode_strips_lighton_when_no_key(self, caplog):
        """Without LIGHTON_API_KEY, auto mode skips lighton and proceeds to the
        local chain — but the skip is observable (INFO log), not silent.

        lighton is the advertised default head; an operator who forgets the key
        must not silently get local-only output (no OCR on scanned PDFs) with
        nothing in the logs. The skip is an expected condition, so it is logged
        at INFO, not escalated to WARNING/ERROR.
        """
        extractor = PDFExtractor()  # no lighton_api_key
        raw = RawContent(
            content=b"%PDF-1.4 fake",
            mime_type="application/pdf",
            source_uri="upload://test.pdf",
            metadata={"extraction_model": "auto"},
        )

        mock_result = ExtractionResult(
            source_uri="upload://test.pdf",
            mime_type="application/pdf",
            derivatives=[Derivative(type="markdown", content="odl text")],
            extraction_method="opendataloader",
        )

        with caplog.at_level(logging.INFO):
            with patch.object(extractor, "_extract_lighton") as mock_lighton, \
                 patch.object(extractor, "_extract_opendataloader", return_value=mock_result):
                result = await extractor.extract(raw)

        # lighton is dropped from the chain — never invoked — and opendataloader
        # becomes the head.
        mock_lighton.assert_not_called()
        assert result.extraction_method == "opendataloader"
        # The skip is logged at INFO and explains itself...
        assert any(
            r.levelno == logging.INFO and "lighton" in r.message.lower()
            for r in caplog.records
        )
        # ...and is NOT escalated to a warning/error (it is an expected config).
        assert not any(r.levelno >= logging.WARNING for r in caplog.records)


class TestDocxExtractor:
    """Tests for DocxExtractor."""

    def test_docx_extractor_supported_types(self):
        """DocxExtractor should support DOCX type."""
        extractor = DocxExtractor()
        assert extractor.supports(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    # The original bug dropped ANY non-default method (not just llamaparse), so
    # parameterize across cloud-OCR methods to guard the whole class.
    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["llamaparse", "mistral", "lighton", "paddleocr"])
    async def test_docx_extract_forwards_extraction_model_to_pdf(self, method):
        """DocxExtractor must propagate extraction_model to the PDF stage, else
        the chosen method is silently ignored for DOCX."""
        mock_pdf_extractor = MagicMock()
        mock_pdf_extractor.extract = AsyncMock(
            return_value=ExtractionResult(
                source_uri="upload://doc.docx",
                mime_type="application/pdf",
                derivatives=[Derivative(type="text", content="Doc content")],
                extraction_method=f"{method}_ocr",
            )
        )
        extractor = DocxExtractor(pdf_extractor=mock_pdf_extractor)

        with patch.object(extractor, "_convert_to_pdf", return_value=b"fake pdf bytes"):
            raw = RawContent(
                content=b"fake docx bytes",
                mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                source_uri="upload://doc.docx",
                filename="doc.docx",
                metadata={"extraction_model": method},
            )
            await extractor.extract(raw)

        pdf_raw = mock_pdf_extractor.extract.call_args[0][0]
        assert pdf_raw.metadata.get("extraction_model") == method


class TestXlsxExtractor:
    """Tests for XlsxExtractor."""

    def test_xlsx_extractor_supported_types(self):
        """XlsxExtractor should support XLSX and XLS types."""
        extractor = XlsxExtractor()
        assert extractor.supports(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        assert extractor.supports("application/vnd.ms-excel")

    def test_xlsx_extractor_name(self):
        """XlsxExtractor should have correct name."""
        extractor = XlsxExtractor()
        assert extractor.name == "xlsx"

    @pytest.mark.asyncio
    async def test_xlsx_extract_produces_derivatives(self):
        """XlsxExtractor should produce markdown and text derivatives."""
        extractor = XlsxExtractor()
        mock_markdown = "| Col1 | Col2 |\n|------|------|\n| A | B |"

        with patch.object(extractor, "_convert", return_value=mock_markdown):
            raw = RawContent(
                content=b"fake xlsx bytes",
                mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                source_uri="upload://data.xlsx",
                filename="data.xlsx",
            )
            result = await extractor.extract(raw)

        assert result.extraction_method == "markitdown-xlsx"
        assert result.source_uri == "upload://data.xlsx"
        assert len(result.derivatives) == 2
        assert result.derivatives[0].type == "markdown"
        assert result.derivatives[0].format == "markdown"
        assert result.derivatives[0].content == mock_markdown
        assert result.derivatives[1].type == "text"
        assert result.derivatives[1].format == "plain"
        assert result.auto_metadata["char_count"] == len(mock_markdown)

    @pytest.mark.asyncio
    async def test_xlsx_extract_error_handling(self):
        """XlsxExtractor should wrap errors in ExtractionError."""
        extractor = XlsxExtractor()

        with patch.object(extractor, "_convert", side_effect=ValueError("bad file")):
            raw = RawContent(
                content=b"bad bytes",
                mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                source_uri="upload://bad.xlsx",
            )
            with pytest.raises(ExtractionError, match="MarkItDown XLSX conversion failed"):
                await extractor.extract(raw)


class TestPptxExtractor:
    """Tests for PptxExtractor."""

    def test_pptx_extractor_supported_types(self):
        """PptxExtractor should support PPTX type."""
        extractor = PptxExtractor()
        assert extractor.supports(
            "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        )

    def test_pptx_extractor_name(self):
        """PptxExtractor should have correct name."""
        extractor = PptxExtractor()
        assert extractor.name == "pptx"

    @pytest.mark.asyncio
    async def test_pptx_extract_delegates_to_pdf(self):
        """PptxExtractor should convert to PDF and delegate to PDFExtractor."""
        mock_pdf_extractor = MagicMock()
        mock_pdf_result = ExtractionResult(
            source_uri="upload://slides.pptx",
            mime_type="application/pdf",
            derivatives=[Derivative(type="text", content="Slide content")],
            extraction_method="pdf-lighton-ocr",
        )
        mock_pdf_extractor.extract = AsyncMock(return_value=mock_pdf_result)

        extractor = PptxExtractor(pdf_extractor=mock_pdf_extractor)

        with patch.object(extractor, "_convert_to_pdf", return_value=b"fake pdf bytes"):
            raw = RawContent(
                content=b"fake pptx bytes",
                mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                source_uri="upload://slides.pptx",
                filename="slides.pptx",
                metadata={"extraction_model": "llamaparse"},
            )
            result = await extractor.extract(raw)

        # Verify PDF extractor was called with PDF RawContent
        mock_pdf_extractor.extract.assert_called_once()
        pdf_raw = mock_pdf_extractor.extract.call_args[0][0]
        assert pdf_raw.mime_type == "application/pdf"
        assert pdf_raw.content == b"fake pdf bytes"
        assert pdf_raw.filename == "slides.pdf"
        # The chosen extraction_model must reach the PDF stage (else selecting
        # llamaparse/mistral/etc. on a PPTX is silently ignored).
        assert pdf_raw.metadata.get("extraction_model") == "llamaparse"

        # Verify result is tagged correctly
        assert result.extraction_method == "pptx-via-pdf"
        assert result.mime_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        assert result.auto_metadata["pdf_extraction_method"] == "pdf-lighton-ocr"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["llamaparse", "mistral", "lighton", "paddleocr"])
    async def test_pptx_forwards_extraction_model_to_pdf(self, method):
        """The chosen method must reach the PDF stage for any non-default value."""
        mock_pdf_extractor = MagicMock()
        mock_pdf_extractor.extract = AsyncMock(
            return_value=ExtractionResult(
                source_uri="upload://slides.pptx",
                mime_type="application/pdf",
                derivatives=[Derivative(type="text", content="Slide content")],
                extraction_method=f"{method}_ocr",
            )
        )
        extractor = PptxExtractor(pdf_extractor=mock_pdf_extractor)

        with patch.object(extractor, "_convert_to_pdf", return_value=b"fake pdf bytes"):
            raw = RawContent(
                content=b"fake pptx bytes",
                mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                source_uri="upload://slides.pptx",
                filename="slides.pptx",
                metadata={"extraction_model": method},
            )
            await extractor.extract(raw)

        pdf_raw = mock_pdf_extractor.extract.call_args[0][0]
        assert pdf_raw.metadata.get("extraction_model") == method

    @pytest.mark.asyncio
    async def test_pptx_convert_to_pdf_libreoffice_not_found(self):
        """PptxExtractor should raise ExtractionError when LibreOffice is missing."""
        extractor = PptxExtractor()

        with patch("subprocess.run", side_effect=FileNotFoundError("libreoffice not found")):
            with pytest.raises(ExtractionError, match="LibreOffice is not installed"):
                extractor._convert_to_pdf(b"fake pptx", source_uri="test://slides.pptx")


class TestExtractorRegistryWithBuiltins:
    """Tests for ExtractorRegistry with built-in extractors."""

    def test_registry_default_has_text(self):
        """default() should include TxtExtractor for text/plain."""
        registry = ExtractorRegistry.default()
        assert registry.supports("text/plain")

        extractor = registry.get_extractor("text/plain")
        assert extractor.name == "txt"

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


    def test_registry_default_has_xlsx(self):
        """default() should include XlsxExtractor."""
        registry = ExtractorRegistry.default()
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        assert registry.supports(mime)

        extractor = registry.get_extractor(mime)
        assert extractor.name == "xlsx"

    def test_registry_default_has_xlsx_legacy(self):
        """default() should include XlsxExtractor for legacy .xls type."""
        registry = ExtractorRegistry.default()
        assert registry.supports("application/vnd.ms-excel")

    def test_registry_default_has_pptx(self):
        """default() should include PptxExtractor."""
        registry = ExtractorRegistry.default()
        mime = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        assert registry.supports(mime)

        extractor = registry.get_extractor(mime)
        assert extractor.name == "pptx"


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
        assert hasattr(ingest, "XlsxExtractor")
        assert hasattr(ingest, "PptxExtractor")
