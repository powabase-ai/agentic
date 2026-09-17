"""Memory bounds for PDF extraction of very large documents.

A scanned book of 1,500 pages renders to roughly 1.7 MiB of PNG per page at
150 DPI, so holding every page image until extraction returns costs gigabytes
per document. These tests pin the two mechanisms that keep a document's
footprint near one page instead:

* ``raw.metadata["page_image_sink"]`` — a callable the host passes to receive
  each rendered page image as soon as it exists. Images delivered to it are not
  retained in the ``ExtractionResult``.
* Mistral OCR batches sized by bytes as well as pages, so a batch never exceeds
  the upload limit, and a document that cannot be split under it is refused
  before any upload.
"""

import io
import threading
import time
from unittest.mock import MagicMock, patch

import fitz
import numpy as np
import pytest
from PIL import Image

from agentic.ingest import Derivative, ExtractionError, PDFExtractor, RawContent


def _text_pdf(pages: int) -> bytes:
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=300, height=300)
        page.insert_text((20, 40), f"page {i + 1} body text")
    data = doc.tobytes()
    doc.close()
    return data


def _scan_pdf(pages: int, side: int = 400) -> bytes:
    """Pages that are nothing but a noisy image, like a scanned book."""
    rng = np.random.default_rng(0)
    doc = fitz.open()
    for _ in range(pages):
        buf = io.BytesIO()
        pixels = rng.integers(0, 255, (side, side), dtype=np.uint8)
        Image.fromarray(pixels, "L").save(buf, "JPEG", quality=90)
        page = doc.new_page(width=300, height=300)
        page.insert_image(page.rect, stream=buf.getvalue())
    data = doc.tobytes()
    doc.close()
    return data


def _raw(data: bytes, **metadata) -> RawContent:
    return RawContent(
        content=data,
        mime_type="application/pdf",
        source_uri="upload://book.pdf",
        filename="book.pdf",
        metadata=metadata,
    )


class TestPageImageStreaming:
    def test_iter_page_images_is_lazy_and_yields_png_per_page(self):
        extractor = PDFExtractor()
        pages = extractor._iter_page_images(_raw(_text_pdf(3)))

        assert not isinstance(pages, list)
        first = next(pages)
        assert first.type == "image"
        assert first.page == 1
        assert first.content.startswith(b"\x89PNG")
        assert [d.page for d in pages] == [2, 3]

    def test_render_page_images_without_sink_still_returns_every_image(self):
        images = PDFExtractor()._render_page_images(_raw(_text_pdf(3)))

        assert [d.page for d in images] == [1, 2, 3]
        assert all(d.content.startswith(b"\x89PNG") for d in images)

    def test_render_page_images_with_sink_delivers_each_page_and_retains_none(self):
        received = []
        raw = _raw(_text_pdf(3), page_image_sink=received.append)

        images = PDFExtractor()._render_page_images(raw)

        assert images == []
        assert [d.page for d in received] == [1, 2, 3]
        assert all(d.content.startswith(b"\x89PNG") for d in received)

    def test_sink_receives_a_page_before_the_next_page_is_rendered(self):
        """Streaming, not render-all-then-deliver: at no point are two
        undelivered page images alive at once."""
        extractor = PDFExtractor()
        rendered = 0
        seen_at_delivery = []
        real_iter = extractor._iter_page_images

        def counting_iter(raw, dpi=150):
            nonlocal rendered
            for d in real_iter(raw, dpi):
                rendered += 1
                yield d

        def sink(d):
            seen_at_delivery.append(rendered)

        with patch.object(extractor, "_iter_page_images", side_effect=counting_iter):
            extractor._render_page_images(_raw(_text_pdf(4), page_image_sink=sink))

        assert seen_at_delivery == [1, 2, 3, 4]

    def test_sink_failure_is_not_swallowed(self):
        """A render failure degrades to no images, as before. A sink failure is
        the host failing to store a derivative, and must not read as success."""

        def sink(d):
            raise RuntimeError("storage down")

        with pytest.raises(RuntimeError, match="storage down"):
            PDFExtractor()._render_page_images(_raw(_text_pdf(2), page_image_sink=sink))

    @pytest.mark.parametrize("method", ["fitz", "pdfplumber"])
    def test_local_methods_stream_images_to_sink(self, method):
        received = []
        raw = _raw(_text_pdf(3), page_image_sink=received.append)

        result = getattr(PDFExtractor(), f"_extract_{method}")(raw)

        assert [d.type for d in result.derivatives].count("image") == 0
        assert [d.page for d in result.derivatives if d.type == "page_text"] == [1, 2, 3]
        assert [d.page for d in received] == [1, 2, 3]
        assert result.auto_metadata["page_count"] == 3


def _ok_response(text):
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"choices": [{"message": {"content": text}}]}
    response.raise_for_status = MagicMock()
    return response


class TestLightOnStreaming:
    @pytest.mark.asyncio
    async def test_lighton_with_sink_uses_real_pages_and_retains_no_images(self):
        received = []
        raw = _raw(
            _text_pdf(3), extraction_model="lighton", page_image_sink=received.append
        )
        extractor = PDFExtractor(lighton_api_key="k")

        with patch("requests.post", side_effect=lambda *a, **k: _ok_response("ocr")):
            result = await extractor.extract(raw)

        assert result.extraction_method == "lighton_ocr"
        assert [d.type for d in result.derivatives].count("image") == 0
        assert [d.page for d in result.derivatives if d.type == "page_text"] == [1, 2, 3]
        assert sorted(d.page for d in received) == [1, 2, 3]
        assert result.auto_metadata["page_count"] == 3

    @pytest.mark.asyncio
    async def test_lighton_with_sink_holds_at_most_a_window_of_pages(self):
        """Rendered-but-not-yet-delivered pages are the document's image
        footprint. With a sink it is bounded by the request window, however
        long the document is."""
        from agentic.knowledge.model_config import LIGHTON_MAX_CONCURRENCY

        total = LIGHTON_MAX_CONCURRENCY * 4
        extractor = PDFExtractor(lighton_api_key="k")
        lock = threading.Lock()
        outstanding = 0
        peak = 0

        def fake_iter(raw, dpi=150):
            nonlocal outstanding, peak
            for page in range(1, total + 1):
                with lock:
                    outstanding += 1
                    peak = max(peak, outstanding)
                yield Derivative(type="image", content=b"png", format="png", page=page)

        def sink(d):
            nonlocal outstanding
            with lock:
                outstanding -= 1

        def post(*a, **k):
            time.sleep(0.01)
            return _ok_response("ocr")

        raw = _raw(b"%PDF-fake", extraction_model="lighton", page_image_sink=sink)
        with (
            patch.object(extractor, "_iter_page_images", side_effect=fake_iter),
            patch.object(extractor, "_render_page_images") as render_all,
            patch("requests.post", side_effect=post),
        ):
            result = await extractor.extract(raw)

        render_all.assert_not_called()
        assert len([d for d in result.derivatives if d.type == "page_text"]) == total
        assert outstanding == 0
        # One page per request slot, plus at most one page rendered ahead per
        # waiting slot.
        assert peak <= LIGHTON_MAX_CONCURRENCY * 2, peak

    @pytest.mark.asyncio
    async def test_lighton_with_sink_stops_rendering_after_a_page_is_refused(self):
        import requests

        from agentic.knowledge.model_config import LIGHTON_MAX_CONCURRENCY

        total = LIGHTON_MAX_CONCURRENCY * 10
        rendered = 0

        def fake_iter(raw, dpi=150):
            nonlocal rendered
            for page in range(1, total + 1):
                rendered += 1
                yield Derivative(type="image", content=b"png", format="png", page=page)

        def refuse(*a, **k):
            response = requests.Response()
            response.status_code = 400
            response._content = b"bad page"
            raise requests.HTTPError("400", response=response)

        extractor = PDFExtractor(lighton_api_key="k")
        raw = _raw(b"%PDF-fake", extraction_model="lighton", page_image_sink=lambda d: None)
        with (
            patch.object(extractor, "_iter_page_images", side_effect=fake_iter),
            patch.object(extractor, "_extract_opendataloader", side_effect=Exception("x")),
            patch.object(extractor, "_extract_fitz", side_effect=Exception("x")),
            patch.object(extractor, "_extract_pdfplumber", side_effect=Exception("x")),
            patch("requests.post", side_effect=refuse),
        ):
            with pytest.raises(ExtractionError):
                await extractor.extract(raw)

        assert rendered < total
        assert rendered <= LIGHTON_MAX_CONCURRENCY * 2


class TestMistralFileLimit:
    @pytest.mark.asyncio
    async def test_batches_are_sized_to_stay_under_the_upload_limit(self):
        data = _scan_pdf(12)
        limit = len(data) // 3
        extractor = PDFExtractor(mistral_api_key="m", max_pages=1000)
        sent = []

        async def fake_single(pdf_bytes, raw, page_offset=0, render_images=True):
            n = fitz.open(stream=pdf_bytes, filetype="pdf").page_count
            sent.append((len(pdf_bytes), page_offset, n))
            from agentic.ingest import ExtractionResult

            return ExtractionResult(
                source_uri=raw.source_uri,
                mime_type=raw.mime_type,
                derivatives=[
                    Derivative(type="markdown", content="md"),
                    *[
                        Derivative(type="page_text", content="t", page=page_offset + i + 1)
                        for i in range(n)
                    ],
                ],
                stats={"pages_processed": n},
            )

        with (
            patch("agentic.knowledge.model_config.MISTRAL_MAX_FILE_BYTES", limit),
            patch.object(extractor, "_extract_mistral_single", side_effect=fake_single),
            patch.object(extractor, "_render_page_images", return_value=[]),
        ):
            result = await extractor._extract_mistral(_raw(data))

        assert len(sent) > 1
        assert all(size <= limit for size, _, _ in sent), sent
        assert sum(n for _, _, n in sent) == 12
        assert [d.page for d in result.derivatives if d.type == "page_text"] == list(
            range(1, 13)
        )

    @pytest.mark.asyncio
    async def test_page_larger_than_the_limit_is_refused_before_upload(self):
        data = _scan_pdf(2)
        extractor = PDFExtractor(mistral_api_key="m")

        with (
            patch("agentic.knowledge.model_config.MISTRAL_MAX_FILE_BYTES", 1000),
            patch.object(extractor, "_extract_mistral_single") as single,
        ):
            with pytest.raises(ExtractionError, match="upload limit"):
                await extractor._extract_mistral(_raw(data))

        single.assert_not_called()

    @pytest.mark.asyncio
    async def test_refusal_is_not_retried(self):
        """ExtractionError is deterministic to _try_method: one attempt, then
        the chain moves on, instead of three uploads that will all fail."""
        data = _scan_pdf(2)
        extractor = PDFExtractor(mistral_api_key="m")
        calls = 0
        real = extractor._extract_mistral

        async def counting(raw):
            nonlocal calls
            calls += 1
            return await real(raw)

        with (
            patch("agentic.knowledge.model_config.MISTRAL_MAX_FILE_BYTES", 1000),
            patch.object(extractor, "_extract_mistral", side_effect=counting),
        ):
            with pytest.raises(ExtractionError):
                await extractor._try_method("mistral", _raw(data))

        assert calls == 1

    def test_default_limit_is_fifty_megabytes(self):
        from agentic.knowledge import model_config

        assert model_config.MISTRAL_MAX_FILE_BYTES == 50 * 1000 * 1000

