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

import asyncio
import base64
import gc
import io
import pickle
import threading
import time
import weakref
from unittest.mock import MagicMock, patch

import fitz
import numpy as np
import pytest
from PIL import Image

from agentic.ingest import (
    Derivative,
    ExtractionError,
    ExtractionResult,
    PageImageSinkError,
    PDFExtractor,
    RawContent,
)
from agentic.ingest.extractor import pdf as pdf_module


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
        the host failing to store a derivative, and must not read as success.
        It arrives as PageImageSinkError, carrying the host's own exception."""
        storage_down = RuntimeError("storage down")

        def sink(d):
            raise storage_down

        with pytest.raises(PageImageSinkError, match="storage down") as excinfo:
            PDFExtractor()._render_page_images(_raw(_text_pdf(2), page_image_sink=sink))

        assert excinfo.value.__cause__ is storage_down
        assert excinfo.value.cause is storage_down
        assert excinfo.value.page == 1

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

        guarded, _ = _guarded_page_range(limit_calls=20)

        with (
            patch("agentic.knowledge.model_config.MISTRAL_MAX_FILE_BYTES", 1000),
            patch.object(pdf_module, "_pdf_page_range", side_effect=guarded),
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



def _page_of(payload) -> int:
    """The page number a fake page image carries, read back out of a request."""
    url = payload["messages"][0]["content"][0]["image_url"]["url"]
    return int(base64.b64decode(url.split(",", 1)[1]).decode().rsplit("-", 1)[1])


def _fake_pages(total):
    """Stand-in for _iter_page_images: page N's content is ``page-N``."""

    def fake_iter(raw, dpi=150):
        for page in range(1, total + 1):
            yield Derivative(
                type="image", content=f"page-{page}".encode(), format="png", page=page
            )

    return fake_iter


class TestSinkFailureAbortsExtraction:
    """A page image the host could not store ends the whole extraction.

    Retrying the method, or moving on to the next one, re-sends every page to
    a paid OCR endpoint and then fails on the same storage again. The caller
    gets PageImageSinkError and decides whether to retry.
    """

    @pytest.mark.asyncio
    async def test_lighton_sends_no_request_after_the_sink_fails(self):
        """Serial, so the count is exact: pages 1-3 were sent, page 3 could not
        be stored, and nothing else is sent by LightOn or any other method."""
        total = 10
        posts = []

        def post(url, json=None, **kwargs):
            posts.append(_page_of(json))
            return _ok_response("ocr")

        def sink(d):
            if d.page == 3:
                raise RuntimeError("storage down")

        extractor = PDFExtractor(lighton_api_key="k", mistral_api_key="m")
        raw = _raw(b"%PDF-fake", extraction_model="auto", page_image_sink=sink)
        with (
            patch("agentic.knowledge.model_config.LIGHTON_MAX_CONCURRENCY", 1),
            patch.object(extractor, "_iter_page_images", side_effect=_fake_pages(total)),
            patch.object(extractor, "_extract_mistral") as mistral,
            patch.object(extractor, "_extract_opendataloader") as opendataloader,
            patch.object(extractor, "_extract_fitz") as fitz_method,
            patch.object(extractor, "_extract_pdfplumber") as pdfplumber,
            patch("requests.post", side_effect=post),
        ):
            with pytest.raises(PageImageSinkError) as excinfo:
                await extractor.extract(raw)

        assert posts == [1, 2, 3]
        assert excinfo.value.page == 3
        assert isinstance(excinfo.value.__cause__, RuntimeError)
        mistral.assert_not_called()
        opendataloader.assert_not_called()
        fitz_method.assert_not_called()
        pdfplumber.assert_not_called()

    @pytest.mark.asyncio
    async def test_lighton_sink_failure_costs_at_most_one_request_per_page(self):
        """Concurrent: pages already on the wire finish, but no page is sent
        twice and the requested method's local fallbacks never run."""
        total = 4
        posts = []

        def post(url, json=None, **kwargs):
            posts.append(_page_of(json))
            return _ok_response("ocr")

        def sink(d):
            raise RuntimeError("storage down")

        extractor = PDFExtractor(lighton_api_key="k")
        raw = _raw(b"%PDF-fake", extraction_model="lighton", page_image_sink=sink)
        with (
            patch.object(extractor, "_iter_page_images", side_effect=_fake_pages(total)),
            patch.object(extractor, "_extract_opendataloader") as opendataloader,
            patch.object(extractor, "_extract_fitz") as fitz_method,
            patch.object(extractor, "_extract_pdfplumber") as pdfplumber,
            patch("requests.post", side_effect=post),
        ):
            with pytest.raises(PageImageSinkError):
                await extractor.extract(raw)

        assert 1 <= len(posts) <= total
        assert len(posts) == len(set(posts)), posts
        opendataloader.assert_not_called()
        fitz_method.assert_not_called()
        pdfplumber.assert_not_called()

    @pytest.mark.asyncio
    async def test_sink_failure_is_reported_over_a_page_refusal(self):
        """Another page's deterministic refusal would send the chain on to the
        next method, which stores pages to the same broken sink."""
        import requests

        def post(url, json=None, **kwargs):
            if _page_of(json) == 2:
                response = requests.Response()
                response.status_code = 400
                response._content = b"bad page"
                raise requests.HTTPError("400", response=response)
            return _ok_response("ocr")

        def sink(d):
            raise RuntimeError("storage down")

        extractor = PDFExtractor(lighton_api_key="k")
        raw = _raw(b"%PDF-fake", page_image_sink=sink)
        with (
            patch("agentic.knowledge.model_config.LIGHTON_MAX_CONCURRENCY", 2),
            patch.object(extractor, "_iter_page_images", side_effect=_fake_pages(2)),
            patch("requests.post", side_effect=post),
        ):
            with pytest.raises(PageImageSinkError):
                await extractor._extract_lighton(raw)

    @pytest.mark.asyncio
    async def test_local_method_sink_failure_does_not_fall_through(self):
        extractor = PDFExtractor()

        def sink(d):
            raise RuntimeError("storage down")

        raw = _raw(_text_pdf(3), extraction_model="auto", page_image_sink=sink)
        with (
            patch(
                "agentic.knowledge.model_config.EXTRACTION_FALLBACK_CHAIN",
                ["fitz", "pdfplumber"],
            ),
            patch.object(extractor, "_extract_pdfplumber") as pdfplumber,
        ):
            with pytest.raises(PageImageSinkError):
                await extractor.extract(raw)

        pdfplumber.assert_not_called()

    @pytest.mark.asyncio
    async def test_sink_failure_in_a_local_fallback_ends_the_fallbacks(self):
        """The requested OCR method failed for its own reason; the first local
        fallback then cannot store a page. The later fallbacks would store to
        the same sink, so none of them runs."""
        extractor = PDFExtractor(lighton_api_key="k")
        raw = _raw(b"%PDF-fake", extraction_model="lighton")
        sink_failure = PageImageSinkError(1, RuntimeError("storage down"))
        with (
            patch.object(
                extractor, "_extract_lighton", side_effect=ExtractionError("refused")
            ),
            patch.object(
                extractor, "_extract_opendataloader", side_effect=sink_failure
            ),
            patch.object(extractor, "_extract_fitz") as fitz_method,
            patch.object(extractor, "_extract_pdfplumber") as pdfplumber,
        ):
            with pytest.raises(PageImageSinkError) as excinfo:
                await extractor.extract(raw)

        assert excinfo.value is sink_failure
        fitz_method.assert_not_called()
        pdfplumber.assert_not_called()

    @pytest.mark.asyncio
    async def test_mistral_sink_failure_uploads_once_and_skips_local_fallbacks(self):
        page = MagicMock(markdown="text", images=[], index=0, dimensions=None)
        client = MagicMock()
        client.ocr.process.return_value = MagicMock(pages=[page])

        def sink(d):
            raise RuntimeError("storage down")

        extractor = PDFExtractor(mistral_api_key="m")
        raw = _raw(_text_pdf(1), extraction_model="mistral", page_image_sink=sink)
        with (
            # create=True: whether mistralai.client exports Mistral depends on
            # the installed mistralai major version.
            patch("mistralai.client.Mistral", return_value=client, create=True),
            patch.object(extractor, "_extract_opendataloader") as opendataloader,
            patch.object(extractor, "_extract_fitz") as fitz_method,
            patch.object(extractor, "_extract_pdfplumber") as pdfplumber,
        ):
            with pytest.raises(PageImageSinkError):
                await extractor.extract(raw)

        assert client.files.upload.call_count == 1
        assert client.ocr.process.call_count == 1
        opendataloader.assert_not_called()
        fitz_method.assert_not_called()
        pdfplumber.assert_not_called()

    @pytest.mark.parametrize("method", ["lighton", "mistral", "paddleocr"])
    @pytest.mark.asyncio
    async def test_method_retry_loops_do_not_retry_a_sink_failure(self, method):
        extractor = PDFExtractor(
            lighton_api_key="k", mistral_api_key="m", paddleocr_api_key="p"
        )
        failure = PageImageSinkError(1, RuntimeError("storage down"))
        # No sleep stub: a retry backs off for at least a second before its
        # second attempt, so a retried sink failure shows up as a timeout here
        # as well as in the call count.
        with patch.object(
            extractor, f"_extract_{method}", side_effect=failure
        ) as run:
            with pytest.raises(PageImageSinkError):
                await asyncio.wait_for(
                    extractor._try_method(method, _raw(b"%PDF-fake")), timeout=0.5
                )

        assert run.call_count == 1


    def test_sink_error_pickles_a_cause_that_cannot_be_pickled_as_text(self):
        """A cause with a multi-argument constructor fails to unpickle, and one
        holding a lock fails to pickle. Neither may make the sink error itself
        unreportable."""

        class NeedsTwoArgs(Exception):
            def __init__(self, bucket, key):
                super().__init__(f"{bucket}/{key} refused")

        class HoldsALock(Exception):
            def __init__(self):
                super().__init__("storage down")
                self.lock = threading.Lock()

        for cause in (NeedsTwoArgs("bucket", "key"), HoldsALock()):
            error = PageImageSinkError(3, cause)

            restored = pickle.loads(pickle.dumps(error))

            assert type(restored) is PageImageSinkError
            assert restored.page == 3
            assert str(restored) == str(error)
            assert type(cause).__name__ in str(restored.cause)
            assert str(cause) in str(restored.cause)
            assert restored.__cause__ is restored.cause

    def test_sink_error_message_survives_a_cause_that_cannot_be_printed(self):
        class Unprintable(Exception):
            def __str__(self):
                raise ValueError("no")

        cause = Unprintable()

        error = PageImageSinkError(2, cause)

        assert error.cause is cause
        assert error.__cause__ is cause
        assert "page 2" in str(error)
        assert "Unprintable" in str(error)

    @pytest.mark.parametrize("sink", ["not callable", 42])
    @pytest.mark.parametrize("model", ["auto", "lighton", "fitz"])
    @pytest.mark.asyncio
    async def test_a_sink_that_is_not_callable_is_refused(self, sink, model):
        """Silently ignoring it would turn the memory bound off with no sign.
        It is the caller's error, so it is raised at once: not retried by a
        method's retry loop, and not absorbed by the fallback chain."""
        extractor = PDFExtractor(lighton_api_key="k", mistral_api_key="m")
        raw = _raw(_text_pdf(2), extraction_model=model, page_image_sink=sink)

        with (
            patch.object(extractor, "_extract_lighton") as lighton,
            patch.object(extractor, "_extract_mistral") as mistral,
            patch.object(extractor, "_extract_opendataloader") as opendataloader,
        ):
            with pytest.raises(TypeError, match="page_image_sink"):
                await asyncio.wait_for(extractor.extract(raw), timeout=0.5)

        lighton.assert_not_called()
        mistral.assert_not_called()
        opendataloader.assert_not_called()
        with pytest.raises(TypeError, match="page_image_sink"):
            extractor._render_page_images(raw)

    def test_a_sink_of_none_means_no_sink(self):
        images = PDFExtractor()._render_page_images(
            _raw(_text_pdf(2), page_image_sink=None)
        )

        assert [d.page for d in images] == [1, 2]

    def test_renderer_is_closed_when_the_sink_fails(self):
        """Not left to the garbage collector: the traceback keeps the frame,
        and with it an open PyMuPDF document, alive for as long as the caller
        holds the error."""
        extractor = PDFExtractor()
        closed = []

        def fake_iter(raw, dpi=150):
            try:
                yield from _fake_pages(5)(raw, dpi)
            finally:
                closed.append(True)

        def sink(d):
            raise RuntimeError("storage down")

        with patch.object(extractor, "_iter_page_images", side_effect=fake_iter):
            with pytest.raises(PageImageSinkError) as excinfo:
                extractor._render_page_images(_raw(b"%PDF-fake", page_image_sink=sink))

        assert excinfo.value.page == 1
        assert closed == [True]

    @pytest.mark.asyncio
    async def test_a_base_exception_from_the_sink_stops_the_other_workers(self):
        """Not an Exception, so not a PageImageSinkError, but the other request
        slots must still stop taking pages rather than run on against a sink
        that is gone."""

        class Interrupted(BaseException):
            pass

        concurrency, total = 2, 20
        sink_calls = []
        posts = []

        def post(url, json=None, **kwargs):
            posts.append(_page_of(json))
            time.sleep(0.05)
            return _ok_response("ocr")

        def sink(d):
            sink_calls.append(d.page)
            raise Interrupted()

        extractor = PDFExtractor(lighton_api_key="k")
        raw = _raw(b"%PDF-fake", page_image_sink=sink)
        with (
            patch("agentic.knowledge.model_config.LIGHTON_MAX_CONCURRENCY", concurrency),
            patch.object(extractor, "_iter_page_images", side_effect=_fake_pages(total)),
            patch("requests.post", side_effect=post),
        ):
            with pytest.raises(Interrupted):
                await extractor._extract_lighton(raw)
            posts_at_raise = len(posts)
            # Give any worker still running the chance to take another page.
            await asyncio.sleep(0.5)

        assert len(sink_calls) == 1
        assert len(posts) == posts_at_raise
        assert posts_at_raise <= concurrency

    @pytest.mark.asyncio
    async def test_a_base_exception_from_the_sink_leaves_no_worker_running(self):
        """The slot that was mid-render when the sink raised must not carry on
        into another request after the extraction has already unwound."""

        class Interrupted(BaseException):
            pass

        def fake_iter(raw, dpi=150):
            for page in range(1, 11):
                if page == 3:
                    # Still rendering when page 1's sink call raises.
                    time.sleep(0.3)
                yield Derivative(
                    type="image", content=f"page-{page}".encode(), format="png", page=page
                )

        def sink(d):
            if d.page == 1:
                time.sleep(0.05)
                raise Interrupted()

        extractor = PDFExtractor(lighton_api_key="k")
        raw = _raw(b"%PDF-fake", page_image_sink=sink)
        with (
            patch("agentic.knowledge.model_config.LIGHTON_MAX_CONCURRENCY", 2),
            patch.object(extractor, "_iter_page_images", side_effect=fake_iter),
            patch("requests.post", side_effect=lambda *a, **k: _ok_response("ocr")),
        ):
            with pytest.raises(Interrupted):
                await extractor._extract_lighton(raw)
            await asyncio.sleep(0.2)
            lingering = [
                t
                for t in asyncio.all_tasks()
                if t is not asyncio.current_task() and not t.done()
            ]

        assert lingering == []

    @pytest.mark.asyncio
    async def test_a_page_refusal_keeps_pages_already_in_flight_from_the_sink(self):
        """The abort need not come from the sink: once any page has failed the
        document, pages whose OCR was already on the wire are not stored."""
        import requests

        concurrency = 4
        in_flight = threading.Barrier(concurrency, timeout=10)
        stored = []

        def post(url, json=None, **kwargs):
            page = _page_of(json)
            in_flight.wait()
            if page == 1:
                response = requests.Response()
                response.status_code = 400
                response._content = b"bad page"
                raise requests.HTTPError("400", response=response)
            # Let the event loop record page 1's refusal first.
            time.sleep(0.3)
            return _ok_response("ocr")

        extractor = PDFExtractor(lighton_api_key="k")
        raw = _raw(b"%PDF-fake", page_image_sink=lambda d: stored.append(d.page))
        with (
            patch("agentic.knowledge.model_config.LIGHTON_MAX_CONCURRENCY", concurrency),
            patch.object(
                extractor, "_iter_page_images", side_effect=_fake_pages(concurrency)
            ),
            patch("requests.post", side_effect=post),
        ):
            with pytest.raises(ExtractionError, match="rejected page 1"):
                await extractor._extract_lighton(raw)

        assert stored == []

    def test_sink_error_survives_pickling(self):
        """Task queues pickle exceptions to report them across processes."""
        cause = RuntimeError("storage down")
        error = PageImageSinkError(7, cause)

        restored = pickle.loads(pickle.dumps(error))

        assert type(restored) is PageImageSinkError
        assert restored.page == 7
        assert isinstance(restored.cause, RuntimeError)
        assert str(restored.cause) == "storage down"
        assert restored.__cause__ is restored.cause
        assert str(restored) == str(error)

    @pytest.mark.asyncio
    async def test_sink_is_not_called_again_once_a_sink_call_has_failed(self):
        """Pages already on the wire when the sink fails still finish OCR, but
        are not handed to a sink that is known to be failing. The sink here
        does not short-circuit itself, as a host's sink need not."""
        concurrency = 4
        in_flight = threading.Barrier(concurrency, timeout=10)
        sink_failed = threading.Event()
        sink_calls = []

        def post(url, json=None, **kwargs):
            page = _page_of(json)
            in_flight.wait()
            if page != 1:
                sink_failed.wait(timeout=10)
                # Let the event loop record the failure before this page's
                # OCR completes.
                time.sleep(0.2)
            return _ok_response(f"text of page {page}")

        def sink(d):
            sink_calls.append(d.page)
            sink_failed.set()
            raise RuntimeError("storage down")

        extractor = PDFExtractor(lighton_api_key="k")
        raw = _raw(b"%PDF-fake", page_image_sink=sink)
        with (
            patch("agentic.knowledge.model_config.LIGHTON_MAX_CONCURRENCY", concurrency),
            patch.object(
                extractor, "_iter_page_images", side_effect=_fake_pages(concurrency)
            ),
            patch("requests.post", side_effect=post),
        ):
            with pytest.raises(PageImageSinkError):
                await extractor._extract_lighton(raw)

        assert sink_calls == [1]


class TestLightOnStreamingOrder:
    @pytest.mark.asyncio
    async def test_page_text_follows_the_document_when_early_pages_finish_last(self):
        total = 4
        stored = []

        def post(url, json=None, **kwargs):
            page = _page_of(json)
            time.sleep(0.05 * (total - page + 1))
            return _ok_response(f"text of page {page}")

        extractor = PDFExtractor(lighton_api_key="k")
        raw = _raw(b"%PDF-fake", page_image_sink=lambda d: stored.append(d.page))
        with (
            patch("agentic.knowledge.model_config.LIGHTON_MAX_CONCURRENCY", total),
            patch.object(extractor, "_iter_page_images", side_effect=_fake_pages(total)),
            patch("requests.post", side_effect=post),
        ):
            result = await extractor._extract_lighton(raw)

        # The premise: completion order really was not document order.
        assert stored != sorted(stored), stored
        page_texts = [d for d in result.derivatives if d.type == "page_text"]
        assert [d.page for d in page_texts] == [1, 2, 3, 4]
        assert [d.content for d in page_texts] == [
            f"text of page {n}" for n in range(1, total + 1)
        ]
        markdown = next(d for d in result.derivatives if d.type == "markdown")
        assert markdown.content == "\n\n".join(d.content for d in page_texts)

    @pytest.mark.asyncio
    async def test_renderer_is_closed_before_the_next_method_opens_the_pdf(self):
        """PyMuPDF is not thread-safe. The render thread must have finished
        closing the document before a fallback method touches PyMuPDF."""
        import requests

        closed = threading.Event()
        closed_when_fallback_ran = []

        def fake_iter(raw, dpi=150):
            try:
                yield from _fake_pages(40)(raw, dpi)
            finally:
                time.sleep(0.3)
                closed.set()

        def refuse(*a, **k):
            response = requests.Response()
            response.status_code = 400
            response._content = b"bad page"
            raise requests.HTTPError("400", response=response)

        def fallback(raw):
            closed_when_fallback_ran.append(closed.is_set())
            raise ExtractionError("not this one")

        extractor = PDFExtractor(lighton_api_key="k")
        raw = _raw(b"%PDF-fake", extraction_model="lighton", page_image_sink=lambda d: None)
        with (
            patch.object(extractor, "_iter_page_images", side_effect=fake_iter),
            patch.object(extractor, "_extract_opendataloader", side_effect=fallback),
            patch.object(extractor, "_extract_fitz", side_effect=fallback),
            patch.object(extractor, "_extract_pdfplumber", side_effect=fallback),
            patch("requests.post", side_effect=refuse),
        ):
            with pytest.raises(ExtractionError):
                await extractor.extract(raw)

        assert closed_when_fallback_ran and all(closed_when_fallback_ran)


def _uneven_pdf(text_pages: int, scan_pages: int) -> bytes:
    """Cheap text pages followed by expensive scanned pages."""
    doc = fitz.open()
    doc.insert_pdf(fitz.open(stream=_text_pdf(text_pages), filetype="pdf"))
    doc.insert_pdf(fitz.open(stream=_scan_pdf(scan_pages), filetype="pdf"))
    data = doc.tobytes()
    doc.close()
    return data


def _guarded_page_range(limit_calls: int):
    """_pdf_page_range that fails the test instead of looping forever."""
    real = pdf_module._pdf_page_range
    calls = [0]

    def guarded(src, start, end):
        calls[0] += 1
        if calls[0] > limit_calls:
            raise AssertionError(f"batch planning did not converge ({calls[0]} builds)")
        return real(src, start, end)

    return guarded, calls


class TestMistralBatchPlanning:
    def test_uneven_pages_are_halved_until_every_batch_fits(self):
        text_pages, scan_pages = 6, 6
        data = _uneven_pdf(text_pages, scan_pages)
        total = text_pages + scan_pages
        src = fitz.open(stream=data, filetype="pdf")
        largest_page = max(
            len(pdf_module._pdf_page_range(src, i, i)) for i in range(total)
        )
        src.close()
        # Room for one scanned page, not two: batches sized from the average
        # page take two or three scanned pages and must be split.
        limit = largest_page * 3 // 2
        guarded, calls = _guarded_page_range(limit_calls=10 * total)

        with patch.object(pdf_module, "_pdf_page_range", side_effect=guarded):
            ranges = pdf_module._plan_pdf_batches(data, max_pages=1000, max_bytes=limit)

        # Halving happened: more ranges were built than were kept, and the
        # kept ranges are not all the size of the first guess.
        assert calls[0] > len(ranges)
        assert len({end - start for start, end in ranges}) > 1
        # Contiguous, in document order, covering every page exactly once.
        assert ranges[0][0] == 0
        assert ranges[-1][1] == total - 1
        assert all(b[0] == a[1] + 1 for a, b in zip(ranges, ranges[1:], strict=False))
        # And every one of them fits.
        src = fitz.open(stream=data, filetype="pdf")
        try:
            sizes = [len(pdf_module._pdf_page_range(src, s, e)) for s, e in ranges]
        finally:
            src.close()
        assert all(size <= limit for size in sizes), (sizes, limit)

    def test_two_page_range_over_the_limit_splits_into_single_pages(self):
        data = _scan_pdf(2)
        src = fitz.open(stream=data, filetype="pdf")
        largest_page = max(len(pdf_module._pdf_page_range(src, i, i)) for i in range(2))
        src.close()
        guarded, _ = _guarded_page_range(limit_calls=20)

        with patch.object(pdf_module, "_pdf_page_range", side_effect=guarded):
            ranges = pdf_module._plan_pdf_batches(
                data, max_pages=2, max_bytes=largest_page + 1
            )

        assert ranges == [(0, 0), (1, 1)]

    @pytest.mark.asyncio
    async def test_extract_mistral_numbers_pages_across_halved_batches(self):
        data = _uneven_pdf(6, 6)
        src = fitz.open(stream=data, filetype="pdf")
        largest_page = max(len(pdf_module._pdf_page_range(src, i, i)) for i in range(12))
        src.close()
        limit = largest_page * 3 // 2
        extractor = PDFExtractor(mistral_api_key="m", max_pages=1000)
        sent = []

        async def fake_single(pdf_bytes, raw, page_offset=0, render_images=True):
            n = fitz.open(stream=pdf_bytes, filetype="pdf").page_count
            sent.append((len(pdf_bytes), page_offset, n))
            return ExtractionResult(
                source_uri=raw.source_uri,
                mime_type=raw.mime_type,
                derivatives=[
                    Derivative(type="markdown", content=f"batch at {page_offset}"),
                    *[
                        Derivative(type="page_text", content="t", page=page_offset + i + 1)
                        for i in range(n)
                    ],
                ],
                stats={"pages_processed": n},
            )

        guarded, _ = _guarded_page_range(limit_calls=200)
        with (
            patch("agentic.knowledge.model_config.MISTRAL_MAX_FILE_BYTES", limit),
            patch.object(pdf_module, "_pdf_page_range", side_effect=guarded),
            patch.object(extractor, "_extract_mistral_single", side_effect=fake_single),
            patch.object(extractor, "_render_page_images", return_value=[]),
        ):
            result = await extractor._extract_mistral(_raw(data))

        assert all(size <= limit for size, _, _ in sent), sent
        assert [offset for _, offset, _ in sent] == sorted(offset for _, offset, _ in sent)
        assert [d.page for d in result.derivatives if d.type == "page_text"] == list(
            range(1, 13)
        )

    @pytest.mark.asyncio
    async def test_batches_are_built_one_at_a_time(self):
        """Building every batch before sending the first holds all of them —
        up to eight 50 MB PDFs for a long scan."""
        data = _scan_pdf(12)
        limit = len(data) // 3
        extractor = PDFExtractor(mistral_api_key="m", max_pages=1000)
        real_range = pdf_module._pdf_page_range
        real_plan = pdf_module._plan_pdf_batches
        builds = [0]
        planned_builds = []
        built_before_each_send = []

        def counting_range(src, start, end):
            builds[0] += 1
            return real_range(src, start, end)

        def recording_plan(*args, **kwargs):
            ranges = real_plan(*args, **kwargs)
            planned_builds.append(builds[0])
            return ranges

        async def fake_single(pdf_bytes, raw, page_offset=0, render_images=True):
            built_before_each_send.append(builds[0] - planned_builds[0])
            n = fitz.open(stream=pdf_bytes, filetype="pdf").page_count
            return ExtractionResult(
                source_uri=raw.source_uri,
                mime_type=raw.mime_type,
                derivatives=[
                    Derivative(type="page_text", content="t", page=page_offset + i + 1)
                    for i in range(n)
                ],
                stats={"pages_processed": n},
            )

        with (
            patch("agentic.knowledge.model_config.MISTRAL_MAX_FILE_BYTES", limit),
            patch.object(pdf_module, "_pdf_page_range", side_effect=counting_range),
            patch.object(pdf_module, "_plan_pdf_batches", side_effect=recording_plan),
            patch.object(extractor, "_extract_mistral_single", side_effect=fake_single),
            patch.object(extractor, "_render_page_images", return_value=[]),
        ):
            await extractor._extract_mistral(_raw(data))

        assert len(built_before_each_send) > 1
        assert built_before_each_send == list(range(1, len(built_before_each_send) + 1))


def _failing_after(pages: int):
    def fake_iter(raw, dpi=150):
        for page in range(1, pages + 1):
            yield Derivative(type="image", content=b"png", format="png", page=page)
        raise RuntimeError(f"cannot render page {pages + 1}")

    return fake_iter


class TestPartialPageImages:
    """Pages a sink has already stored cannot be taken back. If rendering then
    fails, the result says the image set is incomplete instead of reading as a
    clean extraction."""

    @pytest.mark.parametrize("method", ["fitz", "pdfplumber"])
    def test_render_failure_after_pages_reached_the_sink_is_reported(self, method):
        extractor = PDFExtractor()
        received = []
        raw = _raw(_text_pdf(3), page_image_sink=received.append)

        with patch.object(extractor, "_iter_page_images", side_effect=_failing_after(2)):
            result = getattr(extractor, f"_extract_{method}")(raw)

        assert [d.page for d in received] == [1, 2]
        assert result.auto_metadata["page_images_incomplete"] is True
        assert [d.page for d in result.derivatives if d.type == "page_text"] == [1, 2, 3]

    def test_combined_mistral_batches_report_a_partial_image_set(self):
        extractor = PDFExtractor()
        received = []
        raw = _raw(_text_pdf(3), page_image_sink=received.append)

        with patch.object(extractor, "_iter_page_images", side_effect=_failing_after(1)):
            result = extractor._combine_batch_results([], raw, 3)

        assert [d.page for d in received] == [1]
        assert result.auto_metadata["page_images_incomplete"] is True

    @pytest.mark.parametrize("method", ["fitz", "pdfplumber"])
    def test_complete_image_set_carries_no_flag(self, method):
        raw = _raw(_text_pdf(3), page_image_sink=lambda d: None)

        result = getattr(PDFExtractor(), f"_extract_{method}")(raw)

        assert "page_images_incomplete" not in result.auto_metadata

    def test_render_failure_before_any_page_reached_the_sink_carries_no_flag(self):
        """Nothing was stored, so there is no partial set to report: the same
        "no images" outcome as without a sink."""
        extractor = PDFExtractor()
        received = []
        raw = _raw(_text_pdf(3), page_image_sink=received.append)

        with patch.object(extractor, "_iter_page_images", side_effect=_failing_after(0)):
            result = extractor._extract_fitz(raw)

        assert received == []
        assert "page_images_incomplete" not in result.auto_metadata

    def test_render_failure_without_a_sink_degrades_to_no_images_as_before(self):
        extractor = PDFExtractor()

        with patch.object(extractor, "_iter_page_images", side_effect=_failing_after(2)):
            result = extractor._extract_fitz(_raw(_text_pdf(3)))

        assert [d for d in result.derivatives if d.type == "image"] == []
        assert "page_images_incomplete" not in result.auto_metadata

    @pytest.mark.asyncio
    async def test_single_mistral_call_reports_a_partial_image_set(self):
        page = MagicMock(markdown="text", images=[], index=0, dimensions=None)
        client = MagicMock()
        client.ocr.process.return_value = MagicMock(pages=[page])
        extractor = PDFExtractor(mistral_api_key="m")
        raw = _raw(_text_pdf(3), page_image_sink=lambda d: None)

        with (
            patch("mistralai.client.Mistral", return_value=client, create=True),
            patch.object(extractor, "_iter_page_images", side_effect=_failing_after(1)),
        ):
            result = await extractor._extract_mistral_single(raw.content, raw)

        assert result.auto_metadata["page_images_incomplete"] is True

    @pytest.mark.asyncio
    async def test_paddleocr_reports_a_partial_image_set(self):
        response = MagicMock()
        response.json.return_value = {
            "result": {"layoutParsingResults": [{"markdown": {"text": "page"}}]}
        }
        extractor = PDFExtractor(paddleocr_api_key="p")
        raw = _raw(_text_pdf(3), page_image_sink=lambda d: None)

        with (
            patch("requests.post", return_value=response),
            patch.object(extractor, "_iter_page_images", side_effect=_failing_after(1)),
        ):
            result = await extractor._extract_paddleocr(raw)

        assert result.auto_metadata["page_images_incomplete"] is True

    @pytest.mark.asyncio
    async def test_llamaparse_reports_a_partial_image_set(self):
        extractor = PDFExtractor(llamaparse_api_key="l")
        raw = _raw(_text_pdf(3), page_image_sink=lambda d: None)

        with (
            patch(
                "agentic.ingest.extractor.llamaparse.parse_pages",
                return_value=[{"markdown": "page", "page_number": 1}],
            ),
            patch.object(extractor, "_iter_page_images", side_effect=_failing_after(1)),
        ):
            result = await extractor._extract_llamaparse(raw)

        assert result.auto_metadata["page_images_incomplete"] is True

    def test_opendataloader_reports_a_partial_image_set(self):
        import os
        import sys

        def convert(input_path, output_dir, format, quiet):
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "page.md"), "w") as f:
                f.write("page")

        extractor = PDFExtractor()
        raw = _raw(_text_pdf(2), page_image_sink=lambda d: None)
        with (
            patch.dict(sys.modules, {"opendataloader_pdf": MagicMock(convert=convert)}),
            patch.object(extractor, "_iter_page_images", side_effect=_failing_after(1)),
        ):
            result = extractor._extract_opendataloader(raw)

        assert result.auto_metadata["page_images_incomplete"] is True


class TestDeliveredImagesAreReleased:
    """With a sink, a page image the sink has taken must not stay reachable
    from the extractor. A regression here holds every page of a long scan
    again while every other test stays green."""

    def test_local_render_holds_no_page_the_sink_has_already_taken(self):
        extractor = PDFExtractor()
        delivered = []
        alive_at_each_delivery = []

        def sink(d):
            gc.collect()
            alive_at_each_delivery.append(sum(ref() is not None for ref in delivered))
            delivered.append(weakref.ref(d))

        raw = _raw(_text_pdf(6), page_image_sink=sink)
        extractor._extract_fitz(raw)
        gc.collect()

        assert alive_at_each_delivery == [0] * 6
        assert sum(ref() is not None for ref in delivered) == 0

    @pytest.mark.asyncio
    async def test_streamed_lighton_holds_at_most_a_window_of_taken_pages(self):
        from agentic.knowledge.model_config import LIGHTON_MAX_CONCURRENCY

        # More pages than two windows, so retaining taken pages cannot hide
        # inside the bound.
        total = LIGHTON_MAX_CONCURRENCY * 2 + 3
        lock = threading.Lock()
        delivered = []
        peak_alive = 0

        def sink(d):
            nonlocal peak_alive
            with lock:
                gc.collect()
                alive = sum(ref() is not None for ref in delivered)
                peak_alive = max(peak_alive, alive)
                delivered.append(weakref.ref(d))

        def post(*a, **k):
            time.sleep(0.005)
            return _ok_response("ocr")

        extractor = PDFExtractor(lighton_api_key="k")
        raw = _raw(b"%PDF-fake", page_image_sink=sink)
        with (
            patch.object(extractor, "_iter_page_images", side_effect=_fake_pages(total)),
            patch("requests.post", side_effect=post),
        ):
            await extractor._extract_lighton(raw)
        gc.collect()

        assert len(delivered) == total
        assert peak_alive <= LIGHTON_MAX_CONCURRENCY, peak_alive
        assert sum(ref() is not None for ref in delivered) == 0

    def test_pages_are_rendered_one_at_a_time_and_pixmaps_are_freed(self):
        """At page N's delivery exactly N pixmaps have been made — rendering is
        lazy — and none of them is still alive: the raw pixmap, several times
        the PNG, is not kept."""
        extractor = PDFExtractor()
        real_get_pixmap = fitz.Page.get_pixmap
        pixmaps = []
        seen = []

        def counting_get_pixmap(page, *args, **kwargs):
            pix = real_get_pixmap(page, *args, **kwargs)
            pixmaps.append(weakref.ref(pix))
            return pix

        def sink(d):
            gc.collect()
            seen.append((d.page, len(pixmaps), sum(r() is not None for r in pixmaps)))

        with patch.object(fitz.Page, "get_pixmap", counting_get_pixmap):
            extractor._render_page_images(_raw(_text_pdf(5), page_image_sink=sink))

        assert seen == [(n, n, 0) for n in range(1, 6)]


class TestSinkThreadContract:
    """The sink is called from one thread at a time within a pass: the
    caller's thread for the local methods, and a single dedicated thread for
    streamed LightOn, never the event loop's."""

    @pytest.mark.asyncio
    async def test_streamed_lighton_calls_the_sink_on_one_thread_off_the_loop(self):
        concurrency, total = 4, 16
        lock = threading.Lock()
        threads = set()
        active = 0
        peak_active = 0
        loop_thread = threading.get_ident()

        def sink(d):
            nonlocal active, peak_active
            with lock:
                threads.add((threading.get_ident(), threading.current_thread().name))
                active += 1
                peak_active = max(peak_active, active)
            time.sleep(0.01)
            with lock:
                active -= 1

        extractor = PDFExtractor(lighton_api_key="k")
        raw = _raw(b"%PDF-fake", page_image_sink=sink)
        with (
            patch("agentic.knowledge.model_config.LIGHTON_MAX_CONCURRENCY", concurrency),
            patch.object(extractor, "_iter_page_images", side_effect=_fake_pages(total)),
            patch("requests.post", side_effect=lambda *a, **k: _ok_response("ocr")),
        ):
            await extractor._extract_lighton(raw)

        assert peak_active == 1
        assert len(threads) == 1
        (ident, name), = threads
        assert ident != loop_thread
        assert name.startswith("lighton-sink")

    def test_local_methods_call_the_sink_on_the_callers_thread(self):
        threads = set()
        raw = _raw(
            _text_pdf(3),
            page_image_sink=lambda d: threads.add(threading.get_ident()),
        )

        PDFExtractor()._extract_fitz(raw)

        assert threads == {threading.get_ident()}


class TestMistralPlanningBounds:
    def test_page_cap_still_applies_when_the_byte_cap_sizes_batches(self):
        data = _text_pdf(12)

        ranges = pdf_module._plan_pdf_batches(data, max_pages=4, max_bytes=len(data) - 1)

        assert all(end - start + 1 <= 4 for start, end in ranges), ranges
        assert ranges[0][0] == 0 and ranges[-1][1] == 11

    def test_planning_starts_from_the_average_page_not_the_whole_document(self):
        """Without a first guess from the average page size, planning builds
        the whole document once before halving: a second copy of a 365 MB
        source."""
        data = _scan_pdf(12)
        limit = len(data) // 3
        real_range = pdf_module._pdf_page_range
        built = []

        def recording(src, start, end):
            out = real_range(src, start, end)
            built.append(len(out))
            return out

        with patch.object(pdf_module, "_pdf_page_range", side_effect=recording):
            pdf_module._plan_pdf_batches(data, max_pages=1000, max_bytes=limit)

        assert built
        assert max(built) <= 2 * limit, (max(built), limit)
