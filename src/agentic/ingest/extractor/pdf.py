"""
PDFExtractor - extract text from PDF documents with fallback strategy.

Adapted from proven implementation in agentic/etl/transformers/extractors/pdf.py.
Uses fallback strategy: LightOnOCR → OpenDataLoader → PyMuPDF (fitz) → pdfplumber
"""

import asyncio
import base64
import glob as glob_mod
import io
import logging
import math
import os
import re
import ssl
import tempfile
from concurrent.futures import ThreadPoolExecutor

from requests import exceptions as requests_exceptions

from agentic.ingest.extractor.base import (
    ExtractionError,
    Extractor,
    replace_image_annotations,
)
from agentic.ingest.models import Derivative, ExtractionResult, RawContent

logger = logging.getLogger(__name__)


def _count_pages(data: bytes) -> int:
    """Count pages in a PDF from bytes using fitz, with regex fallback."""
    try:
        import fitz
    except ImportError:
        pass  # fall through to regex
    else:
        doc = fitz.open(stream=data, filetype="pdf")
        try:
            return len(doc)
        finally:
            doc.close()
    # Regex fallback (used when fitz is not installed)
    rxcountpages = re.compile(
        rb"/Type\s*/Page([^s]|$)", re.MULTILINE | re.DOTALL
    )
    return len(rxcountpages.findall(data))


class _PageAborted(Exception):
    """A page that was never sent because an earlier page had already failed.

    Not a failure of its own: it carries no diagnosis and must never be the
    error the document reports.
    """


def _response_excerpt(response, limit: int = 200) -> str:
    """``(status N; body: '...')`` for a response, or a note that it is unread.

    ``.text`` is read inside exception handlers, and on a streamed or
    already-closed response reading it can itself raise — which would replace
    the error being reported with one about reporting. An unread body says so
    rather than rendering as ``body: ''``, which is indistinguishable from a
    response that genuinely had none.
    """
    status = getattr(response, "status_code", "?")
    try:
        body = (getattr(response, "text", "") or "").strip().replace("\n", " ")
    except Exception as read_error:  # pragma: no cover - see docstring
        return f" (status {status}; body unreadable: {type(read_error).__name__})"
    if limit > 3 and len(body) > limit:
        body = body[: limit - 3] + "..."
    elif len(body) > limit:
        body = body[:limit]
    return f" (status {status}; body: {body!r})"


def _lighton_error_detail(exc: Exception, limit: int = 200) -> str:
    """Status and body excerpt for an OCR failure, when the error carries one.

    ``404 Client Error: Not Found for url: ...`` says nothing about *why* the
    endpoint refused; the body usually does, and is the only way to tell a
    rejected page apart from a rate limit or an expired key.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return ""
    return _response_excerpt(response, limit)


def _lighton_retry_after(exc: Exception, cap: float) -> float | None:
    """Seconds the endpoint asked us to wait, if it said so and we believe it.

    Only the delta-seconds form is read. The HTTP-date form is legal and
    would need clock-skew handling to be worth anything, so it is ignored
    rather than half-supported.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        raw = headers.get("Retry-After")
    except Exception:  # pragma: no cover - headers is a mapping in practice
        return None
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return None
    # isfinite, not `seconds < 0`: NaN compares False against everything, so a
    # `Retry-After: NaN` passes a negative check and survives min() unchanged
    # — and asyncio.sleep(nan) raises, from inside the handler carrying the
    # 429. inf is rejected here too; a header that says "wait forever" is not
    # a wait, it is a broken header.
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return min(seconds, cap)


# Failures that re-sending the same request cannot fix. `requests` raises
# these for a bad URL or a bad certificate — configuration, not weather — and
# retrying a MemoryError on a base64 workload makes the shortage worse.
_LIGHTON_FATAL_ERRORS: tuple[type[BaseException], ...] = (
    MemoryError,
    TypeError,
    AttributeError,
    NameError,
    ssl.SSLError,
    requests_exceptions.InvalidURL,
    requests_exceptions.MissingSchema,
    requests_exceptions.InvalidSchema,
    requests_exceptions.URLRequired,
)


def _lighton_is_retryable(exc: Exception) -> bool:
    """Whether re-sending the same page could plausibly succeed.

    5xx, 408 and 429 are worth another attempt. Every other 4xx is the
    endpoint saying this request is wrong — a bad key, an oversized page, an
    unknown model — and ``_try_method`` retries the *whole document*, so
    treating those as retryable spends three full passes to be refused three
    times. Raised as ExtractionError instead, which the chain already routes
    straight to the next extraction method.
    """
    if isinstance(exc, _LIGHTON_FATAL_ERRORS):
        return False
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if not isinstance(status, int) or not 400 <= status < 500:
        return True
    return status in (408, 429)


def _split_pdf(data: bytes, batch_size: int) -> list[tuple[bytes, int, int]]:
    """Split a PDF into batches of at most *batch_size* pages.

    Returns a list of (pdf_bytes, start_page, end_page) tuples where
    start_page and end_page are 1-indexed inclusive.
    """
    import fitz

    src = fitz.open(stream=data, filetype="pdf")
    try:
        total = len(src)
        batches: list[tuple[bytes, int, int]] = []

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total) - 1  # 0-indexed inclusive
            batch_doc = fitz.open()  # new empty PDF
            try:
                batch_doc.insert_pdf(src, from_page=start, to_page=end)
                batches.append((batch_doc.tobytes(), start + 1, end + 1))  # 1-indexed
            finally:
                batch_doc.close()

        return batches
    finally:
        src.close()


class PDFExtractor(Extractor):
    """
    PDF extraction with fallback strategy.

    Extraction methods (in order of preference):
    1. LightOnOCR - Default OCR head for scanned PDFs (auto mode; requires API key)
    2. Mistral OCR - OCR fallback behind LightOnOCR (auto mode; requires API key)
    3. OpenDataLoader - High-accuracy structural extraction (local, needs Java)
    4. PyMuPDF (fitz) - Fast, good for text-based PDFs
    5. pdfplumber - Fallback for edge cases

    In auto mode LightOnOCR runs first as the default head, then Mistral OCR,
    then the local chain (opendataloader → fitz → pdfplumber). Either OCR entry
    is skipped when its API key is not configured. Mistral sits ahead of the
    local methods because those read the text layer only — dropping a scanned
    PDF straight to them yields empty output instead of an error. PaddleOCR and
    LlamaParse are available only when explicitly requested. When an explicitly
    requested cloud OCR method (mistral, paddleocr, lighton, llamaparse) fails,
    extraction falls back to those same local methods.

    Example:
        >>> extractor = PDFExtractor()
        >>> raw = RawContent(content=pdf_bytes, mime_type="application/pdf", ...)
        >>> result = await extractor.extract(raw)
        >>> print(result.get_primary_text())
    """

    name = "pdf"
    supported_types = ["application/pdf"]

    def __init__(
        self,
        mistral_api_key: str | None = None,
        paddleocr_api_key: str | None = None,
        paddleocr_base_url: str | None = None,
        lighton_api_key: str | None = None,
        lighton_base_url: str | None = None,
        llamaparse_api_key: str | None = None,
        llamaparse_base_url: str | None = None,
        max_pages: int = 1000,
    ):
        """
        Initialize PDF extractor.

        Args:
            mistral_api_key: API key for Mistral OCR (optional, enables OCR)
            paddleocr_api_key: API key for PaddleOCR-VL (optional)
            paddleocr_base_url: Base URL for PaddleOCR-VL API (optional)
            lighton_api_key: API key for LightOnOCR (optional)
            lighton_base_url: Base URL for LightOnOCR API (optional)
            llamaparse_api_key: API key for LlamaParse / LlamaCloud (optional)
            llamaparse_base_url: Base URL for LlamaParse API (optional)
            max_pages: Maximum pages per Mistral OCR API call (batch size)
        """
        self.mistral_api_key = mistral_api_key
        self.paddleocr_api_key = paddleocr_api_key
        self.paddleocr_base_url = paddleocr_base_url
        self.lighton_api_key = lighton_api_key
        self.lighton_base_url = lighton_base_url
        self.llamaparse_api_key = llamaparse_api_key
        self.llamaparse_base_url = llamaparse_base_url
        self.max_pages = max_pages

    def _render_page_images(self, raw: RawContent, dpi: int = 150) -> list[Derivative]:
        """Render each PDF page as a PNG image derivative using PyMuPDF.

        Args:
            raw: RawContent with PDF bytes
            dpi: Resolution for rendering (default 150)

        Returns:
            List of image Derivative objects, one per page.
        """
        try:
            import fitz
        except ImportError:
            logger.warning("PyMuPDF not available for page image rendering")
            return []

        try:
            doc = fitz.open(stream=raw.content, filetype="pdf")
            try:
                image_derivs = []
                zoom = dpi / 72
                mat = fitz.Matrix(zoom, zoom)
                for page in doc:
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    image_derivs.append(
                        Derivative(
                            type="image",
                            content=pix.tobytes("png"),
                            format="png",
                            page=page.number + 1,  # 1-indexed
                            metadata={"width": pix.width, "height": pix.height, "dpi": dpi},
                        )
                    )
            finally:
                doc.close()
            return image_derivs
        except Exception as e:
            logger.warning(f"Failed to render page images: {e}")
            return []

    async def _try_method(self, method: str, raw: RawContent) -> ExtractionResult:
        """Call a single extraction method by name.

        For ``mistral``, applies the existing retry-with-backoff logic.
        Other methods are called directly.
        """
        if method == "mistral":
            if not self.mistral_api_key:
                raise ExtractionError(
                    "Mistral OCR requested but MISTRAL_API_KEY is not configured",
                    extractor_name=self.name,
                    source_uri=raw.source_uri,
                )
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    return await self._extract_mistral(raw)
                except ExtractionError:
                    raise  # deterministic — don't retry
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt  # 1s, 2s
                        logger.warning(
                            f"Mistral OCR attempt {attempt + 1}/{max_retries} failed: {e}, "
                            f"retrying in {wait}s"
                        )
                        await asyncio.sleep(wait)
                    else:
                        raise
            # unreachable, but keeps mypy happy
            raise ExtractionError(
                "Mistral OCR failed", extractor_name=self.name, source_uri=raw.source_uri
            )
        elif method == "paddleocr":
            if not self.paddleocr_api_key:
                raise ExtractionError(
                    "PaddleOCR requested but PADDLEOCR_API_KEY is not configured",
                    extractor_name=self.name,
                    source_uri=raw.source_uri,
                )
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    return await self._extract_paddleocr(raw)
                except ExtractionError:
                    raise
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt
                        logger.warning(
                            f"PaddleOCR attempt {attempt + 1}/{max_retries} failed: {e}, "
                            f"retrying in {wait}s"
                        )
                        await asyncio.sleep(wait)
                    else:
                        raise
            raise ExtractionError(
                "PaddleOCR failed", extractor_name=self.name, source_uri=raw.source_uri
            )
        elif method == "lighton":
            if not self.lighton_api_key:
                raise ExtractionError(
                    "LightOnOCR requested but LIGHTON_API_KEY is not configured",
                    extractor_name=self.name,
                    source_uri=raw.source_uri,
                )
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    return await self._extract_lighton(raw)
                except ExtractionError:
                    raise
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt
                        logger.warning(
                            f"LightOnOCR attempt {attempt + 1}/{max_retries} failed: {e}, "
                            f"retrying in {wait}s"
                        )
                        await asyncio.sleep(wait)
                    else:
                        raise
            raise ExtractionError(
                "LightOnOCR failed", extractor_name=self.name, source_uri=raw.source_uri
            )
        elif method == "llamaparse":
            if not self.llamaparse_api_key:
                raise ExtractionError(
                    "LlamaParse requested but LLAMAPARSE_API_KEY is not configured",
                    extractor_name=self.name,
                    source_uri=raw.source_uri,
                )
            # Deliberately NO outer retry loop here: parse_pages already retries
            # the poll/result GETs against the SAME job internally. Retrying the
            # whole flow would re-upload and start a NEW billable parse job on
            # every transient error (up to 3 charges for one extraction).
            return await self._extract_llamaparse(raw)
        elif method == "opendataloader":
            return self._extract_opendataloader(raw)
        elif method == "fitz":
            return self._extract_fitz(raw)
        elif method == "pdfplumber":
            return self._extract_pdfplumber(raw)
        else:
            raise ExtractionError(
                f"Unknown extraction method: {method}",
                extractor_name=self.name,
                source_uri=raw.source_uri,
            )

    async def extract(self, raw: RawContent) -> ExtractionResult:
        """
        Extract text from PDF with fallback strategy.

        Reads ``raw.metadata["extraction_model"]`` to decide which method
        to use.  ``"auto"`` (the default) iterates through the configured
        fallback chain.  A specific method name calls that method only.

        Args:
            raw: RawContent with PDF bytes

        Returns:
            ExtractionResult with text derivative
        """
        from agentic.knowledge.model_config import (
            EXTRACTION_DEFAULT_METHOD,
            EXTRACTION_FALLBACK_CHAIN,
        )

        preference = (raw.metadata or {}).get(
            "extraction_model", EXTRACTION_DEFAULT_METHOD
        )

        LOCAL_METHODS = {"opendataloader", "fitz", "pdfplumber"}

        if preference != "auto":
            try:
                return await self._try_method(preference, raw)
            except Exception as e:
                # Local methods have no further fallback
                if preference in LOCAL_METHODS:
                    raise
                # Cloud/OCR methods fall back to local chain
                logger.warning(
                    f"{preference} extraction failed: {e}, "
                    f"falling back to local extraction methods"
                )
                for method in ["opendataloader", "fitz", "pdfplumber"]:
                    try:
                        result = await self._try_method(method, raw)
                        result.auto_metadata["requested_method"] = preference
                        result.auto_metadata["fallback_reason"] = str(e)
                        return result
                    except Exception as fallback_err:
                        logger.warning(
                            f"Local fallback {method} also failed: {fallback_err}"
                        )
                raise ExtractionError(
                    f"{preference} and all local fallbacks failed. "
                    f"Original error: {e}",
                    extractor_name=self.name,
                    source_uri=raw.source_uri,
                ) from e

        # Auto mode — iterate fallback chain
        chain = list(EXTRACTION_FALLBACK_CHAIN)
        # Skip lighton in the chain when no API key is configured. lighton is the
        # default OCR head, so make the skip observable: an operator who forgot to
        # set LIGHTON_API_KEY would otherwise silently get the local-only chain
        # (no OCR on scanned PDFs) with nothing in the logs to explain it.
        if not self.lighton_api_key and "lighton" in chain:
            logger.info(
                "LIGHTON_API_KEY not configured; skipping lighton (the default OCR "
                "head) and falling back to the rest of the chain instead"
            )
            chain = [m for m in chain if m != "lighton"]

        # Same for mistral, the OCR-preserving second hop. Attempting it unkeyed
        # would raise and log a failure on every extraction — noise for the many
        # deployments that configure no MISTRAL_API_KEY. An absent optional key is
        # expected configuration, so this is INFO, matching the lighton skip.
        if not self.mistral_api_key and "mistral" in chain:
            logger.info(
                "MISTRAL_API_KEY not configured; skipping mistral (the OCR fallback "
                "behind lighton) and falling back to the rest of the chain instead"
            )
            chain = [m for m in chain if m != "mistral"]

        last_error: Exception | None = None
        for i, method in enumerate(chain):
            try:
                return await self._try_method(method, raw)
            except Exception as e:
                last_error = e
                remaining = chain[i + 1:]
                if remaining:
                    logger.warning(
                        f"{method} extraction failed: {e}, "
                        f"falling back to {remaining[0]}"
                    )
                else:
                    logger.error(f"{method} extraction failed: {e}, no more fallbacks")

        raise ExtractionError(
            f"All extraction methods failed. Last error: {last_error}",
            extractor_name=self.name,
            source_uri=raw.source_uri,
        ) from last_error

    def _extract_opendataloader(self, raw: RawContent) -> ExtractionResult:
        """Extract using opendataloader-pdf — high-accuracy structural extraction.

        Splits the PDF into single-page PDFs and extracts each independently
        to guarantee 1:1 alignment between page text and page images.
        """
        try:
            import opendataloader_pdf
        except ImportError:
            raise ExtractionError(
                "opendataloader-pdf is required. Install with: pip install opendataloader-pdf",
                extractor_name=self.name,
                source_uri=raw.source_uri,
            ) from None

        try:
            import fitz
        except ImportError:
            raise ExtractionError(
                "PyMuPDF (fitz) is required for per-page PDF splitting. "
                "Install with: pip install pymupdf",
                extractor_name=self.name,
                source_uri=raw.source_uri,
            ) from None

        src_doc = fitz.open(stream=raw.content, filetype="pdf")
        try:
            num_pages = len(src_doc)

            page_markdowns = []
            page_text_derivs = []

            with tempfile.TemporaryDirectory() as tmpdir:
                for page_num in range(num_pages):
                    # Create a single-page PDF via insert_pdf (same pattern as _split_pdf)
                    single = fitz.open()  # new empty PDF
                    try:
                        single.insert_pdf(src_doc, from_page=page_num, to_page=page_num)
                        page_pdf_path = os.path.join(tmpdir, f"page_{page_num}.pdf")
                        single.save(page_pdf_path)
                    finally:
                        single.close()

                    # Extract with opendataloader
                    page_output_dir = os.path.join(tmpdir, f"output_{page_num}")
                    opendataloader_pdf.convert(
                        input_path=[page_pdf_path],
                        output_dir=page_output_dir,
                        format="markdown",
                        quiet=True,
                    )

                    md_files = glob_mod.glob(
                        os.path.join(page_output_dir, "**", "*.md"), recursive=True
                    )
                    page_md = ""
                    if md_files:
                        with open(md_files[0], encoding="utf-8") as f:
                            page_md = f.read().strip()

                    page_markdowns.append(page_md)
                    page_text_derivs.append(
                        Derivative(
                            type="page_text",
                            content=page_md,
                            format="plain",
                            page=page_num + 1,
                        )
                    )
        finally:
            src_doc.close()

        fulltext = "\n\n".join(page_markdowns)

        derivatives = [
            Derivative(type="markdown", content=fulltext, format="markdown"),
        ]
        derivatives.extend(page_text_derivs)
        image_derivs = self._render_page_images(raw)
        derivatives.extend(image_derivs)

        return ExtractionResult(
            source_uri=raw.source_uri,
            mime_type=raw.mime_type,
            derivatives=derivatives,
            auto_metadata={
                "page_count": num_pages,
                "char_count": len(fulltext),
            },
            extraction_method="opendataloader",
            stats={"pages_processed": num_pages},
        )

    def _extract_fitz(self, raw: RawContent) -> ExtractionResult:
        """Extract using PyMuPDF (Fitz) - proven implementation."""
        try:
            import fitz
        except ImportError:
            raise ExtractionError(
                "PyMuPDF (fitz) is required for PDF extraction. "
                "Install with: pip install pymupdf",
                extractor_name=self.name,
                source_uri=raw.source_uri,
            ) from None

        doc = fitz.open(stream=raw.content, filetype="pdf")
        try:
            page_text_strings = []
            page_text_derivs = []

            for page in doc:
                blocks = page.get_text("blocks")
                block_texts = []
                for block in blocks:
                    # block format: (x0, y0, x1, y1, text, block_no, block_type)
                    if len(block) >= 7 and block[6] == 0:  # Text block
                        block_text = block[4]
                        if isinstance(block_text, str):
                            block_text = block_text.replace("\n", " ").strip()
                            if block_text:
                                block_texts.append(block_text)
                page_text = "\n".join(block_texts)
                page_text_strings.append(page_text)
                page_text_derivs.append(
                    Derivative(
                        type="page_text",
                        content=page_text,
                        format="plain",
                        page=page.number + 1,
                    )
                )

            fulltext = "\n\n".join(page_text_strings)
        finally:
            doc.close()

        derivatives = [
            Derivative(
                type="text",
                content=fulltext,
                format="plain",
            ),
        ]
        derivatives.extend(page_text_derivs)

        # Render page images for image-mode retrieval
        image_derivs = self._render_page_images(raw)
        derivatives.extend(image_derivs)

        return ExtractionResult(
            source_uri=raw.source_uri,
            mime_type=raw.mime_type,
            derivatives=derivatives,
            auto_metadata={
                "page_count": len(page_text_strings),
                "char_count": len(fulltext),
            },
            extraction_method="fitz",
            stats={"pages_processed": len(page_text_strings)},
        )

    async def _extract_mistral(self, raw: RawContent) -> ExtractionResult:
        """Orchestrate Mistral OCR — batching large PDFs automatically."""
        page_count = _count_pages(raw.content)
        logger.info(
            f"Mistral OCR: {page_count} pages, batch size {self.max_pages}"
        )

        if page_count <= self.max_pages:
            # Fast path — single API call, render images directly
            return await self._extract_mistral_single(
                raw.content, raw, page_offset=0, render_images=True
            )

        # Batch path — split PDF, process sequentially, combine
        logger.info(
            f"Splitting {page_count}-page PDF into batches of {self.max_pages}"
        )
        try:
            batches = _split_pdf(raw.content, self.max_pages)
        except ImportError:
            raise ExtractionError(
                "PyMuPDF (fitz) is required for PDF splitting. "
                "Install with: pip install pymupdf",
                extractor_name=self.name,
                source_uri=raw.source_uri,
            ) from None
        logger.info(f"Created {len(batches)} batches")

        batch_results: list[ExtractionResult] = []
        for i, (batch_bytes, start_page, end_page) in enumerate(batches):
            logger.info(
                f"Processing batch {i + 1}/{len(batches)}: "
                f"pages {start_page}-{end_page}"
            )
            # page_offset so page_text derivatives get correct 1-indexed numbers
            result = await self._extract_mistral_single(
                batch_bytes,
                raw,
                page_offset=start_page - 1,
                render_images=False,
            )
            batch_results.append(result)

        return self._combine_batch_results(batch_results, raw, page_count)

    async def _extract_mistral_single(
        self,
        pdf_bytes: bytes,
        raw: RawContent,
        page_offset: int = 0,
        render_images: bool = True,
    ) -> ExtractionResult:
        """Send a single PDF (or batch) to Mistral OCR.

        Args:
            pdf_bytes: The PDF bytes to send.
            raw: Original RawContent (for source_uri, filename, full-PDF images).
            page_offset: Added to each page index so batch pages get correct
                         1-indexed numbers in the final result.
            render_images: Whether to render page images (skip in batch mode).
        """
        try:
            from enum import Enum

            from mistralai.client import Mistral
            from pydantic import BaseModel, Field
        except ImportError:
            raise ExtractionError(
                "mistralai is required for Mistral OCR. "
                "Install with: pip install mistralai",
                extractor_name=self.name,
                source_uri=raw.source_uri,
            ) from None

        # Need to write to temp file for Mistral API
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                temp_path = f.name
                f.write(pdf_bytes)
            # 5-min per-operation timeout. Without this, an OCR call that
            # never completes blocks the worker thread forever and wedges the
            # entire Celery threads pool (no thread becomes free to pick up
            # new tasks; pod looks Running to K8s but processes nothing).
            client = Mistral(api_key=self.mistral_api_key, timeout_ms=300_000)

            with open(temp_path, "rb") as f:
                uploaded_file = client.files.upload(
                    file={"file_name": raw.filename or "document.pdf", "content": f},
                    purpose="ocr",
                )

            file_url = client.files.get_signed_url(file_id=uploaded_file.id)

            # Define image annotation model
            class ImageType(str, Enum):
                GRAPH = "graph"
                TEXT = "text"
                TABLE = "table"
                IMAGE = "image"

            class Image(BaseModel):
                image_type: ImageType = Field(..., description="The type of the image.")
                description: str = Field(..., description="A description of the image.")

            from mistralai.extra import response_format_from_pydantic_model

            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={"type": "document_url", "document_url": file_url.url},
                bbox_annotation_format=response_format_from_pydantic_model(Image),
                include_image_base64=False,
            )

            # Process pages with image annotations
            page_markdowns = []
            page_text_derivs = []
            for page in ocr_response.pages:
                annotations = [
                    (img.id, img.image_annotation) for img in page.images
                ]
                page.markdown = replace_image_annotations(
                    page.markdown, annotations
                )
                page_markdowns.append(page.markdown)
                page_meta = {}
                if page.dimensions is not None:
                    page_meta["width"] = page.dimensions.width
                    page_meta["height"] = page.dimensions.height
                    page_meta["dpi"] = page.dimensions.dpi
                page_text_derivs.append(
                    Derivative(
                        type="page_text",
                        content=page.markdown,
                        format="plain",
                        page=page.index + 1 + page_offset,
                        metadata=page_meta,
                    )
                )

            fulltext = "\n\n".join(page_markdowns)

            derivatives = [
                Derivative(
                    type="markdown",
                    content=fulltext,
                    format="markdown",
                ),
            ]
            derivatives.extend(page_text_derivs)

            # Render page images only for single-PDF path (not per-batch)
            if render_images:
                image_derivs = self._render_page_images(raw)
                derivatives.extend(image_derivs)

            return ExtractionResult(
                source_uri=raw.source_uri,
                mime_type=raw.mime_type,
                derivatives=derivatives,
                auto_metadata={
                    "page_count": len(page_markdowns),
                    "char_count": len(fulltext),
                },
                extraction_method="mistral_ocr",
                stats={"pages_processed": len(page_markdowns)},
            )
        finally:
            if temp_path:
                os.unlink(temp_path)

    def _combine_batch_results(
        self,
        batch_results: list[ExtractionResult],
        raw: RawContent,
        total_pages: int,
    ) -> ExtractionResult:
        """Merge results from multiple Mistral OCR batches into one result."""
        all_page_text_derivs: list[Derivative] = []
        markdown_parts: list[str] = []
        total_processed = 0

        for result in batch_results:
            for d in result.derivatives:
                if d.type == "page_text":
                    all_page_text_derivs.append(d)
                elif d.type == "markdown":
                    markdown_parts.append(d.content)
            total_processed += result.stats.get("pages_processed", 0)

        fulltext = "\n\n".join(markdown_parts)

        derivatives = [
            Derivative(
                type="markdown",
                content=fulltext,
                format="markdown",
            ),
        ]
        derivatives.extend(all_page_text_derivs)

        # Render page images once from the full original PDF
        image_derivs = self._render_page_images(raw)
        derivatives.extend(image_derivs)

        logger.info(
            f"Combined {len(batch_results)} batches: "
            f"{total_processed} pages processed"
        )

        return ExtractionResult(
            source_uri=raw.source_uri,
            mime_type=raw.mime_type,
            derivatives=derivatives,
            auto_metadata={
                "page_count": total_pages,
                "char_count": len(fulltext),
            },
            extraction_method="mistral_ocr",
            stats={
                "pages_processed": total_processed,
                "batch_count": len(batch_results),
            },
        )

    async def _extract_paddleocr(self, raw: RawContent) -> ExtractionResult:
        """Extract using PaddleOCR-VL layout parsing API."""
        import requests

        from agentic.knowledge.model_config import (
            PADDLEOCR_DEFAULT_BASE_URL,
            PADDLEOCR_FILE_TYPE_PDF,
            PADDLEOCR_LAYOUT_PARSING_PATH,
            PADDLEOCR_TIMEOUT,
            PADDLEOCR_USE_CHART_RECOGNITION,
            PADDLEOCR_USE_DOC_ORIENTATION_CLASSIFY,
            PADDLEOCR_USE_DOC_UNWARPING,
        )

        base_url = self.paddleocr_base_url or PADDLEOCR_DEFAULT_BASE_URL
        url = f"{base_url}{PADDLEOCR_LAYOUT_PARSING_PATH}"

        b64_pdf = base64.b64encode(raw.content).decode("utf-8")
        payload = {
            "file": b64_pdf,
            "fileType": PADDLEOCR_FILE_TYPE_PDF,
            "useDocOrientationClassify": PADDLEOCR_USE_DOC_ORIENTATION_CLASSIFY,
            "useDocUnwarping": PADDLEOCR_USE_DOC_UNWARPING,
            "useChartRecognition": PADDLEOCR_USE_CHART_RECOGNITION,
        }
        headers = {
            "Authorization": f"token {self.paddleocr_api_key}",
            "Content-Type": "application/json",
        }

        logger.info("PaddleOCR: sending PDF (%d bytes) to %s", len(raw.content), url)

        resp = requests.post(url, json=payload, headers=headers, timeout=PADDLEOCR_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        layout_results = data.get("result", {}).get("layoutParsingResults", [])
        if not layout_results:
            raise ExtractionError(
                "PaddleOCR returned no layout parsing results",
                extractor_name=self.name,
                source_uri=raw.source_uri,
            )

        page_markdowns = []
        page_text_derivs = []
        for i, res in enumerate(layout_results):
            md_block = res.get("markdown", {})
            page_md = md_block.get("text", "")

            # Strip embedded image references (temporary URLs we don't host).
            # PaddleOCR images dict maps {image_path: image_url}.
            # The markdown contains ![...](image_url) patterns we need to remove.
            images = md_block.get("images", {})
            if images:
                for img_path, img_url in images.items():
                    # Remove ![alt](url) where url matches the temp image URL
                    if img_url:
                        pattern = r"!\[[^\]]*\]\(" + re.escape(img_url) + r"\)"
                        page_md = re.sub(pattern, "", page_md)
                    # Also remove references using the image path as URL
                    if img_path:
                        pattern = r"!\[[^\]]*\]\(" + re.escape(img_path) + r"\)"
                        page_md = re.sub(pattern, "", page_md)

            page_markdowns.append(page_md)
            page_text_derivs.append(
                Derivative(
                    type="page_text",
                    content=page_md,
                    format="plain",
                    page=i + 1,
                )
            )

        fulltext = "\n\n".join(page_markdowns)

        derivatives = [
            Derivative(
                type="markdown",
                content=fulltext,
                format="markdown",
            ),
        ]
        derivatives.extend(page_text_derivs)

        # Render page images for image-mode retrieval
        image_derivs = self._render_page_images(raw)
        derivatives.extend(image_derivs)

        return ExtractionResult(
            source_uri=raw.source_uri,
            mime_type=raw.mime_type,
            derivatives=derivatives,
            auto_metadata={
                "page_count": len(page_markdowns),
                "char_count": len(fulltext),
            },
            extraction_method="paddleocr_vl",
            stats={"pages_processed": len(page_markdowns)},
        )

    async def _extract_lighton(self, raw: RawContent) -> ExtractionResult:
        """Extract using LightOnOCR via OpenAI-compatible chat completions API."""
        import requests

        from agentic.knowledge.model_config import (
            LIGHTON_DEFAULT_BASE_URL,
            LIGHTON_DEFAULT_MODEL,
            LIGHTON_MAX_CONCURRENCY,
            LIGHTON_MAX_TOKENS,
            LIGHTON_PAGE_MAX_ATTEMPTS,
            LIGHTON_PAGE_RETRY_BACKOFF,
            LIGHTON_RETRY_AFTER_MAX,
            LIGHTON_TEMPERATURE,
            LIGHTON_TIMEOUT,
            LIGHTON_TOP_P,
        )

        base_url = self.lighton_base_url or LIGHTON_DEFAULT_BASE_URL
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.lighton_api_key}",
            "Content-Type": "application/json",
        }

        # Render pages once — reuse for both API calls and image derivatives
        image_derivs = self._render_page_images(raw)
        if not image_derivs:
            raise ExtractionError(
                "LightOnOCR requires PyMuPDF to render page images",
                extractor_name=self.name,
                source_uri=raw.source_uri,
            )

        logger.info("LightOnOCR: processing %d pages via %s", len(image_derivs), url)

        semaphore = asyncio.Semaphore(LIGHTON_MAX_CONCURRENCY)
        loop = asyncio.get_running_loop()

        # Set by the first page that fails. Pages still queued behind the
        # semaphore check it and never send a request: the caller re-runs the
        # whole document, so they are work about to be thrown away, against an
        # endpoint that may be failing us for load in the first place. Pages
        # already on the wire are *not* abandoned — their outcomes are what
        # decide whether this document is retried at all (see below).
        aborted = False
        # What the retry cost, so a document that needed it is distinguishable
        # from a clean one: same derivatives and same page count either way.
        retried_pages: set = set()
        retry_requests = [0]

        async def _ocr_page(executor, img_deriv: Derivative) -> str:
            nonlocal aborted
            async with semaphore:
                if aborted:
                    raise _PageAborted(img_deriv.page)
                # Built inside the semaphore, not before it. Every task runs
                # to its first suspension the moment it is scheduled, so a
                # payload constructed above this line exists once per page
                # rather than once per slot: 252 pages of base64 (~0.5 MB
                # each, held twice while the concatenation builds the data:
                # URI) instead of eight.
                payload = {
                    "model": LIGHTON_DEFAULT_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "data:image/png;base64,"
                                        + base64.b64encode(img_deriv.content).decode(
                                            "utf-8"
                                        ),
                                    },
                                }
                            ],
                        }
                    ],
                    "max_tokens": LIGHTON_MAX_TOKENS,
                    "temperature": LIGHTON_TEMPERATURE,
                    "top_p": LIGHTON_TOP_P,
                }

                # A dedicated executor, not asyncio.to_thread: that one uses
                # the loop's default pool, sized min(32, cpu_count + 4) — six
                # threads on a 2-vCPU container, which this alone would
                # saturate, stalling every other to_thread user in the
                # process (docx, pptx, xlsx, the embedder). `requests` is
                # synchronous, so calling it without a thread at all would
                # block the event loop for the whole round trip.
                # Retry the *page*, not the document. The document retry in
                # `_try_method` re-renders and re-sends every page, so one 429
                # on the last page of a long filing discards every page that
                # succeeded. Here the successes stay and only this page goes
                # again — and the semaphore slot is deliberately held across
                # the wait, because when the cause is a rate limit the useful
                # response is fewer requests in flight, not the same number.
                for attempt in range(1, LIGHTON_PAGE_MAX_ATTEMPTS + 1):
                    try:
                        resp = await loop.run_in_executor(
                            executor,
                            lambda: requests.post(
                                url,
                                json=payload,
                                headers=headers,
                                timeout=LIGHTON_TIMEOUT,
                            ),
                        )
                        resp.raise_for_status()
                        break
                    except Exception as e:
                        last_attempt = attempt >= LIGHTON_PAGE_MAX_ATTEMPTS
                        if _lighton_is_retryable(e) and not last_attempt:
                            wait = _lighton_retry_after(e, LIGHTON_RETRY_AFTER_MAX)
                            if wait is None:
                                wait = LIGHTON_PAGE_RETRY_BACKOFF * 2 ** (attempt - 1)
                            # Clamped again at the point of use: nothing that
                            # reaches a sleep inside an exception handler may
                            # be able to raise there, whatever a future
                            # caller computes.
                            wait = max(0.0, min(float(wait), LIGHTON_RETRY_AFTER_MAX))
                            # INFO, not WARNING: a page that recovers is not
                            # something an operator needs to act on, and at
                            # 252 pages the warning level turns a successful
                            # extraction into hundreds of them. The final
                            # failure below stays at WARNING.
                            logger.info(
                                "LightOnOCR page %s of %s attempt %d/%d failed: "
                                "%s%s — retrying in %.1fs",
                                img_deriv.page,
                                raw.source_uri,
                                attempt,
                                LIGHTON_PAGE_MAX_ATTEMPTS,
                                e,
                                _lighton_error_detail(e),
                                wait,
                            )
                            retried_pages.add(img_deriv.page)
                            retry_requests[0] += 1
                            await asyncio.sleep(wait)
                            # Re-checked after the wait, not only above the
                            # loop: a page that was already sleeping when
                            # another page failed the document would otherwise
                            # wake and send a fresh request at an endpoint we
                            # have given up on — and backoff is exactly the
                            # rate-limit case. It also holds the known failure
                            # back until every retrying sibling settles.
                            if aborted:
                                raise _PageAborted(img_deriv.page) from None
                            continue

                        # Out of attempts, or an error re-sending cannot fix.
                        # Serially the failing page was wherever the loop
                        # stopped; concurrently it is whichever page loses the
                        # race, so the page number has to be in the log — the
                        # caller only ever reports one exception.
                        aborted = True
                        logger.warning(
                            "LightOnOCR page %s of %s failed after %d attempt(s): %s%s",
                            img_deriv.page,
                            raw.source_uri,
                            attempt,
                            e,
                            _lighton_error_detail(e),
                        )
                        if _lighton_is_retryable(e):
                            raise
                        # "rejected" only when the endpoint did the rejecting.
                        # A bad certificate or an unparseable URL is ours, and
                        # calling it a rejected page sends the reader looking
                        # at the document.
                        refused = _lighton_error_detail(e)
                        reason = (
                            f"rejected page {img_deriv.page}"
                            if refused
                            else f"cannot reach the endpoint for page {img_deriv.page}"
                        )
                        raise ExtractionError(
                            f"LightOnOCR {reason}: {type(e).__name__}: {e}{refused}",
                            extractor_name=self.name,
                            source_uri=raw.source_uri,
                        ) from e

                try:
                    return resp.json()["choices"][0]["message"]["content"]
                except Exception as e:
                    # Broad on purpose. JSONDecodeError is a ValueError and a
                    # bad envelope is a KeyError, but a connection dropping
                    # mid-read raises ChunkedEncodingError — which used to
                    # leave without setting the flag, so the queued pages kept
                    # going. The classification below still distinguishes the
                    # two kinds.
                    # A 2xx the client cannot read is deterministic: the same
                    # request will be unreadable next time, so retrying the
                    # document three times only asks three times. The body is
                    # attached because it holds the answer the exception does
                    # not — a KeyError carries no response, so `'choices'` is
                    # otherwise the operator's entire diagnosis.
                    aborted = True
                    excerpt = _response_excerpt(resp)
                    if not isinstance(e, (ValueError, KeyError, IndexError, TypeError)):
                        # The response never fully arrived — transient, and
                        # for the caller to retry rather than a body we can
                        # describe.
                        logger.warning(
                            "LightOnOCR page %s of %s failed while reading the "
                            "response: %s: %s",
                            img_deriv.page,
                            raw.source_uri,
                            type(e).__name__,
                            e,
                        )
                        raise
                    logger.warning(
                        "LightOnOCR page %s of %s returned an unusable body: %s: %s%s",
                        img_deriv.page,
                        raw.source_uri,
                        type(e).__name__,
                        e,
                        excerpt,
                    )
                    raise ExtractionError(
                        f"LightOnOCR returned an unusable body for page "
                        f"{img_deriv.page}: {type(e).__name__}: {e}{excerpt}",
                        extractor_name=self.name,
                        source_uri=raw.source_uri,
                    ) from e

        # Sized to the semaphore, which is what the endpoint sees; a wider
        # pool would only hold work the semaphore has not admitted.
        executor = ThreadPoolExecutor(
            max_workers=LIGHTON_MAX_CONCURRENCY, thread_name_prefix="lighton-ocr"
        )
        try:
            # `return_exceptions=True`, and no cancellation of siblings: the
            # abort flag has already stopped the queue, so what is left in
            # flight is at most one window of requests that are going to
            # finish regardless — a thread inside requests.post cannot be
            # interrupted. Collecting them costs a bounded wait and buys the
            # two things abandoning them loses: every failure gets logged
            # rather than only whichever lost the race, and the *kind* of
            # failure is known before the document is retried. Reporting a
            # transient 503 while a deterministic 400 was still on the wire
            # costs three full passes and never names the 400 at all.
            #
            # A cancellation from the caller still propagates out of this
            # await, which is what a graceful drain or asyncio.timeout wants:
            # gather's return_exceptions only covers the children.
            #
            # The property that keeps it that way: no `except BaseException`,
            # no `contextlib.suppress`, no re-arming. Two earlier versions of
            # this function lost the caller's error to an await inside an
            # exception handler; the retry loop below still has one, which is
            # why the value it sleeps on is clamped where it is computed *and*
            # where it is used.
            outcomes = await asyncio.gather(*(
                _ocr_page(executor, d) for d in image_derivs
            ), return_exceptions=True)
        finally:
            # Every *task* has settled by here, but on an external
            # cancellation the threads those tasks were waiting on are still
            # inside requests.post — a thread cannot be interrupted. So this
            # releases the pool; it does not stop work in progress, and
            # wait=False is what keeps a cancelled extraction from blocking
            # on requests it can no longer use.
            executor.shutdown(wait=False)

        failures = [
            o
            for o in outcomes
            if isinstance(o, BaseException) and not isinstance(o, _PageAborted)
        ]
        # A _PageAborted can only exist alongside the failure that caused it:
        # both `aborted = True` sites raise in the same breath. Asserting it
        # rather than reasoning about it across eighty lines — the failure
        # mode otherwise is a page slot holding an exception where a string
        # belongs, which surfaces as a TypeError from the join below.
        if not failures and any(isinstance(o, _PageAborted) for o in outcomes):
            raise ExtractionError(
                "LightOnOCR aborted pages without recording why",
                extractor_name=self.name,
                source_uri=raw.source_uri,
            )
        if failures:
            # Deterministic first, whatever the wall clock said. ExtractionError
            # tells `_try_method` not to retry and to fall through to the next
            # extraction method; letting a slower 503 outrank it re-runs the
            # whole document to be refused again.
            deterministic = [f for f in failures if isinstance(f, ExtractionError)]
            reported = (deterministic or failures)[0]
            others = [f for f in failures if f is not reported]
            if others:
                logger.warning(
                    "LightOnOCR: %d pages failed on %s; reporting %s. Others: %s",
                    len(failures),
                    raw.source_uri,
                    "the deterministic rejection" if deterministic else "the first",
                    "; ".join(f"{type(f).__name__}: {f}" for f in others),
                )
            raise reported

        page_markdowns = list(outcomes)

        # Page order comes from the document, never from completion order.
        page_text_derivs = [
            Derivative(
                type="page_text",
                content=page_md,
                format="plain",
                page=img_deriv.page,
            )
            for img_deriv, page_md in zip(image_derivs, page_markdowns, strict=True)
        ]

        fulltext = "\n\n".join(page_markdowns)

        derivatives = [
            Derivative(
                type="markdown",
                content=fulltext,
                format="markdown",
            ),
        ]
        derivatives.extend(page_text_derivs)
        derivatives.extend(image_derivs)

        return ExtractionResult(
            source_uri=raw.source_uri,
            mime_type=raw.mime_type,
            derivatives=derivatives,
            auto_metadata={
                "page_count": len(page_markdowns),
                "char_count": len(fulltext),
            },
            extraction_method="lighton_ocr",
            stats={
                "pages_processed": len(page_markdowns),
                # A document that needed retries produces the same derivatives
                # and the same page count as one that did not; without these
                # a 3x request cost is invisible to every downstream signal.
                "pages_retried": len(retried_pages),
                "retry_requests": retry_requests[0],
            },
        )

    async def _extract_llamaparse(self, raw: RawContent) -> ExtractionResult:
        """Extract using LlamaParse (LlamaCloud) — advanced OCR for complex PDFs.

        Delegates the v2 "Agentic Plus" parse to the shared LlamaParse client and
        builds page_text + markdown derivatives plus rendered page images.
        """
        from agentic.ingest.extractor.llamaparse import parse_pages

        pages = await parse_pages(
            raw.content,
            api_key=self.llamaparse_api_key,
            base_url=self.llamaparse_base_url,
            source_uri=raw.source_uri,
            extractor_name=self.name,
            mime_type=raw.mime_type,
        )

        page_markdowns = [p["markdown"] for p in pages]
        page_text_derivs = [
            Derivative(type="page_text", content=p["markdown"], format="plain", page=p["page_number"])
            for p in pages
        ]

        fulltext = "\n\n".join(page_markdowns)

        derivatives = [
            Derivative(
                type="markdown",
                content=fulltext,
                format="markdown",
            ),
        ]
        derivatives.extend(page_text_derivs)
        # Render page images for image-mode retrieval (consistent with other methods)
        derivatives.extend(self._render_page_images(raw))

        return ExtractionResult(
            source_uri=raw.source_uri,
            mime_type=raw.mime_type,
            derivatives=derivatives,
            auto_metadata={
                "page_count": len(page_markdowns),
                "char_count": len(fulltext),
            },
            extraction_method="llamaparse_ocr",
            stats={"pages_processed": len(page_markdowns)},
        )

    def _extract_pdfplumber(self, raw: RawContent) -> ExtractionResult:
        """Extract using pdfplumber as final fallback."""
        try:
            import pdfplumber
        except ImportError:
            raise ExtractionError(
                "pdfplumber is required as PDF fallback. "
                "Install with: pip install pdfplumber",
                extractor_name=self.name,
                source_uri=raw.source_uri,
            ) from None

        text_parts = []
        page_text_derivs = []

        with pdfplumber.open(io.BytesIO(raw.content)) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
                page_text_derivs.append(
                    Derivative(
                        type="page_text",
                        content=page_text,
                        format="plain",
                        page=i + 1,
                    )
                )

        fulltext = "\n\n".join(text_parts)

        derivatives = [
            Derivative(
                type="text",
                content=fulltext,
                format="plain",
            ),
        ]
        derivatives.extend(page_text_derivs)

        # Render page images for image-mode retrieval
        image_derivs = self._render_page_images(raw)
        derivatives.extend(image_derivs)

        return ExtractionResult(
            source_uri=raw.source_uri,
            mime_type=raw.mime_type,
            derivatives=derivatives,
            auto_metadata={
                "page_count": len(text_parts),
                "char_count": len(fulltext),
            },
            extraction_method="pdfplumber",
            stats={"pages_processed": len(text_parts)},
        )
