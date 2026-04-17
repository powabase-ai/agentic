"""
PDFExtractor - extract text from PDF documents with fallback strategy.

Adapted from proven implementation in agentic/etl/transformers/extractors/pdf.py.
Uses fallback strategy: Mistral OCR → OpenDataLoader → PyMuPDF (fitz) → pdfplumber
"""

import asyncio
import base64
import glob as glob_mod
import io
import logging
import os
import re
import tempfile

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
    1. Mistral OCR - Best for scanned PDFs (requires API key)
    2. OpenDataLoader - High-accuracy structural extraction (local, needs Java)
    3. PyMuPDF (fitz) - Fast, good for text-based PDFs
    4. pdfplumber - Fallback for edge cases

    When an explicit cloud OCR method (mistral, paddleocr, lighton) fails,
    extraction falls back to local methods (opendataloader → fitz → pdfplumber).

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
            max_pages: Maximum pages per Mistral OCR API call (batch size)
        """
        self.mistral_api_key = mistral_api_key
        self.paddleocr_api_key = paddleocr_api_key
        self.paddleocr_base_url = paddleocr_base_url
        self.lighton_api_key = lighton_api_key
        self.lighton_base_url = lighton_base_url
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
                        return await self._try_method(method, raw)
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
        # Skip mistral in the chain when no API key is configured
        if not self.mistral_api_key and "mistral" in chain:
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
        )

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
            client = Mistral(api_key=self.mistral_api_key)

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
            LIGHTON_MAX_TOKENS,
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

        page_markdowns = []
        page_text_derivs = []
        for img_deriv in image_derivs:
            b64_img = base64.b64encode(img_deriv.content).decode("utf-8")
            payload = {
                "model": LIGHTON_DEFAULT_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_img}",
                                },
                            }
                        ],
                    }
                ],
                "max_tokens": LIGHTON_MAX_TOKENS,
                "temperature": LIGHTON_TEMPERATURE,
                "top_p": LIGHTON_TOP_P,
            }

            resp = requests.post(url, json=payload, headers=headers, timeout=LIGHTON_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            page_md = data["choices"][0]["message"]["content"]
            page_markdowns.append(page_md)
            page_text_derivs.append(
                Derivative(
                    type="page_text",
                    content=page_md,
                    format="plain",
                    page=img_deriv.page,
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
