"""
PDFExtractor - extract text from PDF documents with fallback strategy.

Adapted from proven implementation in agentic/etl/transformers/extractors/pdf.py.
Uses fallback strategy: Mistral OCR → PyMuPDF (fitz) → pdfplumber
"""

import asyncio
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
    2. PyMuPDF (fitz) - Fast, good for text-based PDFs
    3. pdfplumber - Fallback for edge cases

    Adapted from proven insurance-demo implementation.

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
        max_pages: int = 1000,
    ):
        """
        Initialize PDF extractor.

        Args:
            mistral_api_key: API key for Mistral OCR (optional, enables OCR)
            max_pages: Maximum pages per Mistral OCR API call (batch size)
        """
        self.mistral_api_key = mistral_api_key
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

    async def extract(self, raw: RawContent) -> ExtractionResult:
        """
        Extract text from PDF with fallback strategy.

        Args:
            raw: RawContent with PDF bytes

        Returns:
            ExtractionResult with text derivative
        """
        # Try Mistral OCR first (if configured) with retry
        if self.mistral_api_key:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    return await self._extract_mistral(raw)
                except ExtractionError as e:
                    # Deterministic errors (import failure, etc.) — don't retry
                    logger.warning(
                        f"Mistral OCR failed (deterministic): {e}, "
                        f"falling back to Fitz"
                    )
                    break
                except Exception as e:
                    # Transient errors (network, rate limit) — retry with backoff
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt  # 1s, 2s
                        logger.warning(
                            f"Mistral OCR attempt {attempt + 1}/{max_retries} failed: {e}, "
                            f"retrying in {wait}s"
                        )
                        await asyncio.sleep(wait)
                    else:
                        logger.warning(
                            f"Mistral OCR failed after {max_retries} attempts: {e}, "
                            f"falling back to Fitz"
                        )

        # Fallback to Fitz (PyMuPDF)
        try:
            return self._extract_fitz(raw)
        except Exception as e:
            logger.warning(f"Fitz failed: {e}, falling back to pdfplumber")

        # Final fallback to pdfplumber
        return self._extract_pdfplumber(raw)

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

            from mistralai import Mistral
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
