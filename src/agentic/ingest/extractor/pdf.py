"""
PDFExtractor - extract text from PDF documents with fallback strategy.

Adapted from proven implementation in agentic/etl/transformers/extractors/pdf.py.
Uses fallback strategy: Mistral OCR → PyMuPDF (fitz) → pdfplumber
"""

import asyncio
import io
import logging
import re
import tempfile

from agentic.ingest.extractor.base import (
    ExtractionError,
    Extractor,
    replace_image_annotations,
)
from agentic.ingest.models import Derivative, ExtractionResult, RawContent

logger = logging.getLogger(__name__)


def _count_pages_from_bytes(data: bytes) -> int:
    """Count pages in PDF from bytes."""
    rxcountpages = re.compile(rb"/Type\s*/Page([^s]|$)", re.MULTILINE | re.DOTALL)
    return len(rxcountpages.findall(data))


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
        max_pages: int = 500,
    ):
        """
        Initialize PDF extractor.

        Args:
            mistral_api_key: API key for Mistral OCR (optional, enables OCR)
            max_pages: Maximum pages to process with Mistral OCR (cost protection)
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
                except Exception as e:
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

        fulltext = "\n".join(page_text_strings)
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
        """Extract using Mistral OCR - best for scanned PDFs."""
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

        # Safeguard to avoid excessive API costs
        page_count = _count_pages_from_bytes(raw.content)
        if page_count > self.max_pages:
            raise ExtractionError(
                f"PDF has {page_count} pages, exceeds max {self.max_pages}",
                extractor_name=self.name,
                source_uri=raw.source_uri,
            )

        # Need to write to temp file for Mistral API
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(raw.content)
            temp_path = f.name

        try:
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
                page_text_derivs.append(
                    Derivative(
                        type="page_text",
                        content=page.markdown,
                        format="plain",
                        page=page.index + 1,
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
                extraction_method="mistral_ocr",
                stats={"pages_processed": len(page_markdowns)},
            )
        finally:
            import os

            os.unlink(temp_path)

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
