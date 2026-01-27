"""
PDFExtractor - extract text from PDF documents with fallback strategy.

Adapted from proven implementation in agentic/etl/transformers/extractors/pdf.py.
Uses fallback strategy: Mistral OCR → PyMuPDF (fitz) → pdfplumber
"""

import io
import logging
import re
import tempfile

from agentic.ingest.extractor.base import ExtractionError, Extractor
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

    async def extract(self, raw: RawContent) -> ExtractionResult:
        """
        Extract text from PDF with fallback strategy.

        Args:
            raw: RawContent with PDF bytes

        Returns:
            ExtractionResult with text derivative
        """
        # Try Mistral OCR first (if configured)
        if self.mistral_api_key:
            try:
                return await self._extract_mistral(raw)
            except Exception as e:
                logger.warning(f"Mistral OCR failed: {e}, falling back to Fitz")

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
        page_texts = []
        page_metas = []

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
            page_texts.append("\n".join(block_texts))

            # Prefer logical page numbers; fall back to physical
            try:
                page_number = page.get_label()
            except Exception:
                page_number = page.number + 1
            page_metas.append({"page": page_number})

        fulltext = "\n".join(page_texts)
        doc.close()

        return ExtractionResult(
            source_uri=raw.source_uri,
            mime_type=raw.mime_type,
            derivatives=[
                Derivative(
                    type="text",
                    content=fulltext,
                    format="plain",
                    metadata={"page_texts": page_texts, "page_metas": page_metas},
                ),
            ],
            auto_metadata={
                "page_count": len(page_texts),
                "char_count": len(fulltext),
            },
            extraction_method="fitz",
            stats={"pages_processed": len(page_texts)},
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
            page_metas = []
            for page in ocr_response.pages:
                for img in page.images:
                    page.markdown = page.markdown.replace(
                        f"![{img.id}]({img.id})",
                        f"![{img.id}]\n**{img.image_annotation}**",
                    )
                page_markdowns.append(page.markdown)
                page_metas.append(
                    {
                        "page_number": page.index,
                        "dimensions": {
                            "width_px": page.dimensions.width
                            if page.dimensions
                            else None,
                            "height_px": page.dimensions.height
                            if page.dimensions
                            else None,
                            "dpi": page.dimensions.dpi if page.dimensions else None,
                        },
                    }
                )

            fulltext = "\n\n".join(page_markdowns)

            return ExtractionResult(
                source_uri=raw.source_uri,
                mime_type=raw.mime_type,
                derivatives=[
                    Derivative(
                        type="markdown",
                        content=fulltext,
                        format="markdown",
                        metadata={
                            "page_texts": page_markdowns,
                            "page_metas": page_metas,
                        },
                    ),
                ],
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
        page_texts = []
        page_metas = []

        with pdfplumber.open(io.BytesIO(raw.content)) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
                page_texts.append(page_text)
                page_metas.append({"page": i + 1})

        fulltext = "\n\n".join(text_parts)

        return ExtractionResult(
            source_uri=raw.source_uri,
            mime_type=raw.mime_type,
            derivatives=[
                Derivative(
                    type="text",
                    content=fulltext,
                    format="plain",
                    metadata={"page_texts": page_texts, "page_metas": page_metas},
                ),
            ],
            auto_metadata={
                "page_count": len(page_texts),
                "char_count": len(fulltext),
            },
            extraction_method="pdfplumber",
            stats={"pages_processed": len(page_texts)},
        )
