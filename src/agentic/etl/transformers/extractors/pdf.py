"""PDF extractor with fallback strategy"""

import logging
import re
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber
from mistralai import Mistral
from mistralai.models import OCRResponse
from mistralai.extra import response_format_from_pydantic_model
from pydantic import BaseModel, Field
from enum import Enum

from .base import BaseExtractor, ExtractionResult
from ...registry import ExtractorRegistry

logger = logging.getLogger(__name__)


# Mistral OCR types (from reference)
class ImageType(str, Enum):
    GRAPH = "graph"
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class Image(BaseModel):
    image_type: ImageType = Field(..., description="The type of the image.")
    description: str = Field(..., description="A description of the image.")


def _count_pages(filename: str) -> int:
    """Count pages in PDF (from reference)"""
    rxcountpages = re.compile(rb"/Type\s*/Page([^s]|$)", re.MULTILINE | re.DOTALL)
    with open(filename, "rb") as infile:
        data = infile.read()
    return len(rxcountpages.findall(data))


def _get_pages_with_image_annotations(ocr_response: OCRResponse):
    """Add image annotations to markdown (from reference)"""
    page_markdowns = []
    page_metas = []
    for page in ocr_response.pages:
        for img in page.images:
            page.markdown = page.markdown.replace(
                f"![{img.id}]({img.id})", f"![{img.id}]\n**{img.image_annotation}**"
            )
        page_markdowns.append(page.markdown)
        page_metas.append({
            "page_number": page.index,
            "dimensions": {
                "width_px": page.dimensions.width if page.dimensions else None,
                "height_px": page.dimensions.height if page.dimensions else None,
                "dpi": page.dimensions.dpi if page.dimensions else None,
            }
        })
    return page_markdowns, page_metas


@ExtractorRegistry.register("pdf")
class PDFExtractor(BaseExtractor):
    """PDF extraction with fallback strategy"""
    
    def __init__(self, mistral_api_key: Optional[str] = None, max_pages: int = 500, **kwargs):
        self.mistral_api_key = mistral_api_key
        self.max_pages = max_pages
    
    def extract(self, file_path: str) -> ExtractionResult:
        """Extract text from PDF with fallback strategy"""
        # Try Mistral OCR first (if configured)
        if self.mistral_api_key:
            try:
                return self._extract_mistral(file_path)
            except Exception as e:
                logger.warning(f"Mistral OCR failed: {e}, falling back to Fitz")
        
        # Fallback to Fitz (PyMuPDF)
        try:
            return self._extract_fitz(file_path)
        except Exception as e:
            logger.warning(f"Fitz failed: {e}, falling back to pdfplumber")
        
        # Final fallback to pdfplumber
        return self._extract_pdfplumber(file_path)
    
    def _extract_fitz(self, file_path: str) -> ExtractionResult:
        """Extract using PyMuPDF (Fitz) - from reference fitz_extractor.py"""
        doc = fitz.open(file_path)
        page_texts = []
        page_metas = []
        
        for page in doc:
            blocks = page.get_text("blocks")
            block_texts = []
            for block in blocks:
                x0, y0, x1, y1, lines, block_no, block_type = block
                if block_type == 0:  # Text block
                    block_text = lines.replace("\n", " ").strip()
                    block_texts.append(block_text)
            page_texts.append("\n".join(block_texts))
            
            # Prefer logical page numbers; fall back to physical
            try:
                page_number = page.get_label()
            except:
                page_number = page.number + 1
            page_metas.append({"page": page_number})
        
        fulltext = "\n".join(page_texts)
        doc.close()
        
        return ExtractionResult(
            text=fulltext,
            metadata={"page_texts": page_texts, "page_metas": page_metas, "extraction_method": "fitz"}
        )
    
    def _extract_mistral(self, file_path: str) -> ExtractionResult:
        """Extract using Mistral OCR - from reference mistral_extractor.py"""
        # Safeguard to avoid excessive API costs
        if _count_pages(file_path) > self.max_pages:
            raise ValueError(
                f"File {file_path} has {_count_pages(file_path)} pages, "
                f"which exceeds the maximum of {self.max_pages} pages."
            )
        
        client = Mistral(api_key=self.mistral_api_key)
        
        with open(file_path, "rb") as f:
            uploaded_file = client.files.upload(
                file={"file_name": file_path, "content": f},
                purpose="ocr"
            )
        
        file_url = client.files.get_signed_url(file_id=uploaded_file.id)
        
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "document_url", "document_url": file_url.url},
            bbox_annotation_format=response_format_from_pydantic_model(Image),
            include_image_base64=False
        )
        
        page_markdowns, page_metas = _get_pages_with_image_annotations(ocr_response)
        
        return ExtractionResult(
            text="\n\n".join(page_markdowns),
            metadata={"page_texts": page_markdowns, "page_metas": page_metas, "extraction_method": "mistral_ocr"}
        )
    
    def _extract_pdfplumber(self, file_path: str) -> ExtractionResult:
        """Extract using pdfplumber as final fallback"""
        text_parts = []
        page_texts = []
        page_metas = []
        
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
                page_texts.append(page_text)
                page_metas.append({"page": i + 1})
        
        return ExtractionResult(
            text="\n\n".join(text_parts),
            metadata={"page_texts": page_texts, "page_metas": page_metas, "extraction_method": "pdfplumber"}
        )

