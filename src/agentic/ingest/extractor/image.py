"""
ImageExtractor - extract text from images using OCR.

Uses Mistral OCR to extract text content from images, while preserving
the original image as a derivative for image-mode retrieval.
"""

import base64
import logging

from agentic.ingest.extractor.base import (
    ExtractionError,
    Extractor,
    replace_image_annotations,
)
from agentic.ingest.models import Derivative, ExtractionResult, RawContent

logger = logging.getLogger(__name__)

# Map MIME types to data URI media types
_MIME_TO_MEDIA = {
    "image/png": "image/png",
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/webp": "image/webp",
    "image/gif": "image/gif",
    "image/tiff": "image/tiff",
}


class ImageExtractor(Extractor):
    """
    Image extraction using OCR.

    Extraction methods (in order of preference):
    1. Mistral OCR - Vision-capable OCR via pixtral (requires API key)

    Produces two derivatives:
    - text/markdown: OCR-extracted text content
    - image: Original image bytes (page=1) for image-mode retrieval

    Example:
        >>> extractor = ImageExtractor(mistral_api_key="sk-...")
        >>> raw = RawContent(content=png_bytes, mime_type="image/png", ...)
        >>> result = await extractor.extract(raw)
        >>> print(result.get_primary_text())
    """

    name = "image"
    supported_types = [
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/webp",
        "image/gif",
        "image/tiff",
    ]

    def __init__(self, mistral_api_key: str | None = None):
        self.mistral_api_key = mistral_api_key

    async def extract(self, raw: RawContent) -> ExtractionResult:
        if self.mistral_api_key:
            return await self._extract_mistral(raw)

        raise ExtractionError(
            "No OCR method configured for image extraction. "
            "Set MISTRAL_API_KEY.",
            extractor_name=self.name,
            source_uri=raw.source_uri,
        )

    async def _extract_mistral(self, raw: RawContent) -> ExtractionResult:
        """Extract text from image using Mistral OCR (pixtral vision)."""
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

        media_type = _MIME_TO_MEDIA.get(raw.mime_type, raw.mime_type)
        b64_data = base64.b64encode(raw.content).decode("ascii")
        data_uri = f"data:{media_type};base64,{b64_data}"

        client = Mistral(api_key=self.mistral_api_key)

        # Define image annotation model
        class ImageType(str, Enum):
            GRAPH = "graph"
            TEXT = "text"
            TABLE = "table"
            IMAGE = "image"

        class Image(BaseModel):
            image_type: ImageType = Field(
                ..., description="The type of the image."
            )
            description: str = Field(
                ..., description="A description of the image."
            )

        from mistralai.extra import response_format_from_pydantic_model

        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "image_url", "image_url": data_uri},
            bbox_annotation_format=response_format_from_pydantic_model(Image),
            include_image_base64=False,
        )

        # Collect OCR text from all pages (typically one for a single image)
        page_markdowns = []
        for page in ocr_response.pages:
            annotations = [
                (img.id, img.image_annotation) for img in page.images
            ]
            page.markdown = replace_image_annotations(
                page.markdown, annotations
            )
            page_markdowns.append(page.markdown)

        ocr_text = "\n\n".join(page_markdowns)

        # Determine image format from MIME type
        fmt = raw.mime_type.split("/")[-1]
        if fmt == "jpeg":
            fmt = "jpg"

        derivatives = [
            # Text derivative
            Derivative(
                type="markdown",
                content=ocr_text,
                format="markdown",
            ),
            # Original image as derivative for image-mode retrieval
            Derivative(
                type="image",
                content=raw.content,
                format=fmt,
                page=1,
                metadata={
                    "width": None,
                    "height": None,
                    "original": True,
                },
            ),
        ]
        # Per-page text derivatives
        for i, page_md in enumerate(page_markdowns):
            derivatives.append(
                Derivative(
                    type="page_text",
                    content=page_md,
                    format="plain",
                    page=i + 1,
                )
            )

        return ExtractionResult(
            source_uri=raw.source_uri,
            mime_type=raw.mime_type,
            derivatives=derivatives,
            auto_metadata={
                "page_count": 1,
                "char_count": len(ocr_text),
            },
            extraction_method="mistral_ocr",
            stats={"pages_processed": 1},
        )
