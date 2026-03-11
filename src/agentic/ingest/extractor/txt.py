"""
TxtExtractor - extract text from plain text files using native page splitting.

Decodes text bytes and splits into virtual pages at natural paragraph
boundaries, producing both a full-text derivative and per-page page_text
derivatives compatible with the indexing pipeline.
"""

from __future__ import annotations

import logging

from agentic.ingest.extractor.base import ExtractionError, Extractor
from agentic.ingest.extractor.page_splitter import split_text_into_pages
from agentic.ingest.models import Derivative, ExtractionResult, RawContent

logger = logging.getLogger(__name__)


class TxtExtractor(Extractor):
    """
    Extract text from plain text files with virtual page splitting.
    """

    name = "txt"
    supported_types = ["text/plain"]

    async def extract(self, raw: RawContent) -> ExtractionResult:
        # Decode bytes
        try:
            text = raw.content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.content.decode("latin-1")
            except Exception as e:
                raise ExtractionError(
                    f"Failed to decode text content: {e}",
                    extractor_name=self.name,
                    source_uri=raw.source_uri,
                    cause=e,
                ) from e

        # Split into virtual pages
        pages = split_text_into_pages(text, target_tokens=1500)
        content = "\n\n".join(pages)

        # Build derivatives: one full text + one page_text per page
        derivatives = [
            Derivative(
                type="text",
                content=content,
                format="plain",
                metadata={
                    "encoding": "utf-8",
                    "char_count": len(content),
                    "line_count": content.count("\n") + 1,
                },
            ),
        ]
        for i, page_text in enumerate(pages, start=1):
            derivatives.append(
                Derivative(
                    type="page_text",
                    content=page_text,
                    format="plain",
                    page=i,
                )
            )

        return ExtractionResult(
            source_uri=raw.source_uri,
            mime_type=raw.mime_type,
            derivatives=derivatives,
            auto_metadata={
                "page_count": len(pages),
                "char_count": len(content),
                "extraction_method": "txt-native",
            },
            extraction_method="txt-native",
            stats={
                "bytes_processed": len(raw.content),
                "pages": len(pages),
            },
        )
