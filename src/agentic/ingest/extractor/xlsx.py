"""
XlsxExtractor - extract content from Excel spreadsheets via MarkItDown.

Converts XLSX/XLS files to markdown tables using MarkItDown.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile

from agentic.ingest.extractor.base import ExtractionError, Extractor
from agentic.ingest.models import Derivative, ExtractionResult, RawContent

logger = logging.getLogger(__name__)


class XlsxExtractor(Extractor):
    """
    Extract content from Excel files using MarkItDown.

    Produces markdown table derivatives from XLSX and XLS files.
    """

    name = "xlsx"
    supported_types = [
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
    ]

    # Map MIME types to file extensions for temp file creation.
    # MarkItDown uses the extension to select the right converter.
    _MIME_TO_EXT = {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "application/vnd.ms-excel": ".xls",
    }

    async def extract(self, raw: RawContent) -> ExtractionResult:
        """
        Extract content from Excel file.

        Args:
            raw: RawContent with XLSX/XLS bytes

        Returns:
            ExtractionResult with markdown table derivatives
        """
        suffix = self._MIME_TO_EXT.get(raw.mime_type, ".xlsx")
        try:
            markdown = await asyncio.to_thread(
                self._convert, raw.content, suffix=suffix
            )
        except Exception as e:
            raise ExtractionError(
                f"MarkItDown XLSX conversion failed: {e}",
                extractor_name=self.name,
                source_uri=raw.source_uri,
            ) from e

        return ExtractionResult(
            source_uri=raw.source_uri,
            mime_type=raw.mime_type,
            derivatives=[
                Derivative(type="markdown", content=markdown, format="markdown"),
                Derivative(type="text", content=markdown, format="plain"),
            ],
            extraction_method="markitdown-xlsx",
            auto_metadata={"char_count": len(markdown)},
        )

    def _convert(self, content: bytes, *, suffix: str = ".xlsx") -> str:
        """Convert Excel bytes to markdown using MarkItDown."""
        from markitdown import MarkItDown

        md = MarkItDown()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(content)
            tmp_path = f.name
        try:
            result = md.convert(tmp_path)
            return result.text_content
        finally:
            os.unlink(tmp_path)
