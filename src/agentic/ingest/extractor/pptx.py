"""
PptxExtractor - extract text from PowerPoint presentations via PDF conversion.

Converts PPTX to PDF using LibreOffice headless, then delegates to
PDFExtractor for page-level extraction with full derivative support.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from agentic.ingest.extractor.base import ExtractionError, Extractor
from agentic.ingest.models import ExtractionResult, RawContent

if TYPE_CHECKING:
    from agentic.ingest.extractor.pdf import PDFExtractor

logger = logging.getLogger(__name__)


class PptxExtractor(Extractor):
    """
    Extract text from PPTX files by converting to PDF via LibreOffice headless,
    then delegating to PDFExtractor for page-level extraction.
    """

    name = "pptx"
    supported_types = [
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ]

    def __init__(self, pdf_extractor: PDFExtractor | None = None):
        self._pdf_extractor = pdf_extractor

    async def extract(self, raw: RawContent) -> ExtractionResult:
        """
        Extract text from PPTX content by converting to PDF first.

        Args:
            raw: RawContent with PPTX bytes

        Returns:
            ExtractionResult with page-level derivatives from PDF pipeline
        """
        # 1. Convert PPTX → PDF via LibreOffice headless
        pdf_bytes = await asyncio.to_thread(
            self._convert_to_pdf, raw.content, source_uri=raw.source_uri
        )

        # 2. Wrap as PDF RawContent with corrected .pdf filename
        pdf_filename = raw.filename
        if pdf_filename:
            pdf_filename = Path(pdf_filename).stem + ".pdf"
        pdf_raw = RawContent(
            content=pdf_bytes,
            mime_type="application/pdf",
            source_uri=raw.source_uri,
            filename=pdf_filename,
        )

        # 3. Delegate to PDF extractor
        pdf_extractor = self._pdf_extractor
        if pdf_extractor is None:
            from agentic.ingest.extractor.pdf import PDFExtractor

            pdf_extractor = PDFExtractor()

        result = await pdf_extractor.extract(pdf_raw)

        # 4. Tag the origin, preserving underlying PDF method for separator logic
        underlying_method = result.extraction_method
        result.extraction_method = "pptx-via-pdf"
        result.mime_type = raw.mime_type
        if result.auto_metadata is None:
            result.auto_metadata = {}
        result.auto_metadata["pdf_extraction_method"] = underlying_method
        return result

    def _convert_to_pdf(
        self, pptx_bytes: bytes, source_uri: str | None = None
    ) -> bytes:
        """Convert PPTX bytes to PDF using LibreOffice headless."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.pptx")
            with open(input_path, "wb") as f:
                f.write(pptx_bytes)

            try:
                proc = subprocess.run(
                    [
                        "libreoffice",
                        "--headless",
                        "--convert-to",
                        "pdf",
                        "--outdir",
                        tmpdir,
                        f"-env:UserInstallation=file://{tmpdir}/profile",
                        input_path,
                    ],
                    capture_output=True,
                    timeout=120,
                )
            except FileNotFoundError as e:
                raise ExtractionError(
                    "LibreOffice is not installed or not in PATH",
                    extractor_name=self.name,
                    source_uri=source_uri,
                ) from e
            except subprocess.TimeoutExpired as e:
                raise ExtractionError(
                    "LibreOffice PPTX→PDF conversion timed out after 120s",
                    extractor_name=self.name,
                    source_uri=source_uri,
                ) from e

            if proc.returncode != 0:
                stderr = proc.stderr.decode("utf-8", errors="replace")
                raise ExtractionError(
                    f"LibreOffice PPTX→PDF conversion failed (rc={proc.returncode}): {stderr}",
                    extractor_name=self.name,
                    source_uri=source_uri,
                )

            pdf_path = os.path.join(tmpdir, "input.pdf")
            if not os.path.exists(pdf_path):
                raise ExtractionError(
                    "LibreOffice did not produce expected PDF output",
                    extractor_name=self.name,
                    source_uri=source_uri,
                )

            with open(pdf_path, "rb") as f:
                return f.read()
