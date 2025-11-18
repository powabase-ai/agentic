"""DOCX extractor"""

import pypandoc
from .base import BaseExtractor, ExtractionResult
from ...registry import ExtractorRegistry


@ExtractorRegistry.register("docx")
class DocxExtractor(BaseExtractor):
    """Extract text from DOCX files, converting to markdown"""
    
    def __init__(self, **kwargs):
        pass
    
    def extract(self, file_path: str) -> ExtractionResult:
        """Extract text from DOCX file, converting to markdown"""
        with open(file_path, "rb") as f:
            markdown_text = pypandoc.convert_text(f.read(), "md", format="docx")
        
        return ExtractionResult(
            text=markdown_text,
            metadata={"extraction_method": "pypandoc", "format": "markdown"}
        )

