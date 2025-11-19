"""Text file extractor"""

from .base import BaseExtractor, ExtractionResult
from ...registry import ExtractorRegistry


@ExtractorRegistry.register("text")
class TextExtractor(BaseExtractor):
    """Extract text from plain text files"""
    
    def __init__(self, **kwargs):
        pass
    
    def extract(self, file_path: str) -> ExtractionResult:
        """Extract text from plain text file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 if UTF-8 fails
            with open(file_path, "r", encoding="latin-1") as f:
                text = f.read()
        
        return ExtractionResult(
            text=text,
            metadata={"extraction_method": "direct_read"}
        )

