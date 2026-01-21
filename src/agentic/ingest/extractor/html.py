"""
HTMLExtractor - extract text from HTML documents.

Uses BeautifulSoup for parsing and text extraction.
"""

import logging
import re

from agentic.ingest.extractor.base import Extractor, ExtractionError
from agentic.ingest.models import RawContent, ExtractionResult, Derivative

logger = logging.getLogger(__name__)


class HTMLExtractor(Extractor):
    """
    Extractor for HTML documents.
    
    Uses BeautifulSoup to parse HTML and extract text content,
    stripping tags while preserving structure.
    
    Produces:
        - "text" derivative with cleaned text
        - Optionally "markdown" derivative (future)
    
    Example:
        >>> extractor = HTMLExtractor()
        >>> raw = RawContent(content=html_bytes, mime_type="text/html", ...)
        >>> result = await extractor.extract(raw)
        >>> print(result.get_primary_text())
    """
    
    name = "html"
    supported_types = [
        "text/html",
        "application/xhtml+xml",
    ]
    
    # Tags to completely remove (including content)
    REMOVE_TAGS = ["script", "style", "head", "nav", "footer", "aside", "noscript"]
    
    # Tags that indicate structure breaks (add newlines)
    BLOCK_TAGS = [
        "p", "div", "section", "article", "main", "header",
        "h1", "h2", "h3", "h4", "h5", "h6",
        "ul", "ol", "li", "br", "hr", "blockquote", "pre",
        "table", "tr", "td", "th",
    ]
    
    def __init__(self, include_links: bool = False, include_alt_text: bool = True):
        """
        Initialize HTML extractor.
        
        Args:
            include_links: Include URL text in parentheses after links
            include_alt_text: Include alt text for images
        """
        self.include_links = include_links
        self.include_alt_text = include_alt_text
    
    async def extract(self, raw: RawContent) -> ExtractionResult:
        """
        Extract text from HTML content.
        
        Args:
            raw: RawContent with HTML bytes
        
        Returns:
            ExtractionResult with text derivative
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ExtractionError(
                "BeautifulSoup is required for HTML extraction. "
                "Install with: pip install beautifulsoup4",
                extractor_name=self.name,
                source_uri=raw.source_uri,
            )
        
        try:
            # Decode content
            try:
                html = raw.content.decode("utf-8")
            except UnicodeDecodeError:
                html = raw.content.decode("latin-1")
            
            # Parse HTML
            soup = BeautifulSoup(html, "html.parser")
            
            # Extract title
            title = None
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)
            
            # Remove unwanted tags
            for tag in self.REMOVE_TAGS:
                for element in soup.find_all(tag):
                    element.decompose()
            
            # Extract text with structure preservation
            text = self._extract_text(soup)
            
            # Clean up whitespace
            text = self._clean_text(text)
            
            # Build metadata
            auto_metadata = {
                "char_count": len(text),
                "line_count": text.count("\n") + 1,
            }
            if title:
                auto_metadata["title"] = title
            
            # Try to extract description meta tag
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc and meta_desc.get("content"):
                auto_metadata["description"] = meta_desc["content"]
            
            return ExtractionResult(
                source_uri=raw.source_uri,
                mime_type=raw.mime_type,
                derivatives=[
                    Derivative(
                        type="text",
                        content=text,
                        format="plain",
                        metadata={
                            "original_encoding": "utf-8",
                        },
                    ),
                ],
                auto_metadata=auto_metadata,
                extraction_method=self.name,
                stats={
                    "bytes_processed": len(raw.content),
                },
            )
            
        except Exception as e:
            raise ExtractionError(
                f"Failed to extract HTML: {e}",
                extractor_name=self.name,
                source_uri=raw.source_uri,
                cause=e,
            )
    
    def _extract_text(self, soup) -> str:
        """
        Extract text from BeautifulSoup object with structure.
        """
        parts = []
        
        # Find main content or use body
        main = soup.find("main") or soup.find("article") or soup.find("body") or soup
        
        for element in main.descendants:
            if element.name in self.BLOCK_TAGS:
                parts.append("\n")
            
            if hasattr(element, "name") and element.name is None:
                # Text node
                text = str(element).strip()
                if text:
                    parts.append(text + " ")
            
            # Handle images with alt text
            if self.include_alt_text and hasattr(element, "name") and element.name == "img":
                alt = element.get("alt", "")
                if alt:
                    parts.append(f"[Image: {alt}] ")
            
            # Handle links
            if self.include_links and hasattr(element, "name") and element.name == "a":
                href = element.get("href", "")
                if href and href.startswith("http"):
                    # Will be added after link text
                    pass
        
        return "".join(parts)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean up extracted text.
        """
        # Replace multiple spaces with single space
        text = re.sub(r" +", " ", text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        
        # Strip leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)
        
        # Strip overall
        return text.strip()
