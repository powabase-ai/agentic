"""
MarkdownHeaderChunking - split text by Markdown headers, then by length.

This strategy first splits by Markdown header structure (h1, h2, h3, etc.)
using LangChain's ExperimentalMarkdownSyntaxTextSplitter, then further
divides each section by length using RecursiveChunking. Header context
is prepended to each chunk for better embedding quality.

Useful for structured Markdown documents (docs, READMEs, etc.).
"""

from agentic.knowledge.chunking.base import ChunkingStrategy
from agentic.knowledge.chunking.recursive import RecursiveChunking
from agentic.knowledge.models import TextChunk
from langchain_text_splitters.markdown import ExperimentalMarkdownSyntaxTextSplitter

# Markdown header symbol → metadata key produced by the splitter.
# Defined locally to decouple from langchain internals — the class attribute
# ExperimentalMarkdownSyntaxTextSplitter.DEFAULT_HEADER_KEYS was removed
# in langchain-text-splitters 1.1.1.
_HEADER_MAP = {
    "#": "Header 1",
    "##": "Header 2",
    "###": "Header 3",
    "####": "Header 4",
    "#####": "Header 5",
    "######": "Header 6",
}
_HEADER_VALUES = frozenset(_HEADER_MAP.values())


class MarkdownHeaderChunking(ChunkingStrategy):
    """
    Chunking strategy that splits by Markdown headers first, then by length.

    This strategy:
    - Splits by Markdown header structure (h1-h6)
    - Further divides sections by length using recursive splitting
    - Prepends header context to each chunk for embedding

    Args:
        chunk_size: Maximum tokens per chunk (default: 2000)
        overlap: Number of overlapping tokens between chunks (default: 50)

    Example:
        >>> chunker = MarkdownHeaderChunking(chunk_size=2000, overlap=50)
        >>> chunks = chunker.chunk("# Intro\\n\\nSome text...")
        >>> for chunk in chunks:
        ...     print(f"Chunk {chunk.index}: {chunk.text[:50]}...")
    """

    name = "markdown_header"

    def __init__(
        self,
        chunk_size: int = 2000,
        overlap: int = 50,
    ):
        if overlap >= chunk_size:
            raise ValueError(
                f"Overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            )

        super().__init__(chunk_size=chunk_size, overlap=overlap)
        self._header_splitter = ExperimentalMarkdownSyntaxTextSplitter()
        self._length_splitter = RecursiveChunking(
            chunk_size=chunk_size,
            overlap=overlap,
        )

    def chunk(
        self,
        text: str,
        source_id: str | None = None,
    ) -> list[TextChunk]:
        """
        Split text into chunks by Markdown headers and length.

        Args:
            text: The Markdown text content to chunk
            source_id: Optional source identifier for chunk metadata

        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []

        # Split by Markdown headers
        sections = self._header_splitter.split_text(text)

        chunks: list[TextChunk] = []
        chunk_index = 0
        search_start = 0

        for section in sections:
            section_content = section.page_content.strip()
            if not section_content:
                continue

            # Find this section's offset in the original text so sub-chunk
            # offsets can be converted from section-relative to document-relative.
            section_offset = text.find(section_content, search_start)
            if section_offset < 0:
                section_offset = text.find(section_content)  # fallback
            if section_offset >= 0:
                search_start = section_offset + len(section_content)
            else:
                section_offset = 0  # last resort

            # Further divide by length using RecursiveChunking
            sub_chunks = self._length_splitter.chunk(
                section_content,
                source_id=source_id,
            )

            for sub_chunk in sub_chunks:
                chunk_text = sub_chunk.text

                # Prepend header to chunk text for embedding context
                header_keys = sorted(
                    k
                    for k in section.metadata.keys()
                    if k in _HEADER_VALUES
                )
                if header_keys:
                    key = header_keys[-1]  # Most specific (deepest) header
                    markdown_symbol = next(
                        sym for sym, val in _HEADER_MAP.items() if val == key
                    )
                    header_text = section.metadata[key].strip()
                    markdown_header = f"{markdown_symbol} {header_text}"
                    chunk_text = "\n".join([markdown_header, chunk_text])

                # Merge metadata
                metadata = dict(sub_chunk.metadata)
                metadata["strategy"] = self.name
                metadata.update(section.metadata)

                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        index=chunk_index,
                        start_char=section_offset + sub_chunk.start_char,
                        end_char=section_offset + sub_chunk.end_char,
                        source_id=source_id,
                        metadata=metadata,
                    )
                )
                chunk_index += 1

        return chunks
