"""
MarkdownHeaderChunking - split text by Markdown headers, then by length.

This strategy first splits by Markdown header structure (h1, h2, h3, etc.)
using LangChain's ExperimentalMarkdownSyntaxTextSplitter, then further
divides each section by length using RecursiveChunking. Header context
is prepended to each chunk for better embedding quality.

The header splitter is a subclass that keeps all text after an unclosed code
fence; the upstream splitter silently discards it (see
_FenceSafeMarkdownSplitter).

Useful for structured Markdown documents (docs, READMEs, etc.).
"""

from langchain_text_splitters.markdown import ExperimentalMarkdownSyntaxTextSplitter

from agentic.knowledge.chunking.base import ChunkingStrategy
from agentic.knowledge.chunking.recursive import RecursiveChunking
from agentic.knowledge.model_config import (
    CHUNK_EMBED_DEFAULT_CHUNK_SIZE,
    CHUNK_EMBED_DEFAULT_OVERLAP,
)
from agentic.knowledge.models import TextChunk

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


class _FenceSafeMarkdownSplitter(ExperimentalMarkdownSyntaxTextSplitter):
    """ExperimentalMarkdownSyntaxTextSplitter that never drops text at an unclosed fence.

    Upstream ``_resolve_code_chunk`` pops every remaining line looking for a
    closing fence and returns ``""`` when there is none, discarding the rest of
    the document (#46). OCR output often has a stray or unbalanced fence, so
    here an unclosed fence instead puts the consumed lines back and its own line
    is carried into the following text as plain content. If nothing but
    whitespace follows it before the next header, rule or end of input, the
    fence line joins the text directly before it, or is dropped when there is
    none, so it is never emitted as a chunk of its own.

    Written against the private ``_resolve_code_chunk`` / ``_complete_chunk_doc``
    of langchain-text-splitters 1.1.x (the pin in pyproject.toml is capped for
    this reason). Re-check both overrides before raising the cap.
    """

    _pending_fence = ""  # unclosed fence line, to start the next chunk
    _carried_fence = ""  # fence line the current chunk starts with
    _fence_follows_text = False  # chunks[-1] ends directly before the fence
    _emitted_last = False  # the last completion appended a chunk

    def _resolve_code_chunk(self, current_line: str, raw_lines: list[str]) -> str:
        consumed: list[str] = []
        while raw_lines:
            raw_line = raw_lines.pop(0)
            consumed.append(raw_line)
            if self._match_code(raw_line):
                return current_line + "".join(consumed)
        raw_lines[:0] = consumed
        # The caller completed the preceding text just before calling us, then
        # stores the returned text as a code chunk. Returning "" makes that chunk
        # empty, so it is discarded, and the fence line is held until
        # _complete_chunk_doc starts the next chunk with it.
        self._fence_follows_text = self._emitted_last
        self._pending_fence = current_line
        return ""

    def _complete_chunk_doc(self) -> None:
        carried = self._carried_fence
        self._carried_fence = ""
        if carried and self.current_chunk.page_content.strip() == carried.strip():
            if self._fence_follows_text:
                self.chunks[-1].page_content += carried
            self.current_chunk.page_content = ""

        count = len(self.chunks)
        super()._complete_chunk_doc()
        self._emitted_last = len(self.chunks) > count

        if self._pending_fence:
            self.current_chunk.page_content = self._carried_fence = self._pending_fence
            self._pending_fence = ""


class MarkdownHeaderChunking(ChunkingStrategy):
    """
    Chunking strategy that splits by Markdown headers first, then by length.

    This strategy:
    - Splits by Markdown header structure (h1-h6)
    - Further divides sections by length using recursive splitting
    - Prepends header context to each chunk for embedding
    - Keeps all text after an unbalanced code fence, treating the stray fence
      line as plain text; closed code blocks are never split on headers

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
        chunk_size: int = CHUNK_EMBED_DEFAULT_CHUNK_SIZE,
        overlap: int = CHUNK_EMBED_DEFAULT_OVERLAP,
    ):
        if overlap >= chunk_size:
            raise ValueError(
                f"Overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            )

        super().__init__(chunk_size=chunk_size, overlap=overlap)
        self._header_splitter = _FenceSafeMarkdownSplitter()
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
