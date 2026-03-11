"""
Shared utility for splitting text into virtual "pages" at natural boundaries.

Used by TxtExtractor and TextExtractor (for markdown) to produce page_text
derivatives without requiring PDF conversion.
"""

from __future__ import annotations

import re


def split_text_into_pages(
    text: str,
    target_tokens: int = 1500,
    heading_aware: bool = False,
) -> list[str]:
    """Split text into virtual pages of approximately `target_tokens` words.

    Args:
        text: The full text to split.
        target_tokens: Target number of word-tokens per page.
        heading_aware: If True, prefer splitting at markdown heading boundaries
                       (``\\n# ``, ``\\n## ``, etc.) before falling back to
                       paragraph boundaries.

    Returns:
        List of page strings (never empty — returns ``[text]`` for short input
        and ``[""]`` for empty input).
    """
    if not text.strip():
        return [text]

    # --- split into blocks ---
    if heading_aware:
        blocks = _split_by_headings_then_paragraphs(text)
    else:
        blocks = _split_paragraphs(text)

    # --- accumulate blocks into pages ---
    pages: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for block in blocks:
        block_tokens = _estimate_tokens(block)

        # Block fits in current page
        if current_tokens + block_tokens <= target_tokens and current_parts:
            current_parts.append(block)
            current_tokens += block_tokens
            continue

        # Block alone exceeds target — need to split it further
        if block_tokens > target_tokens:
            # Flush current page first
            if current_parts:
                pages.append("\n\n".join(current_parts))
                current_parts = []
                current_tokens = 0
            # Split the oversized block
            sub_pages = _split_large_block(block, target_tokens)
            pages.extend(sub_pages)
            continue

        # Start a new page with this block
        if current_parts:
            pages.append("\n\n".join(current_parts))
        current_parts = [block]
        current_tokens = block_tokens

    if current_parts:
        pages.append("\n\n".join(current_parts))

    return pages if pages else [text]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"(?=(?:^|\n)#{1,6}\s)", re.MULTILINE)


def _split_by_headings_then_paragraphs(text: str) -> list[str]:
    """Split on markdown headings first, then paragraphs within each section."""
    sections = _HEADING_RE.split(text)
    blocks: list[str] = []
    for section in sections:
        # Within each heading section, split on blank lines
        for para in _split_paragraphs(section):
            blocks.append(para)
    return blocks


def _split_paragraphs(text: str) -> list[str]:
    """Split on double-newlines, filtering out empty strings."""
    parts = re.split(r"\n\n+", text)
    return [p.strip() for p in parts if p.strip()]


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_large_block(block: str, target_tokens: int) -> list[str]:
    """Split an oversized block first by sentences, then by words."""
    sentences = _SENTENCE_RE.split(block)

    pages: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        stokens = _estimate_tokens(sentence)

        if stokens > target_tokens:
            # Flush accumulated sentences
            if current_parts:
                pages.append(" ".join(current_parts))
                current_parts = []
                current_tokens = 0
            # Hard-split on word boundaries
            pages.extend(_split_by_words(sentence, target_tokens))
            continue

        if current_tokens + stokens > target_tokens and current_parts:
            pages.append(" ".join(current_parts))
            current_parts = []
            current_tokens = 0

        current_parts.append(sentence)
        current_tokens += stokens

    if current_parts:
        pages.append(" ".join(current_parts))

    return pages


def _split_by_words(text: str, target_tokens: int) -> list[str]:
    """Last-resort split: break on whitespace to fit within target_tokens."""
    words = text.split()
    pages: list[str] = []
    for i in range(0, len(words), target_tokens):
        pages.append(" ".join(words[i : i + target_tokens]))
    return pages


def _estimate_tokens(text: str) -> int:
    """Estimate token count as the number of whitespace-separated words."""
    return len(text.split())
