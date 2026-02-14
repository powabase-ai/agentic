"""Token counting and token-character position mapping utilities."""

import tiktoken

_enc: tiktoken.Encoding | None = None


def get_encoding() -> tiktoken.Encoding:
    """Get cached cl100k_base encoding (used by text-embedding-3-small)."""
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("cl100k_base")
    return _enc


def count_tokens(text: str) -> int:
    """Count tokens in text using cl100k_base encoding."""
    if not text:
        return 0
    return len(get_encoding().encode(text))


def build_token_char_map(text: str) -> list[int]:
    """Build mapping: token position -> character offset.

    Returns list where map[i] = character offset after first i tokens.
    map[0] = 0, map[len(tokens)] = len(text).
    """
    enc = get_encoding()
    tokens = enc.encode(text)
    char_map = [0]
    running = 0
    for tok in tokens:
        running += len(enc.decode([tok]))
        char_map.append(running)
    return char_map


def char_to_token_pos(char_map: list[int], char_pos: int) -> int:
    """Find largest token index i where char_map[i] <= char_pos (binary search)."""
    lo, hi = 0, len(char_map) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if char_map[mid] <= char_pos:
            lo = mid
        else:
            hi = mid - 1
    return lo
