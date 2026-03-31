"""Error classification for recovery routing in the ReAct loop."""

from __future__ import annotations

_PROMPT_TOO_LONG_PATTERNS = [
    "prompt is too long",
    "context_length_exceeded",
    "maximum context length",
    "413",
    "request too large",
    "too many tokens",
]
_MAX_OUTPUT_PATTERNS = ["max_tokens", "maximum output", "output token limit"]
_AUTH_PATTERNS = [
    "invalid api key",
    "authentication",
    "unauthorized",
    "permission denied",
]


def classify_error(error: Exception | None) -> str | None:
    if error is None:
        return None
    error_type = type(error).__name__
    if error_type == "RateLimitError":
        return "rate_limit"
    if error_type in ("AuthenticationError", "PermissionDeniedError"):
        return "unrecoverable"
    msg = str(error).lower()
    for p in _PROMPT_TOO_LONG_PATTERNS:
        if p in msg:
            return "prompt_too_long"
    for p in _MAX_OUTPUT_PATTERNS:
        if p in msg:
            return "max_output_tokens"
    for p in _AUTH_PATTERNS:
        if p in msg:
            return "unrecoverable"
    if "server error" in msg or "500" in msg or "502" in msg or "503" in msg:
        return "model_error"
    return "unrecoverable"


def classify_finish_reason(finish_reason: str | None) -> str | None:
    if finish_reason in ("length", "max_tokens"):
        return "max_output_tokens"
    return None
