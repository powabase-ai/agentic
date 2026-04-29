"""Per-provider artifact extraction at stream-end. Best-effort — exceptions
are logged and swallowed (extraction must never fail the run)."""

from __future__ import annotations

import logging
from typing import Any

import litellm

from agentic.agent.message import (
    AnthropicReasoning,
    GeminiReasoning,
    OpenAIReasoning,
    ReasoningArtifact,
)

logger = logging.getLogger(__name__)


def extract_reasoning_artifact(
    *,
    model: str,
    assembled_message,
    final_response: Any,
    requested_effort: str | None,
) -> ReasoningArtifact | None:
    """Build a ReasoningArtifact from a streaming Message and final-chunk response.

    Returns None if no reasoning artifact applies (no provider match, or all
    provider-specific fields empty).

    Best-effort: any exception is logged and returns None.
    """
    try:
        return _extract_inner(
            model=model,
            assembled_message=assembled_message,
            final_response=final_response,
            requested_effort=requested_effort,
        )
    except Exception:
        logger.warning("artifact extraction failed", exc_info=True)
        return None


def _extract_inner(
    *, model, assembled_message, final_response, requested_effort
) -> ReasoningArtifact | None:
    try:
        _, provider, _, _ = litellm.get_llm_provider(model)
    except Exception:
        return None

    summary = getattr(assembled_message, "reasoning_content", "") or None
    blocks = list(getattr(assembled_message, "thinking_blocks", []) or [])
    psf = dict(getattr(assembled_message, "provider_specific_fields", {}) or {})
    usage = getattr(final_response, "usage", None)

    if provider == "anthropic":
        if not blocks and not summary:
            return None
        return AnthropicReasoning(
            thinking_blocks=blocks,
            summary_text=summary,
            requested_effort=requested_effort,
            output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        )

    if provider == "openai":
        encrypted = psf.get("encrypted_content_items", []) or []
        reasoning_count = None
        if usage is not None:
            details = getattr(usage, "completion_tokens_details", None)
            if details is not None:
                reasoning_count = getattr(details, "reasoning_tokens", None)
        if not encrypted and not summary and not reasoning_count:
            return None
        return OpenAIReasoning(
            response_id=getattr(final_response, "id", None),
            encrypted_content_items=encrypted,
            summary_text=summary,
            requested_effort=requested_effort,
            reasoning_token_count=reasoning_count,
        )

    if provider in ("gemini", "vertex_ai"):
        signatures = psf.get("thought_signatures", []) or []
        thoughts_count = getattr(usage, "thoughts_token_count", None) if usage else None
        if not signatures and not summary and not thoughts_count:
            return None
        return GeminiReasoning(
            thought_signatures=signatures,
            summary_text=summary,
            requested_effort=requested_effort,
            thoughts_token_count=thoughts_count,
        )

    return None
