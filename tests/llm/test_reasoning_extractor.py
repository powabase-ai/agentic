"""Tests for per-provider artifact extraction at stream-end."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from agentic.agent.message import (
    AnthropicReasoning,
    GeminiReasoning,
    OpenAIReasoning,
)
from agentic.llm.reasoning_extractor import extract_reasoning_artifact
from agentic.llm.streaming import Message


def _msg(**kwargs):
    return Message(**kwargs)


def _final_response(**kwargs):
    return SimpleNamespace(**kwargs)


def test_anthropic_extracts_thinking_blocks_and_summary():
    msg = _msg(
        thinking_blocks=[{"type": "thinking", "thinking": "x", "signature": "s"}],
        reasoning_content="summary text",
    )
    final = _final_response(usage=SimpleNamespace(completion_tokens=500), id="r1")
    with patch(
        "agentic.llm.reasoning_extractor.litellm.get_llm_provider",
        return_value=("claude-opus-4-7", "anthropic", None, None),
    ):
        artifact = extract_reasoning_artifact(
            model="anthropic/claude-opus-4-7",
            assembled_message=msg,
            final_response=final,
            requested_effort="medium",
        )
    assert isinstance(artifact, AnthropicReasoning)
    assert artifact.thinking_blocks == [
        {"type": "thinking", "thinking": "x", "signature": "s"}
    ]
    assert artifact.summary_text == "summary text"
    assert artifact.requested_effort == "medium"
    assert artifact.output_tokens == 500


def test_anthropic_returns_none_when_no_blocks_no_summary():
    msg = _msg(thinking_blocks=[], reasoning_content="")
    final = _final_response(usage=None, id="r1")
    with patch(
        "agentic.llm.reasoning_extractor.litellm.get_llm_provider",
        return_value=("x", "anthropic", None, None),
    ):
        artifact = extract_reasoning_artifact(
            model="anthropic/x",
            assembled_message=msg,
            final_response=final,
            requested_effort="medium",
        )
    assert artifact is None


def test_openai_extracts_encrypted_items_and_summary():
    msg = _msg(
        provider_specific_fields={
            "encrypted_content_items": [{"type": "reasoning", "encrypted_content": "e"}]
        },
        reasoning_content="openai summary",
    )
    final = _final_response(
        usage=SimpleNamespace(
            completion_tokens_details=SimpleNamespace(reasoning_tokens=300)
        ),
        id="resp_id",
    )
    with patch(
        "agentic.llm.reasoning_extractor.litellm.get_llm_provider",
        return_value=("gpt-5.4", "openai", None, None),
    ):
        artifact = extract_reasoning_artifact(
            model="openai/responses/gpt-5.4",
            assembled_message=msg,
            final_response=final,
            requested_effort="high",
        )
    assert isinstance(artifact, OpenAIReasoning)
    assert artifact.response_id == "resp_id"
    assert artifact.encrypted_content_items == [
        {"type": "reasoning", "encrypted_content": "e"}
    ]
    assert artifact.summary_text == "openai summary"
    assert artifact.requested_effort == "high"
    assert artifact.reasoning_token_count == 300


def test_gemini_extracts_signatures_and_count():
    msg = _msg(
        provider_specific_fields={"thought_signatures": ["sig1", "sig2"]},
        reasoning_content="gemini summary",
    )
    final = _final_response(usage=SimpleNamespace(thoughts_token_count=150))
    with patch(
        "agentic.llm.reasoning_extractor.litellm.get_llm_provider",
        return_value=("gemini-3-pro", "gemini", None, None),
    ):
        artifact = extract_reasoning_artifact(
            model="gemini/gemini-3-pro",
            assembled_message=msg,
            final_response=final,
            requested_effort="medium",
        )
    assert isinstance(artifact, GeminiReasoning)
    assert artifact.thought_signatures == ["sig1", "sig2"]
    assert artifact.thoughts_token_count == 150


def test_unknown_provider_returns_none():
    msg = _msg(reasoning_content="x")
    final = _final_response(usage=None)
    with patch(
        "agentic.llm.reasoning_extractor.litellm.get_llm_provider",
        return_value=("custom", "unknown_provider", None, None),
    ):
        assert (
            extract_reasoning_artifact(
                model="custom/exotic",
                assembled_message=msg,
                final_response=final,
                requested_effort="low",
            )
            is None
        )


def test_extractor_swallows_exceptions_returns_none(caplog):
    """Best-effort: exceptions during extraction never fail the run."""
    msg = _msg()
    final = _final_response()
    with patch(
        "agentic.llm.reasoning_extractor.litellm.get_llm_provider",
        side_effect=RuntimeError("boom"),
    ):
        assert (
            extract_reasoning_artifact(
                model="x",
                assembled_message=msg,
                final_response=final,
                requested_effort=None,
            )
            is None
        )
