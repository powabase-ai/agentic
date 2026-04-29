"""Tests for defensive extraction patterns (LiteLLM bugs from research doc)."""

from types import SimpleNamespace

from agentic.llm.streaming import _extract_reasoning_from_delta


def test_prefers_reasoning_content_field():
    delta = SimpleNamespace(reasoning_content="proper", reasoning=None, content="")
    assert _extract_reasoning_from_delta(delta) == "proper"


def test_falls_back_to_reasoning_field():
    """LiteLLM #21386 — some providers use `reasoning` instead of `reasoning_content`."""
    delta = SimpleNamespace(reasoning_content=None, reasoning="alt", content="")
    assert _extract_reasoning_from_delta(delta) == "alt"


def test_extracts_think_tag_from_content():
    """LiteLLM #26326 — Fireworks AI leaks <think> tags into content."""
    delta = SimpleNamespace(
        reasoning_content=None,
        reasoning=None,
        content="<think>thought text</think>",
    )
    result = _extract_reasoning_from_delta(delta)
    assert result == "thought text"


def test_extracts_thought_prefix():
    """google-genai #2121 — Gemini 2.5 prefixes thought content with THOUGHT:."""
    delta = SimpleNamespace(
        reasoning_content=None,
        reasoning=None,
        content="THOUGHT: I should consider X",
    )
    result = _extract_reasoning_from_delta(delta)
    assert result == "I should consider X"


def test_returns_none_for_normal_content():
    delta = SimpleNamespace(reasoning_content=None, reasoning=None, content="just text")
    assert _extract_reasoning_from_delta(delta) is None


def test_returns_none_when_all_empty():
    delta = SimpleNamespace(reasoning_content=None, reasoning=None, content=None)
    assert _extract_reasoning_from_delta(delta) is None
