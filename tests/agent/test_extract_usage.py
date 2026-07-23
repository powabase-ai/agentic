"""Tests for `Agent._extract_usage` — normalizes LiteLLM's usage object
across Chat Completions / Responses API shapes and Pydantic / dict variants.

These cover the non-trivial branches in agent.py:1428-1481:
  - dict vs Pydantic-like shape (the helper does both)
  - Chat Completions naming (prompt/completion + *_details)
  - Responses API naming (input/output + *_details)
  - reasoning_tokens / cached_tokens nested under either family
  - "drop-None" behaviour that distinguishes a true zero from missing
"""

from __future__ import annotations

import types

from agentic.agent.agent import Agent


def _resp(usage):
    """Wrap a usage payload in a minimal stand-in for a litellm response."""
    return types.SimpleNamespace(usage=usage)


# ---------------------------------------------------------------------------
# Empty / missing
# ---------------------------------------------------------------------------


def test_returns_none_when_response_has_no_usage():
    assert Agent._extract_usage(_resp(None)) is None


def test_returns_none_when_usage_is_empty_dict():
    # `if not response.usage` — empty dict is falsy → None. Same as the
    # missing case; downstream uses `if usage:` to gate the persist call.
    assert Agent._extract_usage(_resp({})) is None


def test_drops_keys_when_top_level_fields_are_none():
    # When `usage` is non-empty but every key is None (some bridges do
    # this), the function returns an empty dict per the drop-None
    # contract — not None. Lets the caller distinguish "no usage at all"
    # from "usage existed but was empty."
    usage = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        # An unrelated key keeps the dict truthy at the top of the function.
        "model": "fake",
    }
    assert Agent._extract_usage(_resp(usage)) == {}


# ---------------------------------------------------------------------------
# Chat Completions naming (OpenAI default + most providers via litellm)
# ---------------------------------------------------------------------------


def test_chat_completions_dict_shape():
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "completion_tokens_details": {"reasoning_tokens": 30},
        "prompt_tokens_details": {"cached_tokens": 80},
    }
    out = Agent._extract_usage(_resp(usage))
    assert out == {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "reasoning_tokens": 30,
        "cached_tokens": 80,
    }


def test_chat_completions_pydantic_shape():
    # Pydantic-like: attribute access, not dict access.
    details = types.SimpleNamespace(reasoning_tokens=30)
    prompt_details = types.SimpleNamespace(cached_tokens=80)
    usage = types.SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        completion_tokens_details=details,
        prompt_tokens_details=prompt_details,
    )
    out = Agent._extract_usage(_resp(usage))
    assert out["reasoning_tokens"] == 30
    assert out["cached_tokens"] == 80


# ---------------------------------------------------------------------------
# Responses API naming (OpenAI Responses; some Anthropic/Gemini bridges)
# ---------------------------------------------------------------------------


def test_responses_api_falls_back_to_input_output_token_names():
    # Provider only sets input/output_tokens; the helper must remap them
    # to prompt/completion_tokens because the rest of the codebase reads
    # those keys.
    usage = {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    }
    out = Agent._extract_usage(_resp(usage))
    assert out == {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }


def test_responses_api_reasoning_tokens_under_output_tokens_details():
    usage = {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "output_tokens_details": {"reasoning_tokens": 30},
        "input_tokens_details": {"cached_tokens": 80},
    }
    out = Agent._extract_usage(_resp(usage))
    assert out["reasoning_tokens"] == 30
    assert out["cached_tokens"] == 80


# ---------------------------------------------------------------------------
# Drop-None contract — distinguishes "0" (real zero) from "missing"
# ---------------------------------------------------------------------------


def test_zero_reasoning_tokens_is_kept():
    # 0 is a real value (provider explicitly told us no reasoning) — must
    # NOT be dropped, otherwise the dashboard can't tell zero from
    # "provider didn't report".
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "completion_tokens_details": {"reasoning_tokens": 0},
    }
    out = Agent._extract_usage(_resp(usage))
    assert out["reasoning_tokens"] == 0


def test_missing_reasoning_tokens_is_omitted():
    # No reasoning_tokens anywhere — the key should be absent from the
    # returned dict so a downstream `usage.get("reasoning_tokens")` is None,
    # not 0.
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }
    out = Agent._extract_usage(_resp(usage))
    assert "reasoning_tokens" not in out


def test_chat_completions_takes_precedence_over_responses_when_both_set():
    # If a wonky provider sets both family names, the Chat Completions
    # name wins (loop iterates Chat first then Responses).
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "completion_tokens_details": {"reasoning_tokens": 30},
        "output_tokens_details": {"reasoning_tokens": 999},
    }
    out = Agent._extract_usage(_resp(usage))
    assert out["reasoning_tokens"] == 30


def test_handles_none_details_object_gracefully():
    # Some providers stub the *_details key to None instead of an object
    # (e.g., proxies that strip nested fields). The _get helper short-
    # circuits on None — without that guard, getattr/dict.get would
    # crash and lose the top-level token counts too.
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "completion_tokens_details": None,
        "prompt_tokens_details": None,
    }
    out = Agent._extract_usage(_resp(usage))
    assert out == {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }
