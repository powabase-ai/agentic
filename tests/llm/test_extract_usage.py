"""Tests for `streaming._extract_usage` — the streaming-path twin of
`Agent._extract_usage`.

Same shape-normalization logic but a different missing-value contract:
the streaming path defaults missing core fields (prompt/completion/total)
to ``0`` instead of dropping the key, because the persist path summing
chunk counts needs an integer to add into. This file pins that contract
so it doesn't drift from the non-streaming variant by accident.
"""

from __future__ import annotations

import types

from agentic.llm.streaming import _extract_usage


# ---------------------------------------------------------------------------
# Empty / missing
# ---------------------------------------------------------------------------


def test_returns_zero_filled_when_chunk_usage_is_none():
    # Streaming variant ALWAYS returns the three core int fields. None
    # input → all zeros, no None values that would crash arithmetic
    # downstream.
    out = _extract_usage(None)
    assert out == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def test_returns_zero_filled_when_no_token_keys_present():
    out = _extract_usage({})
    assert out == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


# ---------------------------------------------------------------------------
# Chat Completions naming (OpenAI default)
# ---------------------------------------------------------------------------


def test_chat_completions_dict_shape():
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "completion_tokens_details": {"reasoning_tokens": 30},
        "prompt_tokens_details": {"cached_tokens": 80},
    }
    out = _extract_usage(usage)
    assert out == {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "reasoning_tokens": 30,
        "cached_tokens": 80,
    }


def test_chat_completions_pydantic_shape():
    details = types.SimpleNamespace(reasoning_tokens=30)
    prompt_details = types.SimpleNamespace(cached_tokens=80)
    usage = types.SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        completion_tokens_details=details,
        prompt_tokens_details=prompt_details,
    )
    out = _extract_usage(usage)
    assert out["reasoning_tokens"] == 30
    assert out["cached_tokens"] == 80


# ---------------------------------------------------------------------------
# Responses API naming
# ---------------------------------------------------------------------------


def test_responses_api_input_output_token_names_remap_to_prompt_completion():
    usage = {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    }
    out = _extract_usage(usage)
    assert out["prompt_tokens"] == 100
    assert out["completion_tokens"] == 50
    assert out["total_tokens"] == 150


def test_responses_api_details_under_output_tokens_details():
    usage = {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "output_tokens_details": {"reasoning_tokens": 30},
        "input_tokens_details": {"cached_tokens": 80},
    }
    out = _extract_usage(usage)
    assert out["reasoning_tokens"] == 30
    assert out["cached_tokens"] == 80


# ---------------------------------------------------------------------------
# Streaming-specific contract: zero-fill, NOT drop
# ---------------------------------------------------------------------------


def test_zero_reasoning_tokens_is_kept():
    # Same as the non-streaming variant — 0 is a real value, must survive.
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "completion_tokens_details": {"reasoning_tokens": 0},
    }
    out = _extract_usage(usage)
    assert out["reasoning_tokens"] == 0


def test_missing_reasoning_tokens_is_omitted_but_core_keys_remain():
    # Reasoning is absent → key dropped (matches non-streaming variant).
    # BUT prompt/completion/total stay at 0 (does NOT match non-streaming
    # variant, which would drop them). This divergence is intentional —
    # see the file docstring.
    usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    out = _extract_usage(usage)
    assert "reasoning_tokens" not in out
    assert "cached_tokens" not in out
    assert out["prompt_tokens"] == 100


def test_chat_completions_takes_precedence_over_responses():
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "completion_tokens_details": {"reasoning_tokens": 30},
        "output_tokens_details": {"reasoning_tokens": 999},
    }
    out = _extract_usage(usage)
    assert out["reasoning_tokens"] == 30


def test_handles_none_details_object_gracefully():
    # Some providers stub the *_details key to None. The _get helper
    # short-circuits on None so the top-level token counts survive.
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "completion_tokens_details": None,
        "prompt_tokens_details": None,
    }
    out = _extract_usage(usage)
    assert out == {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }


def test_none_token_value_treated_as_zero():
    # `prompt or 0` — None is falsy → 0. Different from the non-streaming
    # variant which drops the key. Pin the divergence.
    usage = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }
    out = _extract_usage(usage)
    assert out == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
