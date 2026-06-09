"""Tests for per-call routing helpers."""

import os
from unittest.mock import patch

from agentic.llm.routing import (
    maybe_route_through_responses,
    reasoning_call_kwargs,
)

# ===== maybe_route_through_responses =====


def test_no_routing_when_effort_none():
    assert maybe_route_through_responses("openai/gpt-5.4", None) == "openai/gpt-5.4"


def test_no_routing_for_anthropic():
    with patch("litellm.supports_reasoning", return_value=True):
        assert (
            maybe_route_through_responses("anthropic/claude-opus-4-7", "medium")
            == "anthropic/claude-opus-4-7"
        )


def test_no_routing_for_gemini():
    with patch("litellm.supports_reasoning", return_value=True):
        assert (
            maybe_route_through_responses("gemini/gemini-3-pro-preview", "medium")
            == "gemini/gemini-3-pro-preview"
        )


def test_routes_openai_reasoning_model():
    with patch("litellm.supports_reasoning", return_value=True):
        assert (
            maybe_route_through_responses("openai/gpt-5.4", "medium")
            == "openai/responses/gpt-5.4"
        )


def test_routes_azure_reasoning_model():
    with patch("litellm.supports_reasoning", return_value=True):
        assert (
            maybe_route_through_responses("azure/my-deployment", "medium")
            == "azure/responses/my-deployment"
        )


def test_does_not_double_route():
    with patch("litellm.supports_reasoning", return_value=True):
        assert (
            maybe_route_through_responses("openai/responses/gpt-5.4", "medium")
            == "openai/responses/gpt-5.4"
        )


def test_does_not_route_non_reasoning_openai():
    with patch("litellm.supports_reasoning", return_value=False):
        assert (
            maybe_route_through_responses("openai/gpt-4o", "medium") == "openai/gpt-4o"
        )


def test_does_not_route_when_supports_reasoning_raises():
    with patch("litellm.supports_reasoning", side_effect=Exception("unknown")):
        assert (
            maybe_route_through_responses("openai/some-future-model", "medium")
            == "openai/some-future-model"
        )


def test_routes_bare_openai_model_via_get_llm_provider():
    """Agent UI stores models bare (e.g. ``gpt-5.4`` not ``openai/gpt-5.4``).
    The router must resolve provider via litellm.get_llm_provider and still
    reroute through the Responses bridge."""
    with (
        patch(
            "agentic.llm.routing.litellm.get_llm_provider",
            return_value=("gpt-5.4", "openai", None, None),
        ),
        patch("litellm.supports_reasoning", return_value=True),
    ):
        assert (
            maybe_route_through_responses("gpt-5.4", "medium")
            == "openai/responses/gpt-5.4"
        )


def test_does_not_route_bare_anthropic_model():
    """Bare ``claude-opus-4-7`` resolves to provider ``anthropic`` → unchanged
    (Anthropic returns reasoning natively on Chat Completions)."""
    with patch(
        "agentic.llm.routing.litellm.get_llm_provider",
        return_value=("claude-opus-4-7", "anthropic", None, None),
    ):
        assert (
            maybe_route_through_responses("claude-opus-4-7", "medium")
            == "claude-opus-4-7"
        )


def test_does_not_route_when_get_llm_provider_raises():
    """Custom / unknown models that get_llm_provider can't resolve: leave alone."""
    with patch(
        "agentic.llm.routing.litellm.get_llm_provider",
        side_effect=Exception("unknown"),
    ):
        assert (
            maybe_route_through_responses("custom/unknown-model", "medium")
            == "custom/unknown-model"
        )


# ===== reasoning_call_kwargs =====


def test_call_kwargs_when_effort_none():
    """No effort, no kwargs to add."""
    assert reasoning_call_kwargs(None, "openai/gpt-5.4") == {}


def test_call_kwargs_when_effort_none_on_responses_model():
    """Even on a Responses-routed model, no effort means no kwargs."""
    assert reasoning_call_kwargs(None, "openai/responses/gpt-5.4") == {}


def test_call_kwargs_for_anthropic_uses_top_level_effort():
    """Non-Responses paths use the standard reasoning_effort kwarg."""
    assert reasoning_call_kwargs("medium", "anthropic/claude-opus-4-7") == {
        "reasoning_effort": "medium"
    }


def test_call_kwargs_for_gemini_uses_top_level_effort():
    assert reasoning_call_kwargs("high", "gemini/gemini-3-pro-preview") == {
        "reasoning_effort": "high"
    }


def test_call_kwargs_for_openai_chat_completions_uses_top_level_effort():
    """OpenAI without /responses/ prefix (e.g., a non-reasoning model that
    somehow has effort set) uses top-level kwarg too."""
    assert reasoning_call_kwargs("medium", "openai/gpt-4o") == {
        "reasoning_effort": "medium"
    }


def test_call_kwargs_for_responses_path_packs_effort_only_by_default():
    """Responses-routed models put effort in extra_body['reasoning']. By
    default NO summary is requested: OpenAI rejects reasoning-summary requests
    from unverified orgs with HTTP 400 ("Your organization must be verified to
    generate reasoning summaries"), which would fail the whole call for every
    OpenAI reasoning model. Top-level reasoning_effort is still omitted (it is
    silently dropped on the Responses path in litellm 1.83.14 — A0.1)."""
    with patch.dict(os.environ, {"OPENAI_REASONING_SUMMARY": ""}):
        assert reasoning_call_kwargs("medium", "openai/responses/gpt-5.4") == {
            "extra_body": {"reasoning": {"effort": "medium"}}
        }


def test_call_kwargs_for_azure_responses_path_effort_only_by_default():
    with patch.dict(os.environ, {"OPENAI_REASONING_SUMMARY": ""}):
        assert reasoning_call_kwargs("low", "azure/responses/my-deployment") == {
            "extra_body": {"reasoning": {"effort": "low"}}
        }


def test_call_kwargs_responses_path_includes_summary_when_opted_in():
    """Verified-org deployments can re-enable reasoning summaries (for the
    reasoning-display UI) via OPENAI_REASONING_SUMMARY=1."""
    with patch.dict(os.environ, {"OPENAI_REASONING_SUMMARY": "1"}):
        assert reasoning_call_kwargs("medium", "openai/responses/gpt-5.4") == {
            "extra_body": {"reasoning": {"effort": "medium", "summary": "detailed"}}
        }


def test_call_kwargs_responses_path_summary_opt_in_truthy_values():
    for val in ("true", "TRUE", "yes", "on"):
        with patch.dict(os.environ, {"OPENAI_REASONING_SUMMARY": val}):
            assert reasoning_call_kwargs("high", "openai/responses/gpt-5.4") == {
                "extra_body": {"reasoning": {"effort": "high", "summary": "detailed"}}
            }


def test_call_kwargs_does_not_set_top_level_effort_on_responses_path():
    """Critical: confirm the Responses path does NOT have top-level
    reasoning_effort. The LiteLLM 1.83.14 bug drops it from the outgoing
    request anyway, but we don't even pass it. This is what
    distinguishes the Responses path from the non-Responses path."""
    kwargs = reasoning_call_kwargs("medium", "openai/responses/gpt-5.4")
    assert "reasoning_effort" not in kwargs
