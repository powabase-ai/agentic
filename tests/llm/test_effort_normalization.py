"""Reasoning-effort normalization: a value the model rejects is moved to the
nearest level it accepts, so a user's choice never turns into a provider 400.

The per-model answers come from litellm's cost-map flags plus the entries
agentic registers (agentic/llm/model_cost_additions.json).
"""

import pytest

from agentic.llm.routing import (
    normalize_reasoning_effort,
    reasoning_call_kwargs,
    supported_reasoning_efforts,
)


@pytest.mark.parametrize(
    "model,effort,expected",
    [
        # Adaptive Claude: low..high plus xhigh/max; no none, no minimal.
        ("claude-opus-5-5", "minimal", "low"),
        ("claude-opus-5-5", "none", "low"),
        ("claude-opus-5-5", "max", "max"),
        ("claude-sonnet-5-5", "xhigh", "xhigh"),
        ("claude-fable-5-1", "minimal", "low"),
        ("anthropic/claude-opus-4-8", "minimal", "low"),
        # Pre-adaptive Claude: litellm maps every level to a thinking budget.
        ("claude-sonnet-4-5", "minimal", "minimal"),
        ("claude-sonnet-4-5", "max", "max"),
        # OpenAI: the cost map's per-model flags decide.
        ("gpt-5.5", "minimal", "low"),
        ("gpt-5.5", "none", "none"),
        ("gpt-5.6", "minimal", "low"),
        ("gpt-6-astra", "none", "low"),
        ("gpt-6-luna", "none", "none"),
        ("gpt-6.1-sol", "max", "max"),
        ("gpt-5", "minimal", "minimal"),
        ("gpt-5", "xhigh", "high"),
        ("o3", "max", "high"),
        ("openai/responses/gpt-5.5", "minimal", "low"),
        # Kimi K3 takes low/high/max only; medium goes up, not down.
        ("openrouter/moonshotai/kimi-k3", "medium", "high"),
        ("openrouter/moonshotai/kimi-k3", "none", "low"),
        ("openrouter/moonshotai/kimi-k3", "minimal", "low"),
        ("openrouter/moonshotai/kimi-k3", "max", "max"),
        # Gemini: litellm translates none..high; xhigh/max are rejected.
        ("gemini/gemini-3.8-flash", "xhigh", "high"),
        ("gemini/gemini-3.1-pro-preview", "max", "high"),
        ("gemini/gemini-2.5-flash", "minimal", "minimal"),
        # Supported values pass through unchanged.
        ("claude-opus-5-5", "medium", "medium"),
        ("gpt-6-astra", "high", "high"),
    ],
)
def test_normalize_reasoning_effort(model, effort, expected):
    assert normalize_reasoning_effort(effort, model) == expected


def test_none_effort_stays_none():
    assert normalize_reasoning_effort(None, "claude-opus-5-5") is None


def test_unrecognized_effort_is_dropped():
    """A value no provider knows would 400 everywhere; send no effort at all,
    which every provider accepts."""
    assert normalize_reasoning_effort("turbo", "claude-opus-5-5") is None


def test_effort_is_case_and_whitespace_insensitive():
    assert normalize_reasoning_effort(" High ", "gpt-6-astra") == "high"


def test_unknown_model_passes_effort_through():
    """With no capability data there is nothing to normalize against."""
    assert normalize_reasoning_effort("minimal", "acme/unknown-model") == "minimal"
    assert supported_reasoning_efforts("acme/unknown-model") is None


def test_non_reasoning_model_has_no_supported_efforts():
    assert supported_reasoning_efforts("gpt-4o") is None


def test_reasoning_call_kwargs_sends_the_normalized_effort_to_claude():
    """Claude 4.7+ efforts go out as output_config.effort, which litellm does
    not validate -- 'minimal' would reach Anthropic and 400."""
    kwargs = reasoning_call_kwargs("minimal", "claude-opus-5-5")
    assert kwargs["output_config"] == {"effort": "low"}


def test_reasoning_call_kwargs_sends_the_normalized_effort_on_responses():
    """On the Responses route effort is packed into extra_body, which litellm
    does not validate either."""
    kwargs = reasoning_call_kwargs("minimal", "openai/responses/gpt-5.5")
    assert kwargs["extra_body"]["reasoning"]["effort"] == "low"


def test_reasoning_call_kwargs_with_unrecognized_effort_sends_nothing():
    assert reasoning_call_kwargs("turbo", "claude-opus-5-5") == {}
