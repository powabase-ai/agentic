"""Reasoning-effort normalization: a value the model rejects is moved to the
nearest level it accepts, so a user's choice never turns into a provider 400,
and "none" never switches reasoning on.

The per-model answers come from litellm's bundled cost map (tests/conftest.py
pins it) plus the entries agentic registers (agentic/llm/model_cost_additions.json).
"""

import pytest

from agentic.llm.routing import (
    loop_reasoning_call_kwargs,
    maybe_route_through_responses,
    normalize_reasoning_effort,
    reasoning_call_kwargs,
    sampling_call_kwargs,
    supported_reasoning_efforts,
)


@pytest.mark.parametrize(
    "model,effort,expected",
    [
        # Adaptive Claude: low..max; neither none nor minimal is an
        # output_config effort.
        ("claude-opus-5-5", "minimal", "low"),
        ("claude-opus-5-5", "max", "max"),
        ("claude-sonnet-5-5", "xhigh", "xhigh"),
        ("claude-fable-5-1", "minimal", "low"),
        ("anthropic/claude-opus-4-8", "minimal", "low"),
        # Bedrock ids carry no effort flags; adaptive Claude keeps its levels.
        ("bedrock/us.anthropic.claude-opus-4-7-v1:0", "xhigh", "xhigh"),
        ("bedrock/us.anthropic.claude-opus-4-7-v1:0", "minimal", "low"),
        # Pre-adaptive Claude: litellm maps every level to a thinking budget.
        ("claude-sonnet-4-5", "minimal", "minimal"),
        ("claude-sonnet-4-5", "max", "max"),
        # OpenAI: litellm's resolver, bare-model twin included.
        ("gpt-5.5", "minimal", "low"),
        ("gpt-5.5", "none", "none"),
        ("gpt-5.6", "minimal", "low"),
        ("gpt-6-luna", "none", "none"),
        ("gpt-6.1-sol", "max", "max"),
        ("gpt-5", "minimal", "minimal"),
        ("gpt-5", "xhigh", "high"),
        ("azure/gpt-5-mini", "minimal", "minimal"),
        ("o3", "max", "high"),
        ("openai/responses/gpt-5.5", "minimal", "low"),
        # Kimi K3 takes low/high/max only; medium rounds up, not down.
        ("openrouter/moonshotai/kimi-k3", "medium", "high"),
        ("openrouter/moonshotai/kimi-k3", "minimal", "low"),
        ("openrouter/moonshotai/kimi-k3", "max", "max"),
        # ...wherever litellm declares those levels itself.
        ("moonshot/kimi-k3", "medium", "high"),
        # Gemini: litellm translates none..high; xhigh/max are rejected.
        ("gemini/gemini-3.8-flash", "xhigh", "high"),
        ("gemini/gemini-3.1-pro-preview", "max", "high"),
        ("gemini/gemini-2.5-flash", "minimal", "minimal"),
        ("gemini/gemini-3.8-flash", "none", "none"),
        # Supported values pass through unchanged.
        ("claude-opus-5-5", "medium", "medium"),
        ("gpt-6-astra", "high", "high"),
        # A mapped reasoning model with no effort metadata passes through.
        ("deepseek/deepseek-reasoner", "max", "max"),
        ("openrouter/deepseek/deepseek-r1", "minimal", "minimal"),
    ],
)
def test_normalize_reasoning_effort(model, effort, expected):
    assert normalize_reasoning_effort(effort, model) == expected


@pytest.mark.parametrize(
    "model",
    [
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-opus-4-7",
        "claude-opus-4-8",
        "openrouter/anthropic/claude-opus-4.5",
        "openrouter/deepseek/deepseek-r1",
    ],
)
def test_none_sends_no_effort_where_reasoning_can_be_switched_off(model):
    """'none' is the off switch: on a model without a 'none' level, nothing
    is sent rather than its weakest level, which would switch thinking on."""
    assert normalize_reasoning_effort("none", model) is None
    assert reasoning_call_kwargs("none", model) == {}


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-opus-5-5", "low"),
        ("claude-sonnet-5-5", "low"),
        ("claude-fable-5-1", "low"),
        # Opus 5 / Sonnet 5 aren't thinking_always_on, but omitting thinking
        # runs them adaptive at their (high) default -- not off.
        ("claude-opus-5", "low"),
        ("claude-sonnet-5", "low"),
        ("anthropic.claude-opus-5", "low"),
        ("vertex_ai/claude-opus-5-5", "low"),
        ("bedrock/anthropic.claude-opus-5-5", "low"),
        ("openrouter/moonshotai/kimi-k3", "low"),
        # Kimi K3 reasons on every provider, not just OpenRouter.
        ("moonshot/kimi-k3", "low"),
        ("gpt-6-astra", "low"),
        ("gpt-5", "minimal"),
        ("o3", "low"),
    ],
)
def test_none_on_a_model_that_always_reasons_is_its_weakest_level(model, expected):
    """These can't switch reasoning off; sending nothing would mean their
    (stronger) default, so the lightest level is the closest to 'none'."""
    assert normalize_reasoning_effort("none", model) == expected


def test_none_effort_stays_none():
    assert normalize_reasoning_effort(None, "claude-opus-5-5") is None


def test_unrecognized_effort_is_dropped():
    """A value no provider knows would 400 everywhere; send no effort at all,
    which every provider accepts."""
    assert normalize_reasoning_effort("turbo", "claude-opus-5-5") is None
    assert normalize_reasoning_effort(3, "claude-opus-5-5") is None


def test_unrecognized_effort_does_not_route_through_responses():
    assert maybe_route_through_responses("gpt-5.6", "turbo") == "gpt-5.6"
    assert loop_reasoning_call_kwargs("turbo", "openai/responses/gpt-5.6") == {}


def test_effort_is_case_and_whitespace_insensitive():
    assert normalize_reasoning_effort(" High ", "gpt-6-astra") == "high"


def test_unknown_model_passes_effort_through():
    """With no capability data there is nothing to normalize against."""
    assert normalize_reasoning_effort("minimal", "acme/unknown-model") == "minimal"
    assert supported_reasoning_efforts("acme/unknown-model") is None


def test_non_reasoning_model_has_no_supported_efforts():
    assert supported_reasoning_efforts("gpt-4o") is None


@pytest.mark.parametrize(
    "model", ["openrouter/moonshotai/kimi-k3:batch", "OpenRouter/MoonshotAI/Kimi-K3"]
)
def test_kimi_k3_override_covers_batch_and_any_case(model):
    assert supported_reasoning_efforts(model) == frozenset({"low", "high", "max"})
    assert normalize_reasoning_effort("medium", model) == "high"


def test_kimi_k3_override_is_limited_to_openrouter_ids():
    """A substring match would also catch other providers' entries and
    future kimi-k3.x models, whose levels litellm describes itself."""
    assert supported_reasoning_efforts("perplexity/perplexity/kimi-k3") != frozenset(
        {"low", "high", "max"}
    )


def test_reasoning_call_kwargs_sends_the_normalized_effort_to_claude():
    """Anthropic rejects 'minimal' as an output_config effort."""
    kwargs = reasoning_call_kwargs("minimal", "claude-opus-5-5")
    assert kwargs["output_config"] == {"effort": "low"}


def test_reasoning_call_kwargs_sends_the_normalized_effort_on_responses():
    kwargs = reasoning_call_kwargs("minimal", "openai/responses/gpt-5.5")
    assert kwargs["extra_body"]["reasoning"]["effort"] == "low"


def test_reasoning_call_kwargs_with_unrecognized_effort_sends_nothing():
    assert reasoning_call_kwargs("turbo", "claude-opus-5-5") == {}


# ===== sampling_call_kwargs =====


@pytest.mark.parametrize(
    "model,effort",
    [
        ("claude-opus-5-5", None),
        ("claude-sonnet-5-5", "medium"),
        ("claude-fable-5-1", None),
        ("claude-opus-4-8", None),
        ("gpt-5.6", None),
        ("gpt-5.6", "high"),
        ("openai/responses/gpt-6-astra", "low"),
        ("o3", None),
        # Budget-based Claude thinking is incompatible with a temperature.
        ("claude-haiku-4-5", "medium"),
        ("claude-sonnet-4-5", "high"),
        # Claude 4.7+/5.x take no sampling parameters on any route, including
        # cloud ids whose cost-map entry doesn't say so.
        ("bedrock/anthropic.claude-opus-5-5", None),
        ("bedrock/anthropic.claude-opus-5-5", "high"),
    ],
)
def test_temperature_left_out_where_the_model_rejects_it(model, effort):
    assert sampling_call_kwargs(model, 0.3, effort) == {}


@pytest.mark.parametrize(
    "model,effort",
    [
        ("gpt-4o", None),
        ("claude-haiku-4-5", None),
        ("gpt-5.6", "none"),
        ("gpt-6-luna", "none"),
        ("gemini/gemini-3.8-flash", "medium"),
        ("openrouter/moonshotai/kimi-k3", "high"),
        ("acme/unknown-model", None),
        # The budget-thinking rule is Anthropic's; other providers that map
        # effort to a thinking budget keep their temperature.
        ("deepseek/deepseek-reasoner", "high"),
        # A routed id is looked up under its unrouted name.
        ("azure/responses/eu/gpt-5.4", None),
    ],
)
def test_temperature_kept_where_the_model_takes_it(model, effort):
    assert sampling_call_kwargs(model, 0.3, effort) == {"temperature": 0.3}


def test_no_temperature_sends_none():
    assert sampling_call_kwargs("gpt-4o", None, None) == {}
