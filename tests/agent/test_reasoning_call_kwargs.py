"""Tests that Agent.run forwards reasoning_effort + routes OpenAI through
Responses bridge correctly. Critical contract: on Responses path, top-level
reasoning_effort must NOT be present (LiteLLM 1.83.14 silently drops it
from the outgoing request — verified empirically in A0.1)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agentic.agent.agent import Agent
from agentic.execution.context import ExecutionContext


def _make_response(content="ok", finish_reason="stop"):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    msg.reasoning_content = None
    msg.thinking_blocks = None
    msg.provider_specific_fields = None
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason
    response = MagicMock()
    response.choices = [choice]
    response.usage = None
    response.id = "test-id"
    return response


def test_anthropic_adaptive_model_sends_summarized_thinking():
    """Anthropic adaptive models whose display defaults to "omitted"
    (opus 4.7/4.8, sonnet 5, fable/mythos) get an explicit
    adaptive+summarized thinking config + output_config effort so reasoning
    is surfaced. No model transformation; no top-level reasoning_effort;
    no extra_body."""
    with (
        patch("litellm.supports_reasoning", return_value=True),
        patch(
            "agentic.agent.agent.litellm.completion", return_value=_make_response()
        ) as mock_completion,
        patch.dict("os.environ", {"AGENT_LLM_STREAMING_ENABLED": "false"}),
    ):
        agent = Agent(
            model="anthropic/claude-opus-4-7",
            reasoning_effort="medium",
        )
        agent.run("hi", context=ExecutionContext())

    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs["model"] == "anthropic/claude-opus-4-7"
    assert call_kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert call_kwargs["output_config"] == {"effort": "medium"}
    assert "reasoning_effort" not in call_kwargs
    assert "extra_body" not in call_kwargs


def test_openai_routes_through_responses_with_packed_extra_body():
    """OpenAI reasoning models get rerouted to openai/responses/<model>.
    Effort is packed into extra_body['reasoning']; by default NO summary is
    requested (unverified OpenAI orgs 400 on reasoning-summary requests).
    Top-level reasoning_effort must NOT be set (LiteLLM 1.83.14 bug avoidance)."""
    with (
        patch("litellm.supports_reasoning", return_value=True),
        patch(
            "agentic.agent.agent.litellm.completion", return_value=_make_response()
        ) as mock_completion,
        patch.dict(
            "os.environ",
            {"AGENT_LLM_STREAMING_ENABLED": "false", "OPENAI_REASONING_SUMMARY": ""},
        ),
    ):
        agent = Agent(
            model="openai/gpt-5.4",
            reasoning_effort="medium",
        )
        agent.run("hi", context=ExecutionContext())

    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs["model"] == "openai/responses/gpt-5.4"
    assert call_kwargs["extra_body"] == {"reasoning": {"effort": "medium"}}
    # Critical: NO top-level reasoning_effort on the Responses path
    assert "reasoning_effort" not in call_kwargs


def test_openai_responses_includes_summary_when_opted_in():
    """With OPENAI_REASONING_SUMMARY=1 (verified org), the summary is requested
    again so the reasoning-display UI gets summary text."""
    with (
        patch("litellm.supports_reasoning", return_value=True),
        patch(
            "agentic.agent.agent.litellm.completion", return_value=_make_response()
        ) as mock_completion,
        patch.dict(
            "os.environ",
            {"AGENT_LLM_STREAMING_ENABLED": "false", "OPENAI_REASONING_SUMMARY": "1"},
        ),
    ):
        agent = Agent(model="openai/gpt-5.4", reasoning_effort="medium")
        agent.run("hi", context=ExecutionContext())

    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs["extra_body"] == {
        "reasoning": {"effort": "medium", "summary": "detailed"}
    }


def test_no_reasoning_effort_means_no_routing_no_extra_body():
    with (
        patch(
            "agentic.agent.agent.litellm.completion", return_value=_make_response()
        ) as mock_completion,
        patch.dict("os.environ", {"AGENT_LLM_STREAMING_ENABLED": "false"}),
    ):
        agent = Agent(model="openai/gpt-5.4")  # no reasoning_effort
        agent.run("hi", context=ExecutionContext())

    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs["model"] == "openai/gpt-5.4"  # unchanged
    assert "reasoning_effort" not in call_kwargs
    assert "extra_body" not in call_kwargs


def test_unsupported_model_drops_effort_no_routing():
    """Precheck dropped effort at construct time → no routing, no kwargs."""
    with (
        patch("litellm.supports_reasoning", return_value=False),
        patch(
            "agentic.agent.agent.litellm.completion", return_value=_make_response()
        ) as mock_completion,
        patch.dict("os.environ", {"AGENT_LLM_STREAMING_ENABLED": "false"}),
    ):
        agent = Agent(model="openai/gpt-4o", reasoning_effort="medium")
        agent.run("hi", context=ExecutionContext())

    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs["model"] == "openai/gpt-4o"
    assert "reasoning_effort" not in call_kwargs
    assert "extra_body" not in call_kwargs


def test_fallback_triggers_one_supports_reasoning_lookup_per_model():
    """When model_fallback fires mid-run, _resolved_effort_for caches:
    one new precheck for the fallback model, none for repeat hits on the
    same model. Verifies the per-model cache from Task 2 integrates with
    the call-time routing."""
    rate_limit_err = type("RLE", (Exception,), {"status_code": 429})("rate limited")
    fallback_resp = _make_response()
    call_results = [rate_limit_err, fallback_resp]

    def fake_completion(**kwargs):
        r = call_results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    with (
        patch("litellm.supports_reasoning", return_value=True) as mock_supports,
        patch("agentic.agent.agent.litellm.completion", side_effect=fake_completion),
        patch("agentic.agent.agent.classify_error", return_value="rate_limit"),
        patch.dict("os.environ", {"AGENT_LLM_STREAMING_ENABLED": "false"}),
    ):
        agent = Agent(
            model="anthropic/claude-opus-4-7",
            reasoning_effort="medium",
        )
        # Construction precheck: 1 call to litellm.supports_reasoning
        assert mock_supports.call_count == 1
        agent.run("hi", context=ExecutionContext(), fallback_model="openai/gpt-5.4")

    # The Agent's per-model precheck cache must trigger exactly one
    # _resolve_effort_for_model invocation per distinct model: anthropic at
    # construct, openai/gpt-5.4 lazily at the fallback's first use. Note that
    # maybe_route_through_responses (Task 4) does its OWN supports_reasoning
    # lookup defensively for openai/azure models, so the raw call_count can
    # exceed the distinct-model count; this assertion is on the distinct set.
    call_args_list = [c.kwargs.get("model") for c in mock_supports.call_args_list]
    assert set(call_args_list) == {"anthropic/claude-opus-4-7", "openai/gpt-5.4"}
