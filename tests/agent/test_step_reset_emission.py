"""Tests that step_reset events emit on all three retry paths."""

from __future__ import annotations

from types import SimpleNamespace
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
    response.usage = SimpleNamespace(
        prompt_tokens=10, completion_tokens=10, total_tokens=20
    )
    response.id = "r"
    return response


def test_step_reset_emitted_on_rate_limit_fallback():
    events = []
    ctx = ExecutionContext(on_event=events.append)

    rate_limit_err = type("RLE", (Exception,), {"status_code": 429})("rate limited")
    call_results = [rate_limit_err, _make_response()]

    def fake_completion(**kwargs):
        result = call_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    with (
        patch("agentic.agent.agent.litellm.completion", side_effect=fake_completion),
        patch("agentic.agent.agent.classify_error", return_value="rate_limit"),
        patch.dict("os.environ", {"AGENT_LLM_STREAMING_ENABLED": "false"}),
    ):
        agent = Agent(model="anthropic/claude-opus-4-7")
        agent.run("hi", context=ctx, fallback_model="openai/gpt-5.4")

    resets = [e for e in events if e.get("type") == "step_reset"]
    assert len(resets) == 1
    assert resets[0]["step"] == 1
    assert resets[0]["reason"] == "rate_limit"


def test_step_reset_emitted_on_reactive_compact():
    events = []
    ctx = ExecutionContext(on_event=events.append)

    err = Exception("prompt too long")
    call_results = [err, _make_response()]

    def fake_completion(**kwargs):
        r = call_results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    with (
        patch("agentic.agent.agent.litellm.completion", side_effect=fake_completion),
        patch("agentic.agent.agent.classify_error", return_value="prompt_too_long"),
        patch(
            "agentic.agent.agent.compact_messages",
            return_value=[{"role": "user", "content": "compacted"}],
        ),
        patch.dict("os.environ", {"AGENT_LLM_STREAMING_ENABLED": "false"}),
    ):
        agent = Agent(model="anthropic/claude-opus-4-7")
        agent.run("hi", context=ctx)

    resets = [e for e in events if e.get("type") == "step_reset"]
    assert len(resets) == 1
    assert resets[0]["reason"] == "prompt_too_long"


def test_step_reset_emitted_on_output_recovery():
    events = []
    ctx = ExecutionContext(on_event=events.append)

    truncated = _make_response(content="partial", finish_reason="length")
    final = _make_response(content="full final", finish_reason="stop")
    call_results = [truncated, final]

    def fake_completion(**kwargs):
        return call_results.pop(0)

    with (
        patch("agentic.agent.agent.litellm.completion", side_effect=fake_completion),
        patch(
            "agentic.agent.agent.classify_finish_reason",
            side_effect=["max_output_tokens", None],
        ),
        patch.dict("os.environ", {"AGENT_LLM_STREAMING_ENABLED": "false"}),
    ):
        agent = Agent(model="anthropic/claude-opus-4-7")
        agent.run("hi", context=ctx)

    resets = [e for e in events if e.get("type") == "step_reset"]
    assert len(resets) == 1
    assert resets[0]["reason"] == "output_recovery"
