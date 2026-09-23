"""A mid-run fallback to another provider must not replay the first provider's
reasoning — LiteLLM would convert it into malformed input for the new one."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agentic.agent.agent import Agent
from agentic.agent.tools import BuiltinTool
from agentic.execution.context import ExecutionContext

_BLOCKS = [{"type": "thinking", "thinking": "I should probe.", "signature": "sig1"}]
_OLD_BLOCKS = [{"type": "thinking", "thinking": "earlier", "signature": "sig0"}]
_EVENT = "reasoning_dropped_at_provider_switch"


class _RateLimited(Exception):
    status_code = 429


def _probe_tool():
    return BuiltinTool(
        name="probe",
        description="probe",
        input_schema={"type": "object", "properties": {}},
        handler=lambda args, ctx: "ok",
    )


def _tool_step(blocks=_BLOCKS):
    msg = SimpleNamespace(
        content=None,
        role="assistant",
        reasoning_content="I should probe." if blocks else None,
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                type="function",
                function=SimpleNamespace(name="probe", arguments="{}"),
            )
        ],
        thinking_blocks=blocks,
        provider_specific_fields=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        id="resp_1",
    )


def _answer_step():
    msg = SimpleNamespace(
        content="done",
        role="assistant",
        reasoning_content=None,
        tool_calls=None,
        thinking_blocks=None,
        provider_specific_fields=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        id="resp_2",
    )


def _run_with_fallback(
    fallback_model, history=None, error_type="rate_limit", first_step=None
):
    """Step 1 (Claude, thinking + tool call) succeeds, step 2 fails with an
    error classified as `error_type` (a rate limit or a server error — the two
    fallback sites), so step 2 is retried on `fallback_model`."""
    events: list[dict] = []
    responses = [
        first_step or _tool_step(),
        _RateLimited("provider error"),
        _answer_step(),
    ]

    def fake_completion(**kwargs):
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    with (
        patch("litellm.supports_reasoning", return_value=True),
        patch(
            "agentic.agent.agent.litellm.completion", side_effect=fake_completion
        ) as completion,
        patch("agentic.agent.agent.classify_error", return_value=error_type),
        patch.dict("os.environ", {"AGENT_LLM_STREAMING_ENABLED": "false"}),
    ):
        agent = Agent(model="anthropic/claude-opus-4-8", reasoning_effort="high")
        output = agent.run(
            (history or []) + [{"role": "user", "content": "hi"}],
            context=ExecutionContext(on_event=events.append),
            tools={"probe": _probe_tool()},
            fallback_model=fallback_model,
        )
    assert output.status.is_success()
    assert completion.call_count == 3
    # The retry went through the fallback site for `error_type`.
    assert [e["reason"] for e in events if e.get("type") == "step_reset"] == [
        error_type
    ]
    dropped = [e for e in events if e.get("type") == _EVENT]
    return output, completion.call_args_list[2], dropped


_HISTORY = [
    {"role": "user", "content": "earlier question"},
    {"role": "assistant", "content": "earlier answer", "thinking_blocks": _OLD_BLOCKS},
]


_FALLBACK_ERRORS = pytest.mark.parametrize("error_type", ["rate_limit", "model_error"])


@_FALLBACK_ERRORS
def test_cross_provider_fallback_strips_every_assistant_message(error_type):
    _, fallback_call, dropped = _run_with_fallback(
        "openai/gpt-5.4", _HISTORY, error_type=error_type
    )
    assert fallback_call.kwargs["model"] == "openai/responses/gpt-5.4"
    for message in fallback_call.kwargs["messages"]:
        assert "thinking_blocks" not in message
    # emit_event always adds "seq"/"ts" bookkeeping (see
    # ExecutionContext.emit_event); compare the business fields only, the
    # same way the sibling model_fallback/step_reset events are asserted.
    assert [
        {k: v for k, v in e.items() if k not in ("seq", "ts")} for e in dropped
    ] == [
        {
            "type": _EVENT,
            "step": 2,
            "from_provider": "anthropic",
            "to_provider": "openai",
        }
    ]


def test_the_reasoning_record_survives_the_strip():
    output, _, _ = _run_with_fallback("openai/gpt-5.4")
    step_one = [m for m in output.messages if m.get("role") == "assistant"][0]
    assert step_one["reasoning"]["provider"] == "anthropic"
    assert "thinking_blocks" not in step_one


@_FALLBACK_ERRORS
def test_same_provider_fallback_keeps_the_blocks(error_type):
    _, fallback_call, dropped = _run_with_fallback(
        "anthropic/claude-sonnet-5", _HISTORY, error_type=error_type
    )
    assistants = [
        m for m in fallback_call.kwargs["messages"] if m.get("role") == "assistant"
    ]
    assert assistants[0]["thinking_blocks"] == _OLD_BLOCKS
    assert assistants[-1]["thinking_blocks"] == _BLOCKS
    assert dropped == []


def test_unresolvable_fallback_keeps_the_blocks():
    """The host's rule: a provider LiteLLM cannot name preserves reasoning."""
    _, fallback_call, dropped = _run_with_fallback("no-such-provider-model-xyz")
    prior = [
        m for m in fallback_call.kwargs["messages"] if m.get("role") == "assistant"
    ][-1]
    assert prior["thinking_blocks"] == _BLOCKS
    assert dropped == []


@_FALLBACK_ERRORS
def test_cross_provider_fallback_with_nothing_to_drop_emits_no_event(error_type):
    """The event reports dropped reasoning; a history with no replay fields
    drops nothing, so a provider switch alone emits none."""
    _, fallback_call, dropped = _run_with_fallback(
        "openai/gpt-5.4", error_type=error_type, first_step=_tool_step(blocks=None)
    )
    assert fallback_call.kwargs["model"] == "openai/responses/gpt-5.4"
    assert dropped == []
