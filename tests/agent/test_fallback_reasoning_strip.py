"""A mid-run fallback to another provider must not replay the first provider's
reasoning — LiteLLM would convert it into malformed input for the new one."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

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


def _tool_step():
    msg = SimpleNamespace(
        content=None,
        role="assistant",
        reasoning_content="I should probe.",
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                type="function",
                function=SimpleNamespace(name="probe", arguments="{}"),
            )
        ],
        thinking_blocks=_BLOCKS,
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


def _run_with_fallback(fallback_model, history=None):
    """Step 1 (Claude, thinking + tool call) succeeds, step 2 is rate limited,
    so step 2 is retried on `fallback_model`."""
    events: list[dict] = []
    responses = [_tool_step(), _RateLimited("rate limited"), _answer_step()]

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
        patch("agentic.agent.agent.classify_error", return_value="rate_limit"),
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
    dropped = [e for e in events if e.get("type") == _EVENT]
    return output, completion.call_args_list[2], dropped


_HISTORY = [
    {"role": "user", "content": "earlier question"},
    {"role": "assistant", "content": "earlier answer", "thinking_blocks": _OLD_BLOCKS},
]


def test_cross_provider_fallback_strips_every_assistant_message():
    _, fallback_call, dropped = _run_with_fallback("openai/gpt-5.4", _HISTORY)
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


def test_same_provider_fallback_keeps_the_blocks():
    _, fallback_call, dropped = _run_with_fallback(
        "anthropic/claude-sonnet-5", _HISTORY
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
