"""Every compaction call site hands compaction the loop's reasoning kwargs."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agentic.agent.agent import Agent
from agentic.agent.tools import BuiltinTool
from agentic.execution.context import ExecutionContext

_EXPECTED = {
    "anthropic/claude-opus-4-8": {
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "high"},
    },
    "openai/gpt-5.4": {
        "extra_body": {
            "reasoning": {"effort": "high"},
            "include": ["reasoning.encrypted_content"],
        }
    },
}


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
        reasoning_content=None,
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                type="function",
                function=SimpleNamespace(name="probe", arguments="{}"),
            )
        ],
        thinking_blocks=None,
        provider_specific_fields=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
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


def _run(model, responses):
    captured: list[dict] = []

    def fake_compact(messages, **kwargs):
        captured.append(kwargs)
        return messages  # no progress: the loop carries on uncompacted

    with (
        patch("litellm.supports_reasoning", return_value=True),
        patch("agentic.agent.agent.litellm.completion", side_effect=responses),
        patch("agentic.agent.agent.compact_messages", side_effect=fake_compact),
        patch.dict(
            "os.environ",
            {"AGENT_LLM_STREAMING_ENABLED": "false", "OPENAI_REASONING_SUMMARY": ""},
        ),
    ):
        agent = Agent(model=model, reasoning_effort="high")
        agent.run("hi", context=ExecutionContext(), tools={"probe": _probe_tool()})
    return captured


@pytest.mark.parametrize("model", sorted(_EXPECTED))
def test_proactive_and_phase5_compaction_get_the_loops_kwargs(model):
    with (
        patch("agentic.agent.agent.estimate_token_count", return_value=10_000_000),
        patch("agentic.agent.agent.get_context_threshold", return_value=0),
    ):
        captured = _run(model, [_tool_step(), _answer_step()])
    assert len(captured) >= 2  # proactive (each step) + phase 5 (after tools)
    for kwargs in captured:
        assert kwargs["reasoning_kwargs"] == _EXPECTED[model]


@pytest.mark.parametrize("model", sorted(_EXPECTED))
def test_reactive_compaction_gets_the_loops_kwargs(model):
    captured = _run(
        model,
        [
            Exception("prompt is too long: 200000 tokens > 100000 maximum"),
            _answer_step(),
        ],
    )
    assert captured
    for kwargs in captured:
        assert kwargs["reasoning_kwargs"] == _EXPECTED[model]
