"""Every compaction call site hands compaction the loop's reasoning kwargs."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agentic.agent import compaction
from agentic.agent.agent import Agent
from agentic.agent.tools import BuiltinTool
from agentic.execution.context import ExecutionContext

_INSTRUCTION_TOKENS = compaction.estimate_token_count(
    [{"role": "user", "content": compaction.COMPACTION_INSTRUCTION}]
)

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


@pytest.mark.parametrize("model", ["gpt-5", "anthropic/claude-opus-4-8"])
def test_every_compaction_site_sizes_on_the_threshold_model(model):
    """Compaction sizes its output on the window of ``context_model``; the
    loop must pass the same name its threshold resolves, not the routed one."""
    threshold_models: list[str] = []

    def fake_threshold(m, *args, **kwargs):
        threshold_models.append(m)
        return 0

    with (
        patch("agentic.agent.agent.estimate_token_count", return_value=10_000_000),
        patch("agentic.agent.agent.get_context_threshold", side_effect=fake_threshold),
    ):
        captured = _run(
            model,
            [
                Exception("prompt is too long: 200000 tokens > 100000 maximum"),
                _tool_step(),
                _answer_step(),
            ],
        )
    assert len(captured) >= 3  # reactive, proactive (each step), phase 5
    assert set(threshold_models) == {model}
    assert [kwargs.get("context_model") for kwargs in captured] == [model] * len(
        captured
    )


def _threshold_reserves(reasoning_effort, model="anthropic/claude-opus-4-8"):
    """The output reserve every threshold the loop computes is given, over a
    run that reaches all three sites: proactive (each attempt), the reactive
    truncate target (compaction makes no progress) and phase 5 (after tools)."""
    reserves: list = []
    events: list[dict] = []

    def fake_threshold(model, *args, **kwargs):
        reserves.append(args[0] if args else kwargs.get("max_output_tokens"))
        return 10_000_000

    responses = [
        Exception("prompt is too long: 200000 tokens > 100000 maximum"),
        _tool_step(),
        _answer_step(),
    ]
    with (
        patch("litellm.supports_reasoning", return_value=True),
        patch("agentic.agent.agent.litellm.completion", side_effect=responses),
        patch(
            "agentic.agent.agent.compact_messages",
            side_effect=lambda messages, **kwargs: messages,
        ),
        patch("agentic.agent.agent.get_context_threshold", side_effect=fake_threshold),
        patch.dict(
            "os.environ",
            {"AGENT_LLM_STREAMING_ENABLED": "false", "OPENAI_REASONING_SUMMARY": ""},
        ),
    ):
        agent = Agent(model=model, reasoning_effort=reasoning_effort)
        agent.run(
            "hi",
            context=ExecutionContext(on_event=events.append),
            tools={"probe": _probe_tool()},
        )
    assert any(e.get("type") == "reactive_truncate" for e in events)
    return reserves


def test_threshold_reserves_the_compaction_reasoning_budget():
    """Compaction fires only once the estimate passes the threshold, so the
    room it gets is below whatever the threshold reserved. With reasoning on,
    the threshold reserves the summary call's reasoning budget, or the
    thinking and the summary would share the plain one. It also reserves the
    instruction the compaction call appends to the history."""
    reserves = _threshold_reserves("high")
    assert len(reserves) >= 4
    assert reserves == [16000 + _INSTRUCTION_TOKENS] * len(reserves)


def test_threshold_reserves_the_thinking_budget_of_budget_based_claude():
    """Pre-adaptive Claude thinks within a fixed budget that the compaction
    call's max_tokens must exceed, so the threshold reserves it too."""
    reserves = _threshold_reserves("max", model="anthropic/claude-sonnet-4-5")
    assert len(reserves) >= 4
    assert reserves == [16000 + 16384 + _INSTRUCTION_TOKENS] * len(reserves)


def test_threshold_reserves_max_tokens_without_reasoning():
    reserves = _threshold_reserves(None)
    assert len(reserves) >= 4
    assert reserves == [None] * len(reserves)
