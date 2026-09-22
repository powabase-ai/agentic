"""Session history replays reasoning only to the provider that produced it,
and never carries OpenAI reasoning items across runs."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from agentic.agent.agent import Agent
from agentic.agent.output import AgentOutput
from agentic.agent.session import AgentSession
from agentic.execution.context import ExecutionContext
from agentic.execution.status import ExecutionStatus

_BLOCKS = [{"type": "thinking", "thinking": "earlier", "signature": "sig0"}]
_ITEMS = [
    {"id": "rs_0", "type": "reasoning", "encrypted_content": "enc", "summary": []}
]


def _answer():
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
        id="resp_1",
    )


def _session(assistant_message):
    session = AgentSession()
    session.add_output(
        AgentOutput(
            execution_id="earlier",
            status=ExecutionStatus.COMPLETED,
            content=assistant_message["content"],
            messages=[
                {"role": "user", "content": "earlier question"},
                assistant_message,
            ],
        )
    )
    return session


def _first_request_messages(model, session):
    with (
        patch("litellm.supports_reasoning", return_value=True),
        patch(
            "agentic.agent.agent.litellm.completion", return_value=_answer()
        ) as completion,
        patch.dict(
            "os.environ",
            {"AGENT_LLM_STREAMING_ENABLED": "false", "OPENAI_REASONING_SUMMARY": ""},
        ),
    ):
        agent = Agent(model=model, reasoning_effort="high")
        output = agent.run("next question", session=session, context=ExecutionContext())
    assert output.status.is_success()
    return completion.call_args_list[0].kwargs["messages"]


def _history_answer(messages):
    matches = [
        m
        for m in messages
        if m.get("role") == "assistant" and m.get("content") == "earlier answer"
    ]
    assert len(matches) == 1
    return matches[0]


def _anthropic_session():
    return _session(
        {
            "role": "assistant",
            "content": "earlier answer",
            "thinking_blocks": [dict(b) for b in _BLOCKS],
            "reasoning": {"provider": "anthropic", "thinking_blocks": _BLOCKS},
        }
    )


def test_claude_reasoning_is_not_replayed_to_another_provider():
    messages = _first_request_messages("openai/gpt-5.4", _anthropic_session())
    _history_answer(messages)
    for message in messages:
        assert "thinking_blocks" not in message


def test_claude_reasoning_is_replayed_to_claude():
    messages = _first_request_messages(
        "anthropic/claude-opus-4-8", _anthropic_session()
    )
    assert _history_answer(messages)["thinking_blocks"] == _BLOCKS


def test_openai_reasoning_items_never_cross_runs():
    session = _session(
        {
            "role": "assistant",
            "content": "earlier answer",
            "reasoning_items": [dict(i) for i in _ITEMS],
            "reasoning": {"provider": "openai", "reasoning_items": _ITEMS},
        }
    )
    messages = _first_request_messages("openai/gpt-5.4", session)
    _history_answer(messages)
    for message in messages:
        assert "reasoning_items" not in message
