"""Tests that the assistant msg_dict gets a reasoning field attached when an
artifact extracts (intra-run replay — the Anthropic 400 fix)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agentic.agent.agent import Agent
from agentic.execution.context import ExecutionContext


def _make_response_with_thinking():
    msg = MagicMock()
    msg.content = "answer"
    msg.tool_calls = None
    msg.reasoning_content = "I considered X."
    msg.thinking_blocks = [
        {"type": "thinking", "thinking": "I considered X.", "signature": "sig1"}
    ]
    msg.provider_specific_fields = None
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    response = MagicMock()
    response.choices = [choice]
    response.usage = SimpleNamespace(
        prompt_tokens=10, completion_tokens=200, total_tokens=210
    )
    response.id = "resp1"
    return response


def test_anthropic_msg_dict_carries_reasoning_field():
    """After a Claude run with thinking, state.messages assistant entry has
    a `reasoning` field with the discriminated artifact dump."""
    with (
        patch("litellm.supports_reasoning", return_value=True),
        patch(
            "agentic.agent.agent.litellm.completion",
            return_value=_make_response_with_thinking(),
        ),
        patch(
            "agentic.llm.reasoning_extractor.litellm.get_llm_provider",
            return_value=("claude-opus-4-7", "anthropic", None, None),
        ),
        patch.dict("os.environ", {"AGENT_LLM_STREAMING_ENABLED": "false"}),
    ):
        agent = Agent(model="anthropic/claude-opus-4-7", reasoning_effort="medium")
        output = agent.run("hi", context=ExecutionContext())

    # state.messages is exposed via output.messages
    assistant_msgs = [m for m in output.messages if m.get("role") == "assistant"]
    assert len(assistant_msgs) == 1
    msg = assistant_msgs[0]
    assert "reasoning" in msg
    assert msg["reasoning"]["provider"] == "anthropic"
    assert msg["reasoning"]["thinking_blocks"] == [
        {"type": "thinking", "thinking": "I considered X.", "signature": "sig1"}
    ]


def test_no_artifact_means_no_reasoning_key():
    """Run with no reasoning_effort → no reasoning field on msg_dict."""
    msg = MagicMock()
    msg.content = "answer"
    msg.tool_calls = None
    msg.reasoning_content = None
    msg.thinking_blocks = None
    msg.provider_specific_fields = None
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    response = MagicMock()
    response.choices = [choice]
    response.usage = None
    response.id = None

    with (
        patch("agentic.agent.agent.litellm.completion", return_value=response),
        patch.dict("os.environ", {"AGENT_LLM_STREAMING_ENABLED": "false"}),
    ):
        agent = Agent(model="anthropic/claude-opus-4-7")  # no reasoning_effort
        output = agent.run("hi", context=ExecutionContext())

    assistant_msgs = [m for m in output.messages if m.get("role") == "assistant"]
    assert "reasoning" not in assistant_msgs[0]
