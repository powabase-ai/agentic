"""Tests that AgentOutput carries reasoning_artifact and reasoning_requested."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agentic.agent.agent import Agent
from agentic.agent.message import AnthropicReasoning
from agentic.execution.context import ExecutionContext


def _make_thinking_response():
    msg = MagicMock()
    msg.content = "the answer"
    msg.tool_calls = None
    msg.reasoning_content = "summary"
    msg.thinking_blocks = [{"type": "thinking", "thinking": "x", "signature": "s"}]
    msg.provider_specific_fields = None
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    response = MagicMock()
    response.choices = [choice]
    response.usage = SimpleNamespace(
        prompt_tokens=20, completion_tokens=100, total_tokens=120
    )
    response.id = "rid"
    return response


def test_output_carries_reasoning_artifact():
    with (
        patch("litellm.supports_reasoning", return_value=True),
        patch(
            "agentic.agent.agent.litellm.completion",
            return_value=_make_thinking_response(),
        ),
        patch(
            "agentic.llm.reasoning_extractor.litellm.get_llm_provider",
            return_value=("claude-opus-4-7", "anthropic", None, None),
        ),
        patch.dict("os.environ", {"AGENT_LLM_STREAMING_ENABLED": "false"}),
    ):
        agent = Agent(model="anthropic/claude-opus-4-7", reasoning_effort="medium")
        output = agent.run("hi", context=ExecutionContext())

    assert output.reasoning_artifact is not None
    assert isinstance(output.reasoning_artifact, AnthropicReasoning)
    assert output.reasoning_requested is True


def test_output_reasoning_requested_false_when_no_effort():
    msg = MagicMock()
    msg.content = "answer"
    msg.tool_calls = None
    msg.reasoning_content = None
    msg.thinking_blocks = None
    msg.provider_specific_fields = None
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = SimpleNamespace(
        prompt_tokens=10, completion_tokens=10, total_tokens=20
    )

    with (
        patch("agentic.agent.agent.litellm.completion", return_value=resp),
        patch.dict("os.environ", {"AGENT_LLM_STREAMING_ENABLED": "false"}),
    ):
        agent = Agent(model="anthropic/claude-opus-4-7")
        output = agent.run("hi", context=ExecutionContext())

    assert output.reasoning_artifact is None
    assert output.reasoning_requested is False
