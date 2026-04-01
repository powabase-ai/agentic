"""Tests for DelegateTool - sub-agent delegation in orchestrations."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agentic import Agent
from agentic.agent.tools import BuiltinTool, DelegateTool
from agentic.execution.context import ExecutionContext, MaxDepthExceeded


def _mock_completion_response(content, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content, role="assistant", tool_calls=None
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


class TestDelegateTool:
    def test_create(self):
        agent = Agent(model="gpt-4o-mini", system_prompt="You are a specialist.")
        tool = DelegateTool(
            name="delegate_to_specialist",
            description="Delegate to the specialist agent",
            agent=agent,
        )
        assert tool.name == "delegate_to_specialist"
        assert tool.input_schema["properties"]["task"]["type"] == "string"

    def test_to_function_schema(self):
        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        tool = DelegateTool(
            name="delegate_to_claims",
            description="Handles insurance claims",
            agent=agent,
        )
        schema = tool.to_function_schema()
        assert schema["function"]["name"] == "delegate_to_claims"
        assert "task" in schema["function"]["parameters"]["properties"]

    @patch("agentic.agent.agent.litellm")
    def test_execute_runs_sub_agent(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_completion_response(
            "Claim approved."
        )

        sub_agent = Agent(model="gpt-4o-mini", system_prompt="You handle claims.")
        tool = DelegateTool(
            name="delegate_to_claims",
            description="Claims specialist",
            agent=sub_agent,
        )

        ctx = ExecutionContext(execution_id="parent", depth=0, max_depth=3)
        result = tool.execute({"task": "Check claim #123"}, ctx)

        assert result == "Claim approved."
        mock_litellm.completion.assert_called_once()

    @patch("agentic.agent.agent.litellm")
    def test_execute_with_agent_tools(self, mock_litellm):
        """Sub-agent should have access to its own tools."""
        tool_call_resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        role="assistant",
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                type="function",
                                function=SimpleNamespace(
                                    name="lookup", arguments='{"id": "123"}'
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )
        final_resp = _mock_completion_response("Customer found: John Doe")
        mock_litellm.completion.side_effect = [tool_call_resp, final_resp]

        sub_agent = Agent(model="gpt-4o-mini", system_prompt="You look up customers.")
        lookup_tool = BuiltinTool(
            name="lookup",
            description="Look up customer",
            input_schema={
                "type": "object",
                "properties": {"id": {"type": "string"}},
            },
            handler=lambda args, ctx: "John Doe, Policy #456",
        )

        delegate = DelegateTool(
            name="delegate_to_lookup",
            description="Customer lookup agent",
            agent=sub_agent,
            agent_tools={"lookup": lookup_tool},
        )

        ctx = ExecutionContext(execution_id="parent", depth=0, max_depth=3)
        result = delegate.execute({"task": "Find customer 123"}, ctx)

        assert "John Doe" in result
        assert mock_litellm.completion.call_count == 2

    @patch("agentic.agent.agent.litellm")
    def test_execute_emits_events(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_completion_response("Done.")

        sub_agent = Agent(model="gpt-4o-mini", system_prompt="test")
        tool = DelegateTool(name="delegate", description="test", agent=sub_agent)

        events = []
        ctx = ExecutionContext(
            execution_id="parent",
            depth=0,
            max_depth=3,
            on_event=lambda e: events.append(e),
        )
        tool.execute({"task": "do it"}, ctx)

        event_types = [e["type"] for e in events]
        assert "delegation_started" in event_types
        assert "delegation_completed" in event_types

    def test_execute_respects_depth_limit(self):
        sub_agent = Agent(model="gpt-4o-mini", system_prompt="test")
        tool = DelegateTool(name="delegate", description="test", agent=sub_agent)

        ctx = ExecutionContext(execution_id="deep", depth=3, max_depth=3)
        with pytest.raises(MaxDepthExceeded):
            tool.execute({"task": "do it"}, ctx)
