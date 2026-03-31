"""Tests for the Agent ReAct loop with tool calling."""

from types import SimpleNamespace
from unittest.mock import patch

from agentic import Agent
from agentic.agent.tools import BuiltinTool
from agentic.execution.context import ExecutionContext


def _mock_completion_response(content, finish_reason="stop"):
    """Build a mock litellm response."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    role="assistant",
                    tool_calls=None,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )


def _mock_tool_call_response(tool_name, arguments, call_id="call_1"):
    """Build a mock litellm response with a tool call."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    role="assistant",
                    tool_calls=[
                        SimpleNamespace(
                            id=call_id,
                            type="function",
                            function=SimpleNamespace(
                                name=tool_name,
                                arguments=arguments,
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )


class TestReactLoopNoTools:
    """Without tools, the loop should run one iteration (backward compat)."""

    @patch("agentic.agent.agent.litellm")
    def test_single_step_no_tools(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_completion_response("Hello!")
        agent = Agent(model="gpt-4o-mini", system_prompt="You are helpful.")
        output = agent.run("Hi")
        assert output.content == "Hello!"
        assert output.status.is_success()
        assert output.steps == 1
        assert output.tool_calls == []
        mock_litellm.completion.assert_called_once()


class TestReactLoopWithTools:
    """LLM calls a tool, gets result, then produces final response."""

    @patch("agentic.agent.agent.litellm")
    def test_tool_call_then_response(self, mock_litellm):
        # Step 1: LLM wants to call a tool
        tool_call_response = _mock_tool_call_response("get_time", "{}")
        # Step 2: LLM produces final response
        final_response = _mock_completion_response("The time is 3pm.")

        mock_litellm.completion.side_effect = [tool_call_response, final_response]

        tool = BuiltinTool(
            name="get_time",
            description="Get current time",
            input_schema={"type": "object", "properties": {}},
            handler=lambda args, ctx: "15:00",
        )

        agent = Agent(model="gpt-4o-mini", system_prompt="You are helpful.")
        output = agent.run("What time is it?", tools={"get_time": tool})

        assert output.content == "The time is 3pm."
        assert output.steps == 2
        assert len(output.tool_calls) == 1
        assert output.tool_calls[0].tool_name == "get_time"
        assert output.tool_calls[0].result == "15:00"
        assert mock_litellm.completion.call_count == 2


class TestReactLoopSafety:
    """Max steps and doom loop detection."""

    @patch("agentic.agent.agent.litellm")
    def test_max_steps_enforced(self, mock_litellm):
        # LLM always wants to call the tool
        tool_call_response = _mock_tool_call_response("noop", "{}")
        final_response = _mock_completion_response("I ran out of steps.")

        # 2 tool-calling steps + 1 final (max_steps=3)
        mock_litellm.completion.side_effect = [
            tool_call_response,
            tool_call_response,
            final_response,
        ]

        tool = BuiltinTool(
            name="noop",
            description="Does nothing",
            input_schema={"type": "object"},
            handler=lambda args, ctx: "ok",
        )

        agent = Agent(model="gpt-4o-mini", system_prompt="You are helpful.")
        output = agent.run("Do stuff", tools={"noop": tool}, max_steps=3)

        assert output.steps == 3
        # On the last step, tools should NOT be passed to litellm
        last_call_kwargs = mock_litellm.completion.call_args_list[-1]
        assert last_call_kwargs.kwargs.get("tools") is None

    @patch("agentic.agent.agent.litellm")
    def test_doom_loop_detection(self, mock_litellm):
        def make_tool_call():
            return _mock_tool_call_response("search", '{"q": "same"}')

        mock_litellm.completion.side_effect = [
            make_tool_call(),
            make_tool_call(),
            make_tool_call(),
        ]

        tool = BuiltinTool(
            name="search",
            description="Search",
            input_schema={
                "type": "object",
                "properties": {"q": {"type": "string"}},
            },
            handler=lambda args, ctx: "no results",
        )

        agent = Agent(model="gpt-4o-mini", system_prompt="You are helpful.")
        output = agent.run("Find it", tools={"search": tool}, max_steps=25)

        assert output.status.value == "failed"
        assert (
            "doom loop" in output.error.lower() or "identical" in output.error.lower()
        )


class TestReactLoopEvents:
    """Event emission during the ReAct loop."""

    @patch("agentic.agent.agent.litellm")
    def test_events_emitted(self, mock_litellm):
        tool_call_response = _mock_tool_call_response("ping", "{}")
        final_response = _mock_completion_response("pong")
        mock_litellm.completion.side_effect = [tool_call_response, final_response]

        events = []
        ctx = ExecutionContext(execution_id="test", on_event=lambda e: events.append(e))
        tool = BuiltinTool(
            name="ping",
            description="Ping",
            input_schema={"type": "object"},
            handler=lambda args, ctx: "pong",
        )

        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("test", context=ctx, tools={"ping": tool})

        event_types = [e["type"] for e in events]
        assert "step_started" in event_types
        assert "tool_call" in event_types
        assert "tool_result" in event_types
        assert "step_completed" in event_types

        # output.events must be populated and match the callback's event list
        assert len(output.events) > 0
        output_event_types = [e["type"] for e in output.events]
        assert "step_started" in output_event_types
        assert "tool_call" in output_event_types
        assert "tool_result" in output_event_types
        assert "step_completed" in output_event_types
        assert output.events == events
