"""Tests for token budget enforcement in the Agent ReAct loop."""

from types import SimpleNamespace
from unittest.mock import patch

from agentic import Agent
from agentic.agent.tools import BuiltinTool
from agentic.execution.context import ExecutionContext, TokenBudget


def _mock_completion_response(content, finish_reason="stop", total_tokens=15):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content, role="assistant", tool_calls=None
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=5, total_tokens=total_tokens
        ),
    )


def _mock_tool_call_response(tool_name="noop", arguments="{}", total_tokens=100):
    return SimpleNamespace(
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
                                name=tool_name, arguments=arguments
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=80, completion_tokens=20, total_tokens=total_tokens
        ),
    )


class TestBudgetEnforcement:
    @patch("agentic.agent.agent.litellm")
    def test_budget_consumed_per_step(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_completion_response(
            "Hello!", total_tokens=100
        )

        budget = TokenBudget(max_tokens=1000)
        ctx = ExecutionContext(execution_id="test", budget=budget)

        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        agent.run("Hi", context=ctx)

        assert budget.used_tokens == 100
        assert budget.remaining == 900

    @patch("agentic.agent.agent.litellm")
    def test_budget_pacing_message_injected(self, mock_litellm):
        """When budget is nearly exhausted, a pacing message should be injected."""
        resp1 = _mock_tool_call_response(total_tokens=900)
        resp2 = _mock_completion_response("Wrapping up.", total_tokens=50)

        mock_litellm.completion.side_effect = [resp1, resp2]

        budget = TokenBudget(max_tokens=1000)
        ctx = ExecutionContext(execution_id="test", budget=budget)

        tool = BuiltinTool(
            name="noop",
            description="Does nothing",
            input_schema={"type": "object"},
            handler=lambda a, c: "ok",
        )

        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        agent.run("Do stuff", context=ctx, tools={"noop": tool})

        # Second LLM call should have a budget warning in messages
        second_call = mock_litellm.completion.call_args_list[1]
        messages = second_call.kwargs.get(
            "messages", second_call.args[0] if second_call.args else []
        )
        budget_msgs = [
            m
            for m in messages
            if (m.get("content") or "")
            and (
                "budget" in m.get("content", "").lower()
                or "remaining" in m.get("content", "").lower()
            )
        ]
        assert len(budget_msgs) > 0

    @patch("agentic.agent.agent.litellm")
    def test_budget_exceeded_disables_tools(self, mock_litellm):
        """When budget is fully exhausted, tools should be disabled on next step."""
        resp1 = _mock_tool_call_response(total_tokens=1000)
        resp2 = _mock_completion_response("Budget exhausted.", total_tokens=50)

        mock_litellm.completion.side_effect = [resp1, resp2]

        budget = TokenBudget(max_tokens=1000)
        ctx = ExecutionContext(execution_id="test", budget=budget)

        tool = BuiltinTool(
            name="noop",
            description="Does nothing",
            input_schema={"type": "object"},
            handler=lambda a, c: "ok",
        )

        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        agent.run("Do stuff", context=ctx, tools={"noop": tool})

        # Second call should have tools=None
        second_call = mock_litellm.completion.call_args_list[1]
        assert second_call.kwargs.get("tools") is None

    @patch("agentic.agent.agent.litellm")
    def test_no_budget_no_enforcement(self, mock_litellm):
        """Without a budget, no warnings or enforcement."""
        mock_litellm.completion.return_value = _mock_completion_response("Hello!")

        ctx = ExecutionContext(execution_id="test")  # No budget

        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("Hi", context=ctx)

        assert output.status.is_success()
        # Only one call, no budget messages
        assert mock_litellm.completion.call_count == 1
