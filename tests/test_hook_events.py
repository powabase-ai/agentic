"""Tests for OnRunStart, OnRunComplete, and OnDelegation hook firing points."""

from types import SimpleNamespace
from unittest.mock import patch

from agentic import Agent
from agentic.agent.hooks import HookConfig
from agentic.execution.context import ExecutionContext


def _mock_response(content, finish_reason="stop"):
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


class TestOnRunStartFiring:
    @patch("agentic.agent.agent.litellm")
    def test_on_run_start_hook_blocks_run(self, mock_litellm):
        """An OnRunStart hook with deny should prevent the run from executing."""
        mock_litellm.completion.return_value = _mock_response("Should not reach here")

        hooks = [
            HookConfig(
                event="OnRunStart",
                matcher=None,
                type="rule",
                config={
                    "condition": "message CONTAINS 'blocked'",
                    "action": "deny",
                    "message": "Run blocked at start",
                },
            )
        ]

        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("blocked input", hooks=hooks)

        # If OnRunStart fires and blocks, the LLM should not be called
        assert output.status.value == "failed"
        assert "Run blocked at start" in output.error
        mock_litellm.completion.assert_not_called()

    @patch("agentic.agent.agent.litellm")
    def test_on_run_start_hook_allows(self, mock_litellm):
        """An OnRunStart hook that doesn't match should let the run proceed."""
        mock_litellm.completion.return_value = _mock_response("Hello!")

        hooks = [
            HookConfig(
                event="OnRunStart",
                matcher=None,
                type="rule",
                config={
                    "condition": "message CONTAINS 'blocked'",
                    "action": "deny",
                    "message": "Blocked",
                },
            )
        ]

        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("normal input", hooks=hooks)

        assert output.status.is_success()
        assert output.content == "Hello!"
        mock_litellm.completion.assert_called_once()


class TestOnRunCompleteFiring:
    @patch("agentic.agent.agent.litellm")
    def test_on_run_complete_fires_after_success(self, mock_litellm):
        """OnRunComplete hook should fire after a successful run. Since it's fire-and-forget,
        we verify it fires by checking events."""
        mock_litellm.completion.return_value = _mock_response("Done!")

        events = []
        ctx = ExecutionContext(execution_id="test", on_event=lambda e: events.append(e))

        # Use an HTTP hook that we can detect via events (since rule hooks don't emit events).
        # Actually, let's just check that the agent emits an on_run_complete event.
        hooks = [
            HookConfig(
                event="OnRunComplete",
                matcher=None,
                type="rule",
                config={},  # Empty rule — won't deny anything
            )
        ]

        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("hello", context=ctx, hooks=hooks)

        assert output.status.is_success()
        # Check that an on_run_complete event was emitted
        event_types = [e["type"] for e in events]
        assert "on_run_complete" in event_types


class TestOnDelegationFiring:
    @patch("agentic.agent.agent.litellm")
    def test_on_delegation_hook_blocks_delegation(self, mock_litellm):
        """An OnDelegation hook should fire before DelegateTool executes."""
        # Orchestrator calls delegate_to_specialist
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
                                    name="delegate_to_specialist",
                                    arguments='{"task": "delete everything"}',
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
        final_resp = _mock_response("Delegation was blocked.")
        mock_litellm.completion.side_effect = [tool_call_resp, final_resp]

        from agentic.agent.tools import DelegateTool

        sub_agent = Agent(model="gpt-4o-mini", system_prompt="specialist")
        delegate = DelegateTool(
            name="delegate_to_specialist",
            description="Delegate to specialist",
            agent=sub_agent,
        )

        hooks = [
            HookConfig(
                event="OnDelegation",
                matcher="delegate_to_specialist",
                type="rule",
                config={
                    "condition": "task CONTAINS 'delete'",
                    "action": "deny",
                    "message": "Dangerous delegation blocked",
                },
            )
        ]

        agent = Agent(model="gpt-4o-mini", system_prompt="orchestrator")
        output = agent.run(
            "do it", tools={"delegate_to_specialist": delegate}, hooks=hooks
        )

        assert output.status.is_success()
        # The delegation should have been blocked — sub_agent's LLM should NOT have been called
        # (only 2 calls: orchestrator's tool_call + orchestrator's final response)
        assert mock_litellm.completion.call_count == 2
