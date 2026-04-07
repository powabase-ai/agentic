"""Tests for hard timeout support in Agent.run()."""

import time
from types import SimpleNamespace
from unittest.mock import patch

from agentic import Agent
from agentic.execution.context import ExecutionContext


def _slow_response():
    time.sleep(2)
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Done", role="assistant", tool_calls=None
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


class TestHardTimeout:
    @patch("agentic.agent.agent.litellm")
    def test_timeout_cancels_run(self, mock_litellm):
        mock_litellm.completion.side_effect = lambda **kwargs: _slow_response()
        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("hello", timeout_seconds=0.5)
        assert output.status.value in ("cancelled", "failed")

    @patch("agentic.agent.agent.litellm")
    def test_no_timeout_completes(self, mock_litellm):
        mock_litellm.completion.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Hello!", role="assistant", tool_calls=None
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )
        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("hello")
        assert output.status.is_success()

    @patch("agentic.agent.agent.litellm")
    def test_timeout_uses_existing_abort_signal(self, mock_litellm):
        """When context already has an abort_signal, the timer uses it."""
        import threading

        mock_litellm.completion.side_effect = lambda **kwargs: _slow_response()
        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        existing_event = threading.Event()
        ctx = ExecutionContext(abort_signal=existing_event)
        output = agent.run("hello", context=ctx, timeout_seconds=0.5)
        assert output.status.value in ("cancelled", "failed")
        # The existing event should have been set by the timer
        assert existing_event.is_set()

    @patch("agentic.agent.agent.litellm")
    def test_timer_cancelled_on_normal_completion(self, mock_litellm):
        """Timer is cancelled on normal completion so it does not fire late."""

        mock_litellm.completion.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Hi", role="assistant", tool_calls=None
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )
        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        ctx = ExecutionContext()
        output = agent.run("hello", context=ctx, timeout_seconds=5)
        assert output.status.is_success()
        # After completion the abort signal must NOT be set
        assert not ctx.abort_signal.is_set()
