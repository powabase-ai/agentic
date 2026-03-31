from types import SimpleNamespace
from unittest.mock import patch

from agentic import Agent


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


class TestModelFallback:
    @patch("agentic.agent.agent.litellm")
    def test_fallback_on_rate_limit(self, mock_litellm):
        from litellm.exceptions import RateLimitError

        mock_litellm.completion.side_effect = [
            RateLimitError(
                message="rate limited",
                llm_provider="anthropic",
                model="claude-sonnet-4-6",
            ),
            _mock_completion_response("Fallback response."),
        ]
        agent = Agent(model="claude-sonnet-4-6", system_prompt="test")
        output = agent.run("hello", fallback_model="gpt-4.1-mini")
        assert output.content == "Fallback response."
        assert output.status.is_success()
        assert mock_litellm.completion.call_count == 2
        second_call = mock_litellm.completion.call_args_list[1]
        assert second_call.kwargs["model"] == "gpt-4.1-mini"

    @patch("agentic.agent.agent.litellm")
    def test_no_fallback_without_config(self, mock_litellm):
        from litellm.exceptions import RateLimitError

        mock_litellm.completion.side_effect = RateLimitError(
            message="rate limited", llm_provider="anthropic", model="claude-sonnet-4-6"
        )
        agent = Agent(model="claude-sonnet-4-6", system_prompt="test")
        output = agent.run("hello")
        assert output.status.value == "failed"

    @patch("agentic.agent.agent.litellm")
    def test_fallback_is_free_turn(self, mock_litellm):
        from litellm.exceptions import RateLimitError

        mock_litellm.completion.side_effect = [
            RateLimitError(
                message="rate limited",
                llm_provider="anthropic",
                model="claude-sonnet-4-6",
            ),
            _mock_completion_response("ok"),
        ]
        agent = Agent(model="claude-sonnet-4-6", system_prompt="test")
        output = agent.run("hello", fallback_model="gpt-4.1-mini")
        assert output.steps == 1  # Free recovery


class TestMaxOutputRecovery:
    @patch("agentic.agent.agent.litellm")
    def test_truncated_response_retried(self, mock_litellm):
        truncated = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Partial respon", role="assistant", tool_calls=None
                    ),
                    finish_reason="length",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )
        continued = _mock_completion_response("se. Here's the rest.")
        mock_litellm.completion.side_effect = [truncated, continued]

        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("Write something long")

        assert output.status.is_success()
        assert mock_litellm.completion.call_count == 2
        second_messages = mock_litellm.completion.call_args_list[1].kwargs["messages"]
        continue_msgs = [
            m
            for m in second_messages
            if "truncated" in m.get("content", "").lower()
            or "continue" in m.get("content", "").lower()
        ]
        assert len(continue_msgs) > 0

    @patch("agentic.agent.agent.litellm")
    def test_max_3_recovery_attempts(self, mock_litellm):
        truncated = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Partial", role="assistant", tool_calls=None
                    ),
                    finish_reason="length",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )
        mock_litellm.completion.side_effect = [
            truncated,
            truncated,
            truncated,
            truncated,
        ]

        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("Write something long")

        assert mock_litellm.completion.call_count == 4
        assert output.content is not None


class TestReactiveCompact:
    @patch("agentic.agent.agent.litellm")
    def test_reactive_compact_on_prompt_too_long(self, mock_litellm):
        mock_litellm.completion.side_effect = [
            Exception("prompt is too long: 200000 tokens > 100000 maximum"),
            _mock_completion_response("Compacted response."),
        ]
        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("hello")
        assert output.content == "Compacted response."
        assert output.status.is_success()
        assert mock_litellm.completion.call_count == 2

    @patch("agentic.agent.agent.litellm")
    def test_reactive_compact_is_free_turn(self, mock_litellm):
        mock_litellm.completion.side_effect = [
            Exception("prompt is too long"),
            _mock_completion_response("ok"),
        ]
        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("hello")
        assert output.steps == 1
