"""
Tests for structured output (response_format) support in Agent.run().
"""

from unittest.mock import patch

from agentic import Agent


class TestResponseFormat:
    """Tests for response_format parameter in Agent.run()."""

    def test_response_format_passed_to_litellm(self):
        """When response_format is provided, litellm.completion receives it."""
        with patch("agentic.agent.agent.litellm") as mock_litellm:
            from unittest.mock import MagicMock

            mock_message = MagicMock()
            mock_message.content = "Test response"
            mock_choice = MagicMock()
            mock_choice.message = mock_message
            mock_usage = MagicMock()
            mock_usage.prompt_tokens = 10
            mock_usage.completion_tokens = 20
            mock_usage.total_tokens = 30
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.usage = mock_usage
            mock_litellm.completion.return_value = mock_response

            response_format = {"type": "json_object"}
            agent = Agent(system_prompt="You are helpful")
            agent.run("Hello", response_format=response_format)

            call_kwargs = mock_litellm.completion.call_args.kwargs
            assert "response_format" in call_kwargs
            assert call_kwargs["response_format"] == {"type": "json_object"}

    def test_no_response_format_not_included(self):
        """When response_format is None (default), the key is not in call_kwargs."""
        with patch("agentic.agent.agent.litellm") as mock_litellm:
            from unittest.mock import MagicMock

            mock_message = MagicMock()
            mock_message.content = "Test response"
            mock_choice = MagicMock()
            mock_choice.message = mock_message
            mock_usage = MagicMock()
            mock_usage.prompt_tokens = 10
            mock_usage.completion_tokens = 20
            mock_usage.total_tokens = 30
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.usage = mock_usage
            mock_litellm.completion.return_value = mock_response

            agent = Agent(system_prompt="You are helpful")
            agent.run("Hello")  # no response_format

            call_kwargs = mock_litellm.completion.call_args.kwargs
            assert "response_format" not in call_kwargs
