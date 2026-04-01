from types import SimpleNamespace
from unittest.mock import patch

from agentic.agent.compaction import (
    compact_messages,
    estimate_token_count,
    get_context_threshold,
    prune_messages,
)


class TestEstimateTokenCount:
    def test_simple_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        count = estimate_token_count(messages)
        assert count > 0
        assert count < 100

    def test_empty_messages(self):
        assert estimate_token_count([]) == 0

    def test_none_content(self):
        messages = [{"role": "assistant", "content": None}]
        assert estimate_token_count(messages) == 0


class TestCompactMessages:
    @patch("agentic.agent.compaction.litellm")
    def test_compacts_to_summary(self, mock_litellm):
        mock_litellm.completion.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Summary: user asked about weather, assistant answered sunny."
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=50, completion_tokens=20, total_tokens=70
            ),
        )

        messages = [
            {"role": "system", "content": "You are a weather bot."},
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "It's sunny and 75F."},
            {"role": "user", "content": "What about tomorrow?"},
            {"role": "assistant", "content": "Tomorrow will be rainy."},
            {"role": "user", "content": "Thanks!"},
        ]

        result = compact_messages(messages, keep_last_n=2)

        assert result[0]["role"] == "system"
        assert "weather bot" in result[0]["content"]
        summary_msgs = [m for m in result if "summary" in m.get("content", "").lower()]
        assert len(summary_msgs) > 0
        assert result[-1]["content"] == "Thanks!"

    @patch("agentic.agent.compaction.litellm")
    def test_preserves_system_message(self, mock_litellm):
        mock_litellm.completion.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Summary of conversation."),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=50, completion_tokens=20, total_tokens=70
            ),
        )

        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "resp1"},
            {"role": "user", "content": "msg2"},
            {"role": "assistant", "content": "resp2"},
            {"role": "user", "content": "msg3"},
        ]

        result = compact_messages(messages, keep_last_n=2)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "System prompt."
        assert result[-1]["role"] == "user"
        assert result[-1]["content"] == "msg3"

    def test_short_messages_not_compacted(self):
        messages = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = compact_messages(messages, keep_last_n=2)
        assert result == messages

    @patch("agentic.agent.compaction.litellm")
    def test_compaction_failure_returns_original(self, mock_litellm):
        mock_litellm.completion.side_effect = Exception("API error")

        messages = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "m1"},
            {"role": "assistant", "content": "r1"},
            {"role": "user", "content": "m2"},
            {"role": "assistant", "content": "r2"},
            {"role": "user", "content": "m3"},
        ]

        result = compact_messages(messages, keep_last_n=2)
        assert result == messages  # Fallback to original on error


class TestPruneMessages:
    def test_prune_removes_old_tool_results(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "q1"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "x" * 10000},
            {"role": "assistant", "content": "answer 1"},
            {"role": "user", "content": "q2"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc2",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc2", "content": "y" * 10000},
            {"role": "assistant", "content": "answer 2"},
        ]
        pruned = prune_messages(messages, keep_last_n_turns=1)
        tc1_result = next(m for m in pruned if m.get("tool_call_id") == "tc1")
        assert "[Previous tool result removed" in tc1_result["content"]
        assert len(tc1_result["content"]) < 100
        tc2_result = next(m for m in pruned if m.get("tool_call_id") == "tc2")
        assert tc2_result["content"] == "y" * 10000

    def test_prune_preserves_system_message(self):
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "q1"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "t", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "big result"},
            {"role": "assistant", "content": "done"},
        ]
        pruned = prune_messages(messages, keep_last_n_turns=0)
        assert pruned[0]["role"] == "system"
        assert pruned[0]["content"] == "System prompt."

    def test_prune_preserves_user_and_assistant(self):
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "t", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "result"},
            {"role": "assistant", "content": "world"},
        ]
        pruned = prune_messages(messages, keep_last_n_turns=0)
        user_msgs = [m for m in pruned if m["role"] == "user"]
        assistant_msgs = [m for m in pruned if m["role"] == "assistant"]
        assert len(user_msgs) == 1
        assert len(assistant_msgs) == 2

    def test_prune_no_tool_results_unchanged(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        pruned = prune_messages(messages, keep_last_n_turns=1)
        assert pruned == messages


class TestGetContextThreshold:
    def test_returns_threshold(self):
        threshold = get_context_threshold("claude-sonnet-4-6")
        assert isinstance(threshold, int)
        assert threshold > 0

    def test_default_for_unknown_model(self):
        threshold = get_context_threshold("unknown-model-xyz")
        assert isinstance(threshold, int)
        assert threshold > 0
