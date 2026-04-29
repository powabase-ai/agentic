# agentic/tests/test_normalization.py
from agentic.agent.normalization import normalize_messages


class TestStripMetadata:
    def test_strips_internal_fields(self):
        messages = [
            {"role": "user", "content": "hi", "_injected": True, "_debug": "test"},
            {"role": "assistant", "content": "hello"},
        ]
        result = normalize_messages(messages)
        assert "_injected" not in result[0]
        assert "_debug" not in result[0]
        assert result[0]["content"] == "hi"

    def test_preserves_standard_fields(self):
        messages = [
            {
                "role": "assistant",
                "content": "ok",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            },
        ]
        result = normalize_messages(messages)
        assert result[0]["tool_calls"] == messages[0]["tool_calls"]

    def test_preserves_thinking_blocks_for_anthropic_replay(self):
        """thinking_blocks must survive normalization so Anthropic can replay
        prior reasoning blocks on subsequent litellm.completion calls."""
        messages = [
            {
                "role": "assistant",
                "content": "ok",
                "thinking_blocks": [
                    {
                        "type": "thinking",
                        "thinking": "let me think...",
                        "signature": "abc123",
                    }
                ],
            },
        ]
        result = normalize_messages(messages)
        assert result[0]["thinking_blocks"] == messages[0]["thinking_blocks"]

    def test_preserves_provider_specific_fields_for_openai_gemini_replay(self):
        """provider_specific_fields carries OpenAI/Gemini reasoning replay data
        and must not be stripped."""
        messages = [
            {
                "role": "assistant",
                "content": "ok",
                "provider_specific_fields": {
                    "reasoning_content": "internal reasoning trace",
                },
            },
        ]
        result = normalize_messages(messages)
        assert (
            result[0]["provider_specific_fields"]
            == messages[0]["provider_specific_fields"]
        )


class TestOrphanedToolResults:
    def test_removes_orphaned_tool_result(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "tc_orphan", "content": "result"},
            {"role": "assistant", "content": "ok"},
        ]
        result = normalize_messages(messages)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_keeps_paired_tool_result(self):
        messages = [
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
            {"role": "tool", "tool_call_id": "tc1", "content": "found it"},
        ]
        result = normalize_messages(messages)
        assert len(result) == 2


class TestUnavailableToolCalls:
    def test_strips_unavailable_tool_calls(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    },
                    {
                        "id": "tc2",
                        "type": "function",
                        "function": {"name": "removed_tool", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "ok"},
            {"role": "tool", "tool_call_id": "tc2", "content": "ok"},
        ]
        available_tools = {"search"}
        result = normalize_messages(messages, available_tool_names=available_tools)
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["id"] == "tc1"
        tool_results = [m for m in result if m["role"] == "tool"]
        assert len(tool_results) == 1
        assert tool_results[0]["tool_call_id"] == "tc1"

    def test_no_filtering_when_no_available_tools(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "any_tool", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "ok"},
        ]
        result = normalize_messages(messages, available_tool_names=None)
        assert len(result) == 2


class TestSystemContextTagging:
    def test_tags_injected_messages(self):
        messages = [
            {"role": "system", "content": "KB context here", "_injected": True},
            {"role": "user", "content": "hello"},
        ]
        result = normalize_messages(messages)
        assert "<system-context>" in result[0]["content"]
        assert "KB context here" in result[0]["content"]

    def test_does_not_tag_regular_system_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
        ]
        result = normalize_messages(messages)
        assert "<system-context>" not in result[0]["content"]
