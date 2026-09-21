import copy
import json

import pytest

from agentic.agent.cache import (
    add_cache_breakpoints,
    sort_tools_for_cache,
    uses_explicit_cache_breakpoints,
)
from agentic.agent.tools import (
    BuiltinTool,
    CustomTool,
    DelegateTool,
    KnowledgeSearchTool,
)


class TestSortToolsForCache:
    def test_builtins_before_custom(self):
        custom = CustomTool(
            name="z_custom", description="", input_schema={}, endpoint=""
        )
        builtin = BuiltinTool(
            name="a_builtin", description="", input_schema={}, handler=lambda a, c: ""
        )
        result = sort_tools_for_cache({"z_custom": custom, "a_builtin": builtin})
        assert result[0].name == "a_builtin"
        assert result[1].name == "z_custom"

    def test_alphabetical_within_type(self):
        b1 = BuiltinTool(
            name="zebra", description="", input_schema={}, handler=lambda a, c: ""
        )
        b2 = BuiltinTool(
            name="alpha", description="", input_schema={}, handler=lambda a, c: ""
        )
        result = sort_tools_for_cache({"zebra": b1, "alpha": b2})
        assert result[0].name == "alpha"
        assert result[1].name == "zebra"

    def test_full_priority_order(self):
        from unittest.mock import MagicMock

        builtin = BuiltinTool(
            name="db", description="", input_schema={}, handler=lambda a, c: ""
        )
        kb = KnowledgeSearchTool(name="search_docs", description="")
        custom = CustomTool(name="api", description="", input_schema={}, endpoint="")
        delegate = DelegateTool(name="delegate_to_x", description="", agent=MagicMock())
        tools = {
            "delegate_to_x": delegate,
            "api": custom,
            "search_docs": kb,
            "db": builtin,
        }
        result = sort_tools_for_cache(tools)
        names = [t.name for t in result]
        assert names == ["db", "search_docs", "api", "delegate_to_x"]

    def test_deterministic_across_calls(self):
        b1 = BuiltinTool(
            name="a", description="", input_schema={}, handler=lambda a, c: ""
        )
        b2 = BuiltinTool(
            name="b", description="", input_schema={}, handler=lambda a, c: ""
        )
        tools = {"b": b2, "a": b1}
        r1 = [t.name for t in sort_tools_for_cache(tools)]
        r2 = [t.name for t in sort_tools_for_cache(tools)]
        assert r1 == r2

    def test_empty_tools(self):
        assert sort_tools_for_cache({}) == []


_EPHEMERAL = {"type": "ephemeral"}


def _tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {"name": "alpha", "description": "A", "parameters": {}},
        },
        {
            "type": "function",
            "function": {"name": "beta", "description": "B", "parameters": {}},
        },
    ]


def _history() -> list[dict]:
    return [
        {"role": "system", "content": "You are a bot."},
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "alpha", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
    ]


def _count_markers(messages: list[dict], tools: list[dict] | None) -> int:
    """Every `cache_control` key anywhere in the request — each one spends
    one of the provider's (at most 4) breakpoints."""
    return json.dumps(messages).count('"cache_control"') + json.dumps(
        tools or []
    ).count('"cache_control"')


class TestUsesExplicitCacheBreakpoints:
    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-8",
            "anthropic/claude-opus-4-8",
            "vertex_ai/claude-opus-4-8",
            "bedrock/us.anthropic.claude-opus-4-8",
            "bedrock/global.anthropic.claude-opus-4-6-v1",
        ],
    )
    def test_claude_on_anthropic_vertex_and_bedrock(self, model):
        assert uses_explicit_cache_breakpoints(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            # Bedrock Claude models without prompt caching: whether Bedrock
            # rejects a cachePoint for them is unverified, so they are left
            # unmarked (the gate fails closed on Bedrock).
            "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        ],
    )
    def test_bedrock_claude_without_prompt_caching(self, model):
        assert uses_explicit_cache_breakpoints(model) is False

    @pytest.mark.parametrize(
        "model",
        [
            # Providers that cache automatically, or whose pass-through of
            # cache_control is unverified — an unknown key can 400 there.
            "gpt-5.4",
            "openai/responses/gpt-5.4",
            "openrouter/deepseek/deepseek-v4-pro",
            "openrouter/anthropic/claude-opus-4-8",
            "gemini/gemini-2.5-pro",
            "vertex_ai/gemini-2.5-pro",
            "bedrock/amazon.nova-pro-v1:0",
        ],
    )
    def test_other_routes(self, model):
        assert uses_explicit_cache_breakpoints(model) is False

    def test_unresolvable_model_is_false(self):
        assert uses_explicit_cache_breakpoints("no-such-provider-model-xyz") is False


class TestAddCacheBreakpoints:
    def test_non_claude_model_sends_no_markers(self):
        messages, tools = add_cache_breakpoints("gpt-5.4", _history(), _tools())
        assert messages == _history()
        assert tools == _tools()
        assert _count_markers(messages, tools) == 0

    def test_claude_marks_last_tool_only(self):
        _, tools = add_cache_breakpoints("claude-opus-4-8", _history(), _tools())
        assert tools[-1]["cache_control"] == _EPHEMERAL
        assert "cache_control" not in tools[0]

    def test_claude_marks_system_and_last_message_only(self):
        messages, _ = add_cache_breakpoints("claude-opus-4-8", _history(), _tools())
        assert messages[0]["cache_control"] == _EPHEMERAL
        assert messages[-1]["cache_control"] == _EPHEMERAL
        for middle in messages[1:-1]:
            assert "cache_control" not in json.dumps(middle)

    def test_three_breakpoints_leaves_one_spare(self):
        messages, tools = add_cache_breakpoints("claude-opus-4-8", _history(), _tools())
        assert _count_markers(messages, tools) == 3

    def test_marked_content_is_otherwise_identical(self):
        # The breakpoint must be the only difference — any other change to
        # the prefix bytes would stop the cached prefix from ever matching.
        messages, tools = add_cache_breakpoints("claude-opus-4-8", _history(), _tools())
        assert [
            {k: v for k, v in m.items() if k != "cache_control"} for m in messages
        ] == _history()
        assert [
            {k: v for k, v in t.items() if k != "cache_control"} for t in tools
        ] == _tools()

    def test_inputs_are_not_mutated(self):
        history, tools = _history(), _tools()
        history[0]["content"] = [{"type": "text", "text": "You are a bot."}]
        before = (copy.deepcopy(history), copy.deepcopy(tools))
        add_cache_breakpoints("claude-opus-4-8", history, tools)
        assert (history, tools) == before

    def test_list_content_marks_a_copy_of_the_last_block(self):
        history = _history()
        history[0]["content"] = [
            {"type": "text", "text": "part one"},
            {"type": "text", "text": "part two"},
        ]
        history.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                    {"type": "text", "text": "what is this?"},
                ],
            }
        )
        messages, _ = add_cache_breakpoints("claude-opus-4-8", history, _tools())
        assert messages[0]["content"][-1]["cache_control"] == _EPHEMERAL
        assert "cache_control" not in messages[0]["content"][0]
        assert "cache_control" not in messages[0]
        assert messages[-1]["content"][-1]["cache_control"] == _EPHEMERAL
        assert "cache_control" not in messages[-1]["content"][0]

    def test_tool_message_with_list_content_is_marked_at_message_level(self):
        # A tool message's breakpoint belongs on the tool_result block
        # itself, which is what the message-level key maps to.
        history = _history()
        history[-1]["content"] = [{"type": "text", "text": "result"}]
        messages, _ = add_cache_breakpoints("claude-opus-4-8", history, _tools())
        assert messages[-1]["cache_control"] == _EPHEMERAL
        assert "cache_control" not in messages[-1]["content"][0]

    def test_no_system_message(self):
        messages, tools = add_cache_breakpoints(
            "claude-opus-4-8", _history()[1:], _tools()
        )
        assert "cache_control" not in messages[0]
        assert messages[-1]["cache_control"] == _EPHEMERAL
        assert _count_markers(messages, tools) == 2

    def test_no_tools(self):
        messages, tools = add_cache_breakpoints("claude-opus-4-8", _history(), None)
        assert tools is None
        assert _count_markers(messages, tools) == 2

    @staticmethod
    def _with_caller_markers(count: int) -> list[dict]:
        """History whose earlier user turns already carry block-level
        breakpoints, as a caller managing its own caching would send."""
        history = _history()
        for i in range(count):
            history.insert(
                1 + 2 * i,
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"doc {i}",
                            "cache_control": _EPHEMERAL,
                        }
                    ],
                },
            )
            history.insert(2 + 2 * i, {"role": "assistant", "content": "noted"})
        return history

    def test_caller_markers_count_toward_the_limit(self):
        # 2 caller markers leave room for 2: the moving history breakpoint and
        # the system prompt. The tools breakpoint goes first — a read at the
        # system breakpoint already covers the tools prefix.
        messages, tools = add_cache_breakpoints(
            "claude-opus-4-8", self._with_caller_markers(2), _tools()
        )
        assert _count_markers(messages, tools) == 4
        assert "cache_control" not in tools[-1]
        assert messages[0]["cache_control"] == _EPHEMERAL
        assert messages[-1]["cache_control"] == _EPHEMERAL

    def test_three_caller_markers_leave_room_for_the_history_breakpoint(self):
        messages, tools = add_cache_breakpoints(
            "claude-opus-4-8", self._with_caller_markers(3), _tools()
        )
        assert _count_markers(messages, tools) == 4
        assert "cache_control" not in messages[0]
        assert messages[-1]["cache_control"] == _EPHEMERAL

    def test_caller_marker_on_the_newest_block_is_kept_and_costs_no_slot(self):
        # The caller already marked the block the history breakpoint would go
        # on: keep theirs (with its TTL) rather than overwrite or add another.
        history = _history()[:2]
        history[-1] = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "question",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
        }
        messages, tools = add_cache_breakpoints("claude-opus-4-8", history, _tools())
        assert messages[-1]["content"][-1]["cache_control"] == {
            "type": "ephemeral",
            "ttl": "1h",
        }
        assert messages[0]["cache_control"] == _EPHEMERAL
        assert tools[-1]["cache_control"] == _EPHEMERAL
        assert _count_markers(messages, tools) == 3

    def test_four_caller_markers_add_nothing(self):
        history, tools = self._with_caller_markers(4), _tools()
        messages, tools_out = add_cache_breakpoints("claude-opus-4-8", history, tools)
        assert _count_markers(messages, tools_out) == 4
        assert messages == history
        assert tools_out == tools

    def test_single_system_message_is_marked_once(self):
        messages, _ = add_cache_breakpoints(
            "claude-opus-4-8", [{"role": "system", "content": "s"}], None
        )
        assert messages == [
            {"role": "system", "content": "s", "cache_control": _EPHEMERAL}
        ]
