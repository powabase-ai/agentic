"""Tests for concurrent tool execution and safety caps in Agent.run()."""

import time
from types import SimpleNamespace
from unittest.mock import patch

from agentic import Agent
from agentic.agent.tools import BuiltinTool


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


def _mock_multi_tool_response(*tool_names):
    tool_calls = [
        SimpleNamespace(
            id=f"call_{i}",
            type="function",
            function=SimpleNamespace(name=name, arguments="{}"),
        )
        for i, name in enumerate(tool_names)
    ]
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None, role="assistant", tool_calls=tool_calls
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


class TestConcurrentExecution:
    @patch("agentic.agent.agent.litellm")
    def test_concurrent_safe_tools_run_in_parallel(self, mock_litellm):
        mock_litellm.completion.side_effect = [
            _mock_multi_tool_response("slow_a", "slow_b"),
            _mock_completion_response("done"),
        ]

        def slow_handler(args, ctx):
            time.sleep(0.3)
            return "result"

        tool_a = BuiltinTool(
            name="slow_a",
            description="",
            input_schema={"type": "object"},
            handler=slow_handler,
            is_concurrency_safe=True,
        )
        tool_b = BuiltinTool(
            name="slow_b",
            description="",
            input_schema={"type": "object"},
            handler=slow_handler,
            is_concurrency_safe=True,
        )

        start = time.monotonic()
        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("test", tools={"slow_a": tool_a, "slow_b": tool_b})
        elapsed = time.monotonic() - start

        assert output.status.is_success()
        assert (
            elapsed < 0.55
        ), f"Tools ran sequentially ({elapsed:.2f}s), expected parallel"

    @patch("agentic.agent.agent.litellm")
    def test_exclusive_tools_run_sequentially(self, mock_litellm):
        execution_order = []
        mock_litellm.completion.side_effect = [
            _mock_multi_tool_response("write_a", "write_b"),
            _mock_completion_response("done"),
        ]

        def handler_a(args, ctx):
            execution_order.append("a_start")
            time.sleep(0.1)
            execution_order.append("a_end")
            return "a"

        def handler_b(args, ctx):
            execution_order.append("b_start")
            execution_order.append("b_end")
            return "b"

        tool_a = BuiltinTool(
            name="write_a",
            description="",
            input_schema={"type": "object"},
            handler=handler_a,
            is_concurrency_safe=False,
        )
        tool_b = BuiltinTool(
            name="write_b",
            description="",
            input_schema={"type": "object"},
            handler=handler_b,
            is_concurrency_safe=False,
        )

        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        agent.run("test", tools={"write_a": tool_a, "write_b": tool_b})
        assert execution_order == ["a_start", "a_end", "b_start", "b_end"]

    @patch("agentic.agent.agent.litellm")
    def test_mixed_concurrent_and_exclusive(self, mock_litellm):
        mock_litellm.completion.side_effect = [
            _mock_multi_tool_response("read_a", "read_b", "write_c"),
            _mock_completion_response("done"),
        ]

        tool_a = BuiltinTool(
            name="read_a",
            description="",
            input_schema={"type": "object"},
            handler=lambda a, c: "a",
            is_concurrency_safe=True,
        )
        tool_b = BuiltinTool(
            name="read_b",
            description="",
            input_schema={"type": "object"},
            handler=lambda a, c: "b",
            is_concurrency_safe=True,
        )
        tool_c = BuiltinTool(
            name="write_c",
            description="",
            input_schema={"type": "object"},
            handler=lambda a, c: "c",
            is_concurrency_safe=False,
        )

        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run(
            "test",
            tools={"read_a": tool_a, "read_b": tool_b, "write_c": tool_c},
        )
        assert output.status.is_success()
        assert len(output.tool_calls) == 3


class TestSafetyCaps:
    @patch("agentic.agent.agent.litellm")
    def test_result_truncated_at_max_result_chars(self, mock_litellm):
        mock_litellm.completion.side_effect = [
            _mock_multi_tool_response("big_tool"),
            _mock_completion_response("done"),
        ]

        tool = BuiltinTool(
            name="big_tool",
            description="",
            input_schema={"type": "object"},
            handler=lambda a, c: "x" * 100000,
            max_result_chars=1000,
        )

        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("test", tools={"big_tool": tool})

        tool_result_msgs = [m for m in output.messages if m.get("role") == "tool"]
        assert len(tool_result_msgs) == 1
        assert len(tool_result_msgs[0]["content"]) < 1200
        assert "Truncated" in tool_result_msgs[0]["content"]

    @patch("agentic.agent.agent.litellm")
    def test_unlimited_result_not_truncated(self, mock_litellm):
        mock_litellm.completion.side_effect = [
            _mock_multi_tool_response("unlimited"),
            _mock_completion_response("done"),
        ]

        tool = BuiltinTool(
            name="unlimited",
            description="",
            input_schema={"type": "object"},
            handler=lambda a, c: "x" * 100000,
            max_result_chars=None,
        )

        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("test", tools={"unlimited": tool})

        tool_result_msgs = [m for m in output.messages if m.get("role") == "tool"]
        assert len(tool_result_msgs[0]["content"]) == 100000
