"""Unit test: DelegateTool invokes on_run_complete after agent.run()."""

from unittest.mock import MagicMock

from agentic.agent.tools import DelegateTool
from agentic.execution.context import ExecutionContext
from agentic.execution.status import ExecutionStatus


def _make_agent_output(content="done", status=ExecutionStatus.COMPLETED):
    output = MagicMock()
    output.content = content
    output.status = status
    output.error = None
    output.usage = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    output.steps = 2
    output.events = [{"type": "step", "seq": 1}]
    output.tool_calls = []
    output.messages = []
    output.started_at = None
    output.completed_at = None
    return output


def test_delegate_tool_invokes_hook_with_payload():
    fake_agent = MagicMock()
    fake_agent.name = "Specialist"
    fake_agent.run.return_value = _make_agent_output()

    captured = {}

    def hook(payload):
        captured.update(payload)

    tool = DelegateTool(
        name="delegate_to_Specialist",
        description="",
        agent=fake_agent,
        on_run_complete=hook,
    )
    ctx = ExecutionContext(orchestration_run_id="orch-run-1")

    result = tool.execute({"task": "do it"}, ctx)

    assert result == "done"
    assert captured["task"] == "do it"
    assert captured["content"] == "done"
    assert captured["status"] == ExecutionStatus.COMPLETED
    assert captured["usage"]["total_tokens"] == 3
    assert captured["agent_name"] == "Specialist"
    assert captured["orchestration_run_id"] == "orch-run-1"
    assert captured["child_execution_id"]  # non-empty
    assert "events" in captured
    assert "tool_calls" in captured
    assert isinstance(captured["tool_calls"], list)
    assert "messages" in captured
    assert "reasoning_steps" not in captured


def test_delegate_tool_hook_exceptions_do_not_break_delegation():
    fake_agent = MagicMock()
    fake_agent.name = "Specialist"
    fake_agent.run.return_value = _make_agent_output()

    def bad_hook(payload):
        raise RuntimeError("DB is down")

    tool = DelegateTool(
        name="delegate_to_Specialist",
        description="",
        agent=fake_agent,
        on_run_complete=bad_hook,
    )
    ctx = ExecutionContext(orchestration_run_id="orch-run-1")

    # Must not raise — persistence failure is logged and swallowed.
    result = tool.execute({"task": "do it"}, ctx)
    assert result == "done"


def test_delegate_tool_without_hook_works_as_before():
    fake_agent = MagicMock()
    fake_agent.name = "Specialist"
    fake_agent.run.return_value = _make_agent_output()

    tool = DelegateTool(name="delegate_to_Specialist", description="", agent=fake_agent)
    ctx = ExecutionContext()
    assert tool.execute({"task": "hi"}, ctx) == "done"
