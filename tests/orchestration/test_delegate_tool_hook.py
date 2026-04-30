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


def test_delegate_tool_propagates_agent_id_to_hook_payload():
    """The platform layer threads its agent UUID through OrchestrationEntity
    → DelegateTool.agent_id; the hook payload must echo it so the platform
    can persist child agent_runs with the right agent_id (otherwise the
    per-agent dashboard breakdown under-counts).
    """
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
        agent_id="agent-uuid-123",
    )
    ctx = ExecutionContext()
    tool.execute({"task": "do it"}, ctx)
    assert captured["agent_id"] == "agent-uuid-123"


def test_delegate_tool_agent_id_defaults_to_none():
    """Callers that don't register an agent_id (non-platform delegations)
    still get a payload — agent_id is None, not missing or KeyError."""
    fake_agent = MagicMock()
    fake_agent.name = "Specialist"
    fake_agent.run.return_value = _make_agent_output()

    captured = {}
    tool = DelegateTool(
        name="delegate_to_Specialist",
        description="",
        agent=fake_agent,
        on_run_complete=lambda p: captured.update(p),
    )
    tool.execute({"task": "do it"}, ExecutionContext())
    assert captured["agent_id"] is None


# ---------------------------------------------------------------------------
# 3b: Sequential and Parallel engines invoke the hook
# ---------------------------------------------------------------------------


def _make_mock_entity(agent_name: str, content: str = "done"):
    """Build a minimal OrchestrationEntity-like mock for sequential/parallel tests."""
    from agentic.orchestration.orchestration import OrchestrationEntity

    fake_agent = MagicMock()
    fake_agent.name = agent_name
    fake_agent.run.return_value = _make_agent_output(content=content)

    entity = OrchestrationEntity(
        entity_type="agent",
        agent=fake_agent,
        position=None,
        config={},
        agent_tools=None,
    )
    return entity


def _make_orchestration(strategy: str, entities):
    from agentic.orchestration.orchestration import Orchestration

    orch = Orchestration(name="test", description="test", strategy=strategy)
    orch.entities = entities
    return orch


def test_sequential_engine_invokes_hook_per_agent():
    from agentic.orchestration.strategies import SequentialEngine

    entity_a = _make_mock_entity("AgentA", content="result-a")
    entity_a.position = 0
    entity_b = _make_mock_entity("AgentB", content="result-b")
    entity_b.position = 1

    orch = _make_orchestration("sequential", [entity_a, entity_b])

    captured = []

    def hook(payload):
        captured.append(payload)

    ctx = ExecutionContext(orchestration_run_id="orch-seq-1")
    engine = SequentialEngine()
    engine.execute(orch, "go", None, ctx, on_delegate_complete=hook)

    assert len(captured) == 2
    agent_names = [p["agent_name"] for p in captured]
    assert agent_names == ["AgentA", "AgentB"]
    assert captured[0]["content"] == "result-a"
    assert captured[1]["content"] == "result-b"
    for p in captured:
        assert p["orchestration_run_id"] == "orch-seq-1"
        assert p["child_execution_id"]
        assert "status" in p
        assert "tool_calls" in p
        assert isinstance(p["tool_calls"], list)
        assert "messages" in p
        assert "reasoning_steps" not in p


def test_parallel_engine_invokes_hook_per_agent():
    from agentic.orchestration.strategies import ParallelEngine

    entity_a = _make_mock_entity("ParAgentA", content="par-result-a")
    entity_b = _make_mock_entity("ParAgentB", content="par-result-b")

    orch = _make_orchestration("parallel", [entity_a, entity_b])

    captured = []

    def hook(payload):
        captured.append(payload)

    ctx = ExecutionContext(orchestration_run_id="orch-par-1")
    engine = ParallelEngine()
    # ParallelEngine with 2 agents invokes a merge step; patch Agent to avoid real LLM calls.
    import unittest.mock as mock

    with mock.patch("agentic.orchestration.strategies.Agent") as MockAgent:
        merge_output = _make_agent_output(content="merged")
        mock_merger = MagicMock()
        mock_merger.run.return_value = merge_output
        MockAgent.return_value = mock_merger

        engine.execute(orch, "go", None, ctx, on_delegate_complete=hook)

    assert len(captured) == 2
    agent_names = {p["agent_name"] for p in captured}
    assert agent_names == {"ParAgentA", "ParAgentB"}
    for p in captured:
        assert p["orchestration_run_id"] == "orch-par-1"
        assert p["child_execution_id"]
        assert "status" in p
        assert "tool_calls" in p
        assert isinstance(p["tool_calls"], list)
        assert "messages" in p
        assert "reasoning_steps" not in p


def test_sequential_engine_propagates_agent_id():
    """Sequential's hook payload must echo entity.agent_id so the platform
    layer can persist child agent_runs with the right agent_id."""
    from agentic.orchestration.strategies import SequentialEngine

    entity_a = _make_mock_entity("AgentA")
    entity_a.position = 0
    entity_a.agent_id = "uuid-a"
    entity_b = _make_mock_entity("AgentB")
    entity_b.position = 1
    entity_b.agent_id = "uuid-b"

    orch = _make_orchestration("sequential", [entity_a, entity_b])
    captured = []
    SequentialEngine().execute(
        orch, "go", None, ExecutionContext(), on_delegate_complete=captured.append
    )
    assert {p["agent_id"] for p in captured} == {"uuid-a", "uuid-b"}


def test_parallel_engine_propagates_agent_id():
    """Same propagation contract for the parallel engine."""
    from agentic.orchestration.strategies import ParallelEngine

    entity_a = _make_mock_entity("ParAgentA")
    entity_a.agent_id = "uuid-par-a"
    entity_b = _make_mock_entity("ParAgentB")
    entity_b.agent_id = "uuid-par-b"

    orch = _make_orchestration("parallel", [entity_a, entity_b])
    captured = []

    import unittest.mock as mock

    with mock.patch("agentic.orchestration.strategies.Agent") as MockAgent:
        merge_output = _make_agent_output(content="merged")
        mock_merger = MagicMock()
        mock_merger.run.return_value = merge_output
        MockAgent.return_value = mock_merger
        ParallelEngine().execute(
            orch, "go", None, ExecutionContext(), on_delegate_complete=captured.append
        )

    assert {p["agent_id"] for p in captured} == {"uuid-par-a", "uuid-par-b"}
