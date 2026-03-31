import pytest

from agentic.execution.context import ExecutionContext, MaxDepthExceeded, TokenBudget


class TestTokenBudget:
    def test_remaining(self):
        budget = TokenBudget(max_tokens=1000)
        assert budget.remaining == 1000

    def test_consume(self):
        budget = TokenBudget(max_tokens=1000)
        budget.consume(300)
        assert budget.remaining == 700
        assert budget.used_tokens == 300

    def test_remaining_never_negative(self):
        budget = TokenBudget(max_tokens=100)
        budget.consume(200)
        assert budget.remaining == 0

    def test_allocate_child(self):
        budget = TokenBudget(max_tokens=1000)
        budget.consume(400)
        child = budget.allocate_child()
        assert child.max_tokens == 600
        assert child.used_tokens == 0

    def test_exceeded(self):
        budget = TokenBudget(max_tokens=100)
        assert not budget.exceeded
        budget.consume(100)
        assert budget.exceeded


class TestExecutionContext:
    def test_child_context(self):
        events = []
        ctx = ExecutionContext(
            execution_id="parent-1",
            session_id="sess-1",
            on_event=lambda e: events.append(e),
            depth=0,
            max_depth=3,
        )
        child = ctx.child_context()
        assert child.parent_run_id == "parent-1"
        assert child.session_id == "sess-1"
        assert child.depth == 1
        assert child.on_event is ctx.on_event

    def test_child_context_depth_limit(self):
        ctx = ExecutionContext(
            execution_id="deep",
            depth=3,
            max_depth=3,
        )
        with pytest.raises(MaxDepthExceeded):
            ctx.child_context()

    def test_child_context_inherits_budget(self):
        budget = TokenBudget(max_tokens=1000)
        budget.consume(400)
        ctx = ExecutionContext(
            execution_id="parent",
            budget=budget,
        )
        child = ctx.child_context()
        assert child.budget.max_tokens == 600

    def test_child_context_no_budget(self):
        ctx = ExecutionContext(execution_id="parent")
        child = ctx.child_context()
        assert child.budget is None

    def test_emit_event(self):
        events = []
        ctx = ExecutionContext(
            execution_id="test",
            on_event=lambda e: events.append(e),
        )
        ctx.emit_event({"type": "test_event", "data": "hello"})
        assert len(events) == 1
        assert events[0]["type"] == "test_event"
        assert "seq" in events[0]
        assert "ts" in events[0]

    def test_emit_event_no_callback(self):
        ctx = ExecutionContext(execution_id="test")
        # Should not raise
        ctx.emit_event({"type": "ignored"})
