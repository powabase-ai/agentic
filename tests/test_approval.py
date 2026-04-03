"""Tests for approval gate primitives on ExecutionContext."""

import threading

from agentic.execution.context import ExecutionContext


class TestApprovalGate:
    def test_get_approval_event_creates_event(self):
        """get_approval_event returns a threading.Event that is not yet set."""
        ctx = ExecutionContext()
        event = ctx.get_approval_event()
        assert isinstance(event, threading.Event)
        assert not event.is_set()

    def test_get_approval_event_returns_same_instance(self):
        """get_approval_event is idempotent — repeated calls return the same object."""
        ctx = ExecutionContext()
        event1 = ctx.get_approval_event()
        event2 = ctx.get_approval_event()
        assert event1 is event2

    def test_set_approval_decision_sets_event(self):
        """set_approval_decision stores the decision and fires the event."""
        ctx = ExecutionContext()
        event = ctx.get_approval_event()
        assert not event.is_set()

        ctx.set_approval_decision({"approved": True, "reason": "looks good"})

        assert event.is_set()
        assert ctx.get_approval_decision() == {"approved": True, "reason": "looks good"}

    def test_approval_decision_default_none(self):
        """get_approval_decision returns None before any decision is set."""
        ctx = ExecutionContext()
        assert ctx.get_approval_decision() is None

    def test_approval_flow_threaded(self):
        """A waiting thread unblocks once another thread calls set_approval_decision."""
        ctx = ExecutionContext()
        results = {}

        def waiter():
            event = ctx.get_approval_event()
            # Block until decision arrives (give up after 2 s to avoid hanging tests)
            signalled = event.wait(timeout=2.0)
            results["signalled"] = signalled
            results["decision"] = ctx.get_approval_decision()

        t = threading.Thread(target=waiter)
        t.start()

        # Give the waiter thread time to reach event.wait()
        import time

        time.sleep(0.05)

        ctx.set_approval_decision({"approved": False, "reason": "needs changes"})
        t.join(timeout=2.0)

        assert not t.is_alive(), "waiter thread did not finish in time"
        assert results["signalled"] is True
        assert results["decision"] == {"approved": False, "reason": "needs changes"}
