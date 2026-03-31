import threading

from agentic.execution.context import ExecutionContext


class TestAbortSignal:
    def test_not_aborted_by_default(self):
        ctx = ExecutionContext()
        assert ctx.is_aborted is False

    def test_not_aborted_without_signal(self):
        ctx = ExecutionContext(abort_signal=None)
        assert ctx.is_aborted is False

    def test_aborted_when_signal_set(self):
        signal = threading.Event()
        ctx = ExecutionContext(abort_signal=signal)
        assert ctx.is_aborted is False
        signal.set()
        assert ctx.is_aborted is True

    def test_child_inherits_abort_signal(self):
        signal = threading.Event()
        ctx = ExecutionContext(execution_id="parent", abort_signal=signal)
        child = ctx.child_context()
        assert child.abort_signal is signal
        signal.set()
        assert child.is_aborted is True

    def test_with_metadata_inherits_abort_signal(self):
        signal = threading.Event()
        ctx = ExecutionContext(execution_id="parent", abort_signal=signal)
        new_ctx = ctx.with_metadata(foo="bar")
        assert new_ctx.abort_signal is signal
