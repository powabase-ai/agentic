"""Agent.stream() must propagate the caller's contextvars into its worker thread.

Agent.stream() accumulates the LLM response on a background thread. That thread
is where `accumulate_stream` — and therefore litellm's callbacks — run, so any
ContextVar the caller set is only visible to those callbacks if the thread
starts from a *copy of the caller's context*.

This is the same requirement the concurrent-tool path already meets: see the
`pool.submit(contextvars.copy_context().run, ...)` call in agent.py and the
comment above it, which states that without it "workers see ContextVar defaults
and downstream billing falls back to non-deterministic ids".

With a bare `threading.Thread(target=_worker)` the worker started from a fresh
context, so a consumer that tags each LLM call with the current run id via a
ContextVar read the default instead and lost that correlation on every
tool-less streaming run.
"""

from __future__ import annotations

import contextvars
import threading

import agentic.llm.streaming as streaming_mod
from agentic.agent.agent import Agent

# Default is deliberately distinguishable from anything the test sets, so a
# failure reads as "worker got the default" rather than "worker got nothing".
_run_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "test_run_id", default=None
)


def _consume(gen):
    while True:
        try:
            next(gen)
        except StopIteration:
            return


def test_stream_worker_thread_sees_callers_contextvar(
    mock_litellm_streaming, monkeypatch
):
    """The value set before stream() must be readable inside the worker thread.

    Counterfactual: with the bare `threading.Thread(target=_worker)` this test
    records None (the ContextVar default) and fails on the identity assertion.
    With `copy_context().run` it records the caller's value.
    """
    seen: dict[str, object] = {}
    real_accumulate = streaming_mod.accumulate_stream

    def spy(*args, **kwargs):
        # Runs on the worker thread — this is the observation point.
        seen["run_id"] = _run_id_var.get()
        seen["thread"] = threading.current_thread().name
        return real_accumulate(*args, **kwargs)

    monkeypatch.setattr(streaming_mod, "accumulate_stream", spy)

    token = _run_id_var.set("run-abc123")
    try:
        agent = Agent(system_prompt="You are helpful")
        _consume(agent.stream("Hello"))
    finally:
        _run_id_var.reset(token)

    assert "run_id" in seen, "spy never ran — stream() did not reach accumulate_stream"
    assert seen["thread"] != threading.current_thread().name, (
        "accumulate_stream ran on the calling thread; this test no longer "
        "exercises the worker-thread context boundary it was written for"
    )
    assert seen["run_id"] == "run-abc123", (
        f"worker thread saw {seen['run_id']!r}, not the caller's value. The "
        f"thread was started without contextvars.copy_context().run, so "
        f"callbacks on this path read ContextVar defaults."
    )
