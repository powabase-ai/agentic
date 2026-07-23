"""Reactive prompt_too_long path falls back to deterministic truncation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agentic.agent.agent import Agent
from agentic.execution.context import ExecutionContext
from agentic.execution.status import ExecutionStatus


def _make_response(content="ok"):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    msg.reasoning_content = None
    msg.thinking_blocks = None
    msg.provider_specific_fields = None
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    response = MagicMock()
    response.choices = [choice]
    response.usage = SimpleNamespace(
        prompt_tokens=10, completion_tokens=10, total_tokens=20
    )
    response.id = "r"
    return response


# A history that genuinely exceeds the truncation target. The reactive path
# truncates to `get_context_threshold(...) * 0.5`, so with the threshold pinned
# at _THRESHOLD below the target is 100 tokens / 400 chars, and this history is
# ~3,000 tokens. Running the fallback on a zero-token history would make its
# "did it shrink" assertion `0 < 0` and prove nothing.
_THRESHOLD = 200
_TARGET = _THRESHOLD // 2
_BIG_HISTORY = [
    {"role": "user", "content": f"question {i} " + "x" * 2000}
    if i % 2 == 0
    else {"role": "assistant", "content": f"answer {i} " + "x" * 2000}
    for i in range(6)
]


def _run_with_compaction(compact_side_effect):
    """Run one prompt_too_long recovery with the given compact_messages stub."""
    events: list[dict] = []
    ctx = ExecutionContext(on_event=events.append)
    call_results = [Exception("prompt too long"), _make_response()]

    def fake_completion(**kwargs):
        r = call_results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    with (
        patch("agentic.agent.agent.litellm.completion", side_effect=fake_completion),
        patch("agentic.agent.agent.classify_error", return_value="prompt_too_long"),
        patch("agentic.agent.agent.get_context_threshold", return_value=_THRESHOLD),
        patch("agentic.agent.agent.compact_messages", side_effect=compact_side_effect),
        patch.dict("os.environ", {"AGENT_LLM_STREAMING_ENABLED": "false"}),
    ):
        agent = Agent(model="anthropic/claude-opus-4-7")
        agent.run(list(_BIG_HISTORY), context=ctx)
    return events


def test_truncation_fallback_when_compaction_no_ops():
    # compact_messages returns its input list object == no progress.
    events = _run_with_compaction(lambda messages, **kw: messages)

    truncs = [e for e in events if e.get("type") == "reactive_truncate"]
    assert len(truncs) == 1
    assert truncs[0]["step"] == 1
    # The fallback must have done real work: STRICTLY smaller, and actually
    # under the target it was given. `after <= before` would pass on a history
    # that was never truncated at all.
    assert truncs[0]["before_tokens"] > _TARGET
    assert truncs[0]["after_tokens"] < truncs[0]["before_tokens"]
    assert truncs[0]["after_tokens"] <= _TARGET


def test_truncate_event_reports_succeeded():
    events = _run_with_compaction(lambda messages, **kw: messages)
    truncs = [e for e in events if e.get("type") == "reactive_truncate"]
    assert truncs[0]["succeeded"] is True


def test_truncate_event_reports_not_succeeded_when_it_bailed_out():
    """truncate_messages returns its input object when a postcondition guard
    trips — the event must not claim a truncation that did not happen."""
    with patch(
        "agentic.agent.agent.truncate_messages", side_effect=lambda msgs, _t: msgs
    ):
        events = _run_with_compaction(lambda messages, **kw: messages)
    truncs = [e for e in events if e.get("type") == "reactive_truncate"]
    assert len(truncs) == 1
    assert truncs[0]["succeeded"] is False


def test_truncate_event_reports_not_succeeded_when_nothing_shrank():
    """truncate_messages rebuilds dicts through normalize_messages, so a
    truncation that sheds nothing still returns a NEW list. An identity check
    (`truncated is not compacted`) reads that as success; only a token delta
    tells the truth."""
    with patch(
        "agentic.agent.agent.truncate_messages",
        # New list object, byte-identical content — exactly what a no-op
        # rebuild produces.
        side_effect=lambda msgs, _t: [dict(m) for m in msgs],
    ):
        events = _run_with_compaction(lambda messages, **kw: messages)
    truncs = [e for e in events if e.get("type") == "reactive_truncate"]
    assert len(truncs) == 1
    assert truncs[0]["after_tokens"] == truncs[0]["before_tokens"]
    assert truncs[0]["succeeded"] is False


def test_successful_recovery_re_arms_the_compact_latch():
    """`has_attempted_compact` gates the whole reactive branch and nothing ever
    reset it, so a run that recovered once had no recovery left for the rest of
    its life. A recovery that actually shrank the history must clear it."""
    events: list[dict] = []
    ctx = ExecutionContext(on_event=events.append)
    # Four consecutive recoveries: three would only just reach the breaker's
    # `< 3` bound, so a run that drove fewer could not distinguish a re-armed
    # latch from a counter that never reset.
    call_results = [
        Exception("prompt too long"),
        Exception("prompt too long"),
        Exception("prompt too long"),
        Exception("prompt too long"),
        _make_response(),
    ]

    def fake_completion(**kwargs):
        r = call_results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    ok_compact = _halving_compact

    with (
        patch("agentic.agent.agent.litellm.completion", side_effect=fake_completion),
        patch("agentic.agent.agent.classify_error", return_value="prompt_too_long"),
        patch("agentic.agent.agent.get_context_threshold", return_value=_THRESHOLD),
        patch("agentic.agent.agent.compact_messages", side_effect=ok_compact),
        patch.dict("os.environ", {"AGENT_LLM_STREAMING_ENABLED": "false"}),
    ):
        agent = Agent(model="anthropic/claude-opus-4-7")
        out = agent.run(list(_BIG_HISTORY), context=ctx)

    # With a permanent one-shot latch the second prompt_too_long has no
    # recovery path and only one reactive_compact ever fires. With a latch that
    # re-arms but a counter that never resets, the fourth is refused by
    # `compact_failure_count < 3` — so asserting on all four pins both guards.
    assert len([e for e in events if e.get("type") == "reactive_compact"]) == 4
    assert out.status == ExecutionStatus.COMPLETED


def _halving_compact(messages, **kw):
    """A compaction stub that shrinks *relative to its input*.

    Returning a fixed result regardless of input only ever satisfied an
    identity check. By the time the reactive site runs, the proactive site has
    already compacted, so a fixed 1-message return is not smaller than what it
    is handed and is correctly read as no progress.
    """
    chars = sum(len(str(m.get("content", ""))) for m in messages)
    return [{"role": "user", "content": "c" * max(1, chars // 2)}]


def test_no_truncation_when_compaction_succeeded():
    events = _run_with_compaction(_halving_compact)
    assert [e for e in events if e.get("type") == "reactive_truncate"] == []
    # the normal reactive_compact path still fired
    assert [e for e in events if e.get("type") == "reactive_compact"]


class _Runaway(BaseException):
    """Escapes the loop's own `except Exception` handler.

    A plain Exception sentinel is swallowed by the reactive branch, classified
    as another prompt_too_long, and recovered from — so the test it was meant
    to fail would hang instead.
    """


def test_reactive_branch_terminates_when_compaction_never_shrinks():
    """The two guards on the reactive branch — the `has_attempted_compact`
    latch and `compact_failure_count < 3` — are both cleared by the very event
    they exist to bound if "progress" is read as list identity. `continue` also
    skips the turn_count increment, so `max_steps` is never consulted here.
    A compaction that returns a new-but-not-smaller list then makes the state a
    fixed point: an unbounded run issuing one full-context call per iteration.

    The asymmetry that triggers it is ordinary, not contrived: the real call
    sends `self.max_tokens` while compaction hardcodes its own, so a history
    can be un-sendable by the agent yet perfectly compactable.
    """
    calls = {"n": 0}

    def always_too_long(**kwargs):
        calls["n"] += 1
        if calls["n"] > 40:
            raise _Runaway(f"unbounded: {calls['n']} provider calls")
        raise Exception("prompt too long")

    events: list[dict] = []
    ctx = ExecutionContext(on_event=events.append)
    with (
        patch("agentic.agent.agent.litellm.completion", side_effect=always_too_long),
        patch("agentic.agent.agent.classify_error", return_value="prompt_too_long"),
        patch("agentic.agent.agent.get_context_threshold", return_value=_THRESHOLD),
        # New list object every time, byte-identical content: compaction that
        # reports success while doing no work.
        patch(
            "agentic.agent.agent.compact_messages",
            side_effect=lambda messages, **kw: [dict(m) for m in messages],
        ),
        patch(
            "agentic.agent.agent.truncate_messages",
            side_effect=lambda msgs, _t: [dict(m) for m in msgs],
        ),
        patch.dict("os.environ", {"AGENT_LLM_STREAMING_ENABLED": "false"}),
    ):
        agent = Agent(model="anthropic/claude-opus-4-7")
        out = agent.run(list(_BIG_HISTORY), context=ctx, max_steps=5)

    assert out.status == ExecutionStatus.FAILED
    # Fails fast rather than looping: one recovery attempt, then the branch
    # stays closed because nothing shrank.
    assert calls["n"] <= 4, f"expected fast failure, got {calls['n']} provider calls"
    assert len([e for e in events if e.get("type") == "reactive_compact"]) == 1
