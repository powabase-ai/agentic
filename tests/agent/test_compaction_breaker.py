"""Circuit breaker counts CONSECUTIVE compaction failures, and event fidelity."""

from __future__ import annotations

import itertools
from types import SimpleNamespace
from unittest.mock import patch

from agentic import Agent
from agentic.agent.loop_state import LoopState
from agentic.agent.tools import BuiltinTool
from agentic.execution.context import ExecutionContext
from agentic.execution.status import ExecutionStatus


def _tool_call_response(call_id, i=0):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    role="assistant",
                    tool_calls=[
                        SimpleNamespace(
                            id=call_id,
                            type="function",
                            function=SimpleNamespace(
                                name="probe", arguments=f'{{"i": {i}}}'
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _text_response(content="done"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content, role="assistant", tool_calls=None
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _probe_tool():
    return BuiltinTool(
        name="probe",
        description="probe",
        input_schema={"type": "object", "properties": {"i": {"type": "integer"}}},
        handler=lambda args, ctx: "ok " + "z" * 200,
    )


_HISTORY = [
    {"role": "user", "content": "start " + "x" * 600},
    {"role": "assistant", "content": "a1 " + "x" * 600},
    {"role": "user", "content": "q2 " + "x" * 600},
    {"role": "assistant", "content": "a2 " + "x" * 600},
    {"role": "user", "content": "now answer"},
]


class TestLoopStateCompactSuccess:
    def test_with_compact_success_clears_counter(self):
        state = LoopState.initial(messages=[], model="m")
        state = state.with_compact_failure().with_compact_failure()
        assert state.compact_failure_count == 2
        assert state.with_compact_success().compact_failure_count == 0

    def test_with_compact_success_preserves_everything_else(self):
        state = LoopState.initial(messages=[{"role": "user"}], model="m")
        state = state.with_compact_failure()
        after = state.with_compact_success()
        assert after.messages == state.messages
        assert after.current_model == state.current_model
        assert after.turn_count == state.turn_count
        assert after.has_attempted_compact == state.has_attempted_compact


def _run_alternating_compaction(max_steps=8):
    """Run an agent where compaction alternates fail, success, fail, success...

    A *cumulative* failure counter reaches 3 partway through and permanently
    disables compaction. A *consecutive* counter never exceeds 1, so calls
    continue for the whole run.
    """
    calls: list[int] = []

    def fake_compact(messages, **kw):
        calls.append(1)
        if len(calls) % 2 == 1:  # odd call -> no progress (transient failure)
            return messages
        return [
            {"role": "user", "content": "summary " + "y" * 500},
            {"role": "user", "content": "continue"},
        ]

    responses = [_tool_call_response(f"c{i}", i) for i in range(max_steps - 1)]
    responses.append(_text_response())

    with (
        patch("agentic.agent.agent.litellm") as mock_litellm,
        patch("agentic.agent.agent.get_context_threshold", return_value=0),
        patch("agentic.agent.agent.compact_messages", side_effect=fake_compact),
    ):
        mock_litellm.completion.side_effect = responses
        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        agent.run(_HISTORY, tools={"probe": _probe_tool()}, max_steps=max_steps)
    return len(calls)


def _run_proactive_site_only(max_steps=12):
    """Exercise ONLY the proactive (site 1) compaction path.

    ``get_context_threshold`` is called once per step by site 1 and then once
    by the main-loop compaction check (site 3), in that order. Returning 0 for
    the site-1 call and a huge value for the site-3 call keeps site 1 firing
    every step while site 3 never triggers — so the only breaker reset in play
    is site 1's own ``state.recover(..., compact_failure_count=0)``.

    Without that reset, ``LoopState.recover`` falls back to the *cumulative*
    counter it was handed None for, the breaker reaches 3, and compaction is
    disabled for the rest of the run.
    """
    calls: list[int] = []
    threshold_calls = itertools.count()

    def fake_threshold(model, max_output_tokens=None):
        return 0 if next(threshold_calls) % 2 == 0 else 10**9

    def fake_compact(messages, **kw):
        calls.append(1)
        if len(calls) % 3 != 0:  # fail, fail, succeed, fail, fail, succeed...
            return messages
        return [
            {"role": "user", "content": "summary " + "y" * 500},
            {"role": "user", "content": "continue"},
        ]

    responses = [_tool_call_response(f"c{i}", i) for i in range(max_steps - 1)]
    responses.append(_text_response())

    with (
        patch("agentic.agent.agent.litellm") as mock_litellm,
        patch("agentic.agent.agent.get_context_threshold", side_effect=fake_threshold),
        patch("agentic.agent.agent.compact_messages", side_effect=fake_compact),
    ):
        mock_litellm.completion.side_effect = responses
        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        agent.run(_HISTORY, tools={"probe": _probe_tool()}, max_steps=max_steps)
    return len(calls)


class TestConsecutiveFailureCounting:
    def test_interleaved_failures_do_not_disable_compaction(self):
        # With a cumulative counter this caps out at 4-5 calls (the breaker
        # trips on the 3rd lifetime failure and never re-arms).
        assert _run_alternating_compaction(max_steps=16) > 10

    def test_proactive_success_resets_the_breaker(self):
        # Site-1-only. The failure pattern is fail, fail, SUCCESS, fail, ...:
        # with the reset the consecutive count never reaches 3 and compaction
        # runs every step; without it the count carries past the success and
        # the breaker trips after 4 calls.
        assert _run_proactive_site_only(max_steps=12) > 6


class TestProactiveCompactEvent:
    """The event must not assert a compaction that did not happen."""

    def _run(self, compact_side_effect):
        events: list[dict] = []
        ctx = ExecutionContext(on_event=events.append)
        with (
            patch("agentic.agent.agent.litellm") as mock_litellm,
            patch("agentic.agent.agent.get_context_threshold", return_value=0),
            patch(
                "agentic.agent.agent.compact_messages", side_effect=compact_side_effect
            ),
        ):
            mock_litellm.completion.side_effect = [_text_response()]
            agent = Agent(model="gpt-4o-mini", system_prompt="test")
            agent.run(
                _HISTORY, tools={"probe": _probe_tool()}, max_steps=1, context=ctx
            )
        return [e for e in events if e.get("type") == "proactive_compact"]

    def test_succeeded_false_when_compaction_makes_no_progress(self):
        evts = self._run(lambda messages, **kw: messages)
        assert evts
        assert evts[0]["succeeded"] is False

    def test_succeeded_true_when_compaction_works(self):
        evts = self._run(
            lambda messages, **kw: [
                {"role": "user", "content": "summary"},
                {"role": "user", "content": "continue"},
            ]
        )
        assert evts
        assert evts[0]["succeeded"] is True

    def test_reports_tokens_after(self):
        evts = self._run(
            lambda messages, **kw: [
                {"role": "user", "content": "summary"},
                {"role": "user", "content": "continue"},
            ]
        )
        evt = evts[0]
        assert "tokens_after" in evt
        assert evt["tokens_after"] < evt["tokens_before"]

    def test_tokens_after_present_even_when_compaction_failed(self):
        evt = self._run(lambda messages, **kw: messages)[0]
        assert "tokens_after" in evt
        assert "tokens_before" in evt

    def test_bigger_history_is_rejected_not_merely_reported(self):
        # A new-but-larger list is not progress. Reporting success off list
        # identity claimed it was — and asserting only on the event flag pins
        # the symptom while leaving the consequences unpinned: the larger
        # history still got installed, and the breaker still got reset, so a
        # compaction that consistently grows the conversation kept buying an
        # extra full-context call every step with nothing to stop it.
        evts = self._run(
            lambda messages, **kw: [
                {"role": "user", "content": "bloat " + "z" * 20000},
                {"role": "user", "content": "continue"},
            ]
        )
        assert evts
        assert evts[0]["succeeded"] is False
        # The bloated history is discarded, not installed.
        assert evts[0]["tokens_after"] <= evts[0]["tokens_before"]


class TestProactiveCompactionExceptionContained:
    """A raising compact_messages must not take the run down.

    The handler's `compacted = normalized` assignment is load-bearing: it is
    what makes the identity check below trip the circuit breaker instead of
    treating the failure as a successful compaction.
    """

    def _run_with_raising_compact(self):
        events: list[dict] = []
        ctx = ExecutionContext(on_event=events.append)

        def boom(messages, **kw):
            raise RuntimeError("provider 500 during compaction")

        with (
            patch("agentic.agent.agent.litellm") as mock_litellm,
            patch("agentic.agent.agent.get_context_threshold", return_value=0),
            patch("agentic.agent.agent.compact_messages", side_effect=boom),
        ):
            mock_litellm.completion.side_effect = [_text_response()]
            agent = Agent(model="gpt-4o-mini", system_prompt="test")
            out = agent.run(
                _HISTORY, tools={"probe": _probe_tool()}, max_steps=1, context=ctx
            )
        return out, [e for e in events if e.get("type") == "proactive_compact"]

    def test_run_completes_despite_compaction_error(self):
        out, _ = self._run_with_raising_compact()
        assert out.status == ExecutionStatus.COMPLETED
        assert out.content == "done"

    def test_event_reports_failure_and_breaker_trips(self):
        _, evts = self._run_with_raising_compact()
        assert evts, "proactive_compact event must still be emitted"
        assert evts[0]["succeeded"] is False


class TestCompactionModelMatchesRealCall:
    """The compaction call must go out under the same routed model string as
    the real call, or it lands in a different provider cache pool and the
    un-pruned history it is fed buys nothing."""

    def _models(self, model, reasoning_effort):
        seen: dict = {}

        def fake_compact(messages, **kw):
            seen.update(kw)
            return [{"role": "user", "content": "s"}, {"role": "user", "content": "c"}]

        with (
            patch("agentic.agent.agent.litellm") as mock_litellm,
            patch("agentic.agent.agent.get_context_threshold", return_value=0),
            patch("agentic.agent.agent.compact_messages", side_effect=fake_compact),
        ):
            mock_litellm.completion.side_effect = [_text_response()]
            agent = Agent(
                model=model,
                system_prompt="test",
                reasoning_effort=reasoning_effort,
            )
            agent.run(_HISTORY, tools={"probe": _probe_tool()}, max_steps=2)
            real_model = mock_litellm.completion.call_args.kwargs["model"]
        return seen["model"], real_model

    def test_openai_reasoning_model_routed_through_responses(self):
        compaction_model, real_model = self._models("gpt-5.4", "medium")
        assert real_model == "openai/responses/gpt-5.4"
        assert compaction_model == real_model

    def test_non_reasoning_model_left_unrouted(self):
        compaction_model, real_model = self._models("gpt-4o-mini", None)
        assert real_model == "gpt-4o-mini"
        assert compaction_model == real_model


class TestToolsForwardedToCompaction:
    """Compaction must send the tools the real call sends — no more, no less.

    Tool definitions lead the cached prefix, so any mismatch with the real
    call breaks the prefix match at the first block and makes the
    (full-context) compaction call entirely cache-cold.
    """

    def _seen_kwargs(self, max_steps):
        seen: dict = {}

        def fake_compact(messages, **kw):
            seen.update(kw)
            return [{"role": "user", "content": "s"}, {"role": "user", "content": "c"}]

        with (
            patch("agentic.agent.agent.litellm") as mock_litellm,
            patch("agentic.agent.agent.get_context_threshold", return_value=0),
            patch("agentic.agent.agent.compact_messages", side_effect=fake_compact),
        ):
            mock_litellm.completion.side_effect = [_text_response()]
            agent = Agent(model="gpt-4o-mini", system_prompt="test")
            agent.run(_HISTORY, tools={"probe": _probe_tool()}, max_steps=max_steps)
        return seen

    def test_proactive_site_passes_tool_schemas(self):
        seen = self._seen_kwargs(max_steps=2)  # step 1 is not the last step
        assert "tools" in seen
        names = [t["function"]["name"] for t in seen["tools"]]
        assert "probe" in names

    def test_proactive_site_omits_tools_on_last_step(self):
        # The real call drops tools on the last step to force a text answer
        # (`step_tools = None if is_last_step else tool_schemas`). Compaction
        # sending them anyway is a prefix mismatch on exactly the call whose
        # cache we are trying to hit.
        seen = self._seen_kwargs(max_steps=1)  # step 1 IS the last step
        assert seen["tools"] is None


class TestMaxTokensThreadedIntoThreshold:
    """The threshold must reserve the output budget we actually request."""

    def _threshold_calls(self, max_tokens):
        calls = []

        def fake_threshold(model, max_output_tokens=None):
            calls.append((model, max_output_tokens))
            return 10_000_000  # never over threshold; we only inspect the args

        with (
            patch("agentic.agent.agent.litellm") as mock_litellm,
            patch(
                "agentic.agent.agent.get_context_threshold", side_effect=fake_threshold
            ),
        ):
            mock_litellm.completion.side_effect = [_text_response()]
            agent = Agent(
                model="gpt-4o-mini", system_prompt="test", max_tokens=max_tokens
            )
            agent.run(_HISTORY, tools={"probe": _probe_tool()}, max_steps=1)
        return calls

    def test_configured_max_tokens_reaches_threshold(self):
        calls = self._threshold_calls(32_000)
        assert calls
        assert all(mt == 32_000 for _, mt in calls), calls

    def test_none_max_tokens_passed_through(self):
        calls = self._threshold_calls(None)
        assert calls
        assert all(mt is None for _, mt in calls), calls


class TestBreakerTripsOnNonShrinkingCompaction:
    """The breaker must count a new-but-not-smaller result as a failure.

    Identity-based success meant `compact_failure_count` never left 0 on a
    compaction that returned a fresh list without shrinking it, so the
    `< 3` guard never engaged and every remaining step paid for another
    full-context compaction call. A 5-step run issued 9 of them.
    """

    def _count_compaction_calls(self, compact_side_effect, max_steps=6):
        calls: list[int] = []

        def counting(messages, **kw):
            calls.append(1)
            return compact_side_effect(messages, **kw)

        responses = [_tool_call_response(f"c{i}", i) for i in range(max_steps - 1)]
        responses.append(_text_response())
        with (
            patch("agentic.agent.agent.litellm") as mock_litellm,
            patch("agentic.agent.agent.get_context_threshold", return_value=0),
            patch("agentic.agent.agent.compact_messages", side_effect=counting),
        ):
            mock_litellm.completion.side_effect = responses
            agent = Agent(model="gpt-4o-mini", system_prompt="test")
            agent.run(_HISTORY, tools={"probe": _probe_tool()}, max_steps=max_steps)
        return len(calls)

    def test_new_but_not_smaller_result_trips_the_breaker(self):
        # Fresh list every call, same content: no work done.
        n = self._count_compaction_calls(
            lambda messages, **kw: [dict(m) for m in messages]
        )
        assert n <= 3, f"breaker never engaged: {n} compaction calls"

    def test_growing_result_trips_the_breaker(self):
        # Strictly worse than no-op — must not be read as success either.
        n = self._count_compaction_calls(
            lambda messages, **kw: [
                *[dict(m) for m in messages],
                {"role": "user", "content": "bloat " + "z" * 5000},
            ]
        )
        assert n <= 3, f"breaker never engaged: {n} compaction calls"
