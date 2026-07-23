import itertools
from types import SimpleNamespace
from unittest.mock import patch

from agentic import Agent
from agentic.agent.tools import BuiltinTool
from agentic.llm.routing import maybe_route_through_responses


def _mock_tool_call_response(tool_name, arguments, call_id="call_1"):
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
                                name=tool_name, arguments=arguments
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


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


class TestModelFallback:
    @patch("agentic.agent.agent.litellm")
    def test_fallback_on_rate_limit(self, mock_litellm):
        from litellm.exceptions import RateLimitError

        mock_litellm.completion.side_effect = [
            RateLimitError(
                message="rate limited",
                llm_provider="anthropic",
                model="claude-sonnet-4-6",
            ),
            _mock_completion_response("Fallback response."),
        ]
        agent = Agent(model="claude-sonnet-4-6", system_prompt="test")
        output = agent.run("hello", fallback_model="gpt-4.1-mini")
        assert output.content == "Fallback response."
        assert output.status.is_success()
        assert mock_litellm.completion.call_count == 2
        second_call = mock_litellm.completion.call_args_list[1]
        assert second_call.kwargs["model"] == "gpt-4.1-mini"

    @patch("agentic.agent.agent.litellm")
    def test_no_fallback_without_config(self, mock_litellm):
        from litellm.exceptions import RateLimitError

        mock_litellm.completion.side_effect = RateLimitError(
            message="rate limited", llm_provider="anthropic", model="claude-sonnet-4-6"
        )
        agent = Agent(model="claude-sonnet-4-6", system_prompt="test")
        output = agent.run("hello")
        assert output.status.value == "failed"

    @patch("agentic.agent.agent.litellm")
    def test_fallback_is_free_turn(self, mock_litellm):
        from litellm.exceptions import RateLimitError

        mock_litellm.completion.side_effect = [
            RateLimitError(
                message="rate limited",
                llm_provider="anthropic",
                model="claude-sonnet-4-6",
            ),
            _mock_completion_response("ok"),
        ]
        agent = Agent(model="claude-sonnet-4-6", system_prompt="test")
        output = agent.run("hello", fallback_model="gpt-4.1-mini")
        assert output.steps == 1  # Free recovery


class TestMaxOutputRecovery:
    @patch("agentic.agent.agent.litellm")
    def test_truncated_response_retried(self, mock_litellm):
        truncated = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Partial respon", role="assistant", tool_calls=None
                    ),
                    finish_reason="length",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )
        continued = _mock_completion_response("se. Here's the rest.")
        mock_litellm.completion.side_effect = [truncated, continued]

        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("Write something long")

        assert output.status.is_success()
        assert mock_litellm.completion.call_count == 2
        second_messages = mock_litellm.completion.call_args_list[1].kwargs["messages"]
        continue_msgs = [
            m
            for m in second_messages
            if "truncated" in m.get("content", "").lower()
            or "continue" in m.get("content", "").lower()
        ]
        assert len(continue_msgs) > 0

    @patch("agentic.agent.agent.litellm")
    def test_max_3_recovery_attempts(self, mock_litellm):
        truncated = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Partial", role="assistant", tool_calls=None
                    ),
                    finish_reason="length",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )
        mock_litellm.completion.side_effect = [
            truncated,
            truncated,
            truncated,
            truncated,
        ]

        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("Write something long")

        assert mock_litellm.completion.call_count == 4
        assert output.content is not None


class TestReactiveCompact:
    @patch("agentic.agent.agent.litellm")
    def test_reactive_compact_on_prompt_too_long(self, mock_litellm):
        mock_litellm.completion.side_effect = [
            Exception("prompt is too long: 200000 tokens > 100000 maximum"),
            _mock_completion_response("Compacted response."),
        ]
        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("hello")
        assert output.content == "Compacted response."
        assert output.status.is_success()
        assert mock_litellm.completion.call_count == 2

    @patch("agentic.agent.agent.litellm")
    def test_reactive_compact_is_free_turn(self, mock_litellm):
        mock_litellm.completion.side_effect = [
            Exception("prompt is too long"),
            _mock_completion_response("ok"),
        ]
        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("hello")
        assert output.steps == 1


def _looping_tool():
    return BuiltinTool(
        name="probe",
        description="probe",
        input_schema={"type": "object", "properties": {"i": {"type": "integer"}}},
        handler=lambda args, ctx: "ok",
    )


def _always_over_threshold():
    """Patch the loop so every compaction check fires."""
    return (
        patch("agentic.agent.agent.estimate_token_count", return_value=10_000_000),
        patch("agentic.agent.agent.get_context_threshold", return_value=0),
    )


class TestCompactionCircuitBreaker:
    """compact_messages returning its input == no progress; stop after 3 tries.

    Without the identity check the counter stays 0 forever and the agent burns
    a full-context compaction call at every step until max_steps.
    """

    @patch("agentic.agent.agent.litellm")
    def test_no_progress_compaction_capped_at_three_attempts(self, mock_litellm):
        # 5 distinct tool calls (distinct args dodge doom-loop detection),
        # then a final answer.
        mock_litellm.completion.side_effect = [
            _mock_tool_call_response("probe", f'{{"i": {i}}}', call_id=f"c{i}")
            for i in range(5)
        ] + [_mock_completion_response("done")]

        est, thr = _always_over_threshold()
        with (
            est,
            thr,
            patch(
                "agentic.agent.agent.compact_messages",
                side_effect=lambda msgs, **kw: msgs,  # no progress: same object back
            ) as compact,
        ):
            agent = Agent(model="gpt-4o-mini", system_prompt="test")
            output = agent.run("hi", tools={"probe": _looping_tool()}, max_steps=6)

        assert output.status.is_success()
        assert compact.call_count == 3

    @patch("agentic.agent.agent.litellm")
    def test_reactive_no_progress_counts_toward_the_cap(self, mock_litellm):
        # First call fails with prompt_too_long -> reactive compaction path.
        mock_litellm.completion.side_effect = (
            [
                Exception("prompt is too long: 200000 tokens > 100000 maximum"),
            ]
            + [
                _mock_tool_call_response("probe", f'{{"i": {i}}}', call_id=f"c{i}")
                for i in range(5)
            ]
            + [_mock_completion_response("done")]
        )

        est, thr = _always_over_threshold()
        with (
            est,
            thr,
            patch(
                "agentic.agent.agent.compact_messages",
                side_effect=lambda msgs, **kw: msgs,
            ) as compact,
        ):
            agent = Agent(model="gpt-4o-mini", system_prompt="test")
            output = agent.run("hi", tools={"probe": _looping_tool()}, max_steps=6)

        assert output.status.is_success()
        # 1 proactive + 1 reactive + 1 proactive = cap reached.
        # If the reactive path still reset the counter to 0 this would be 4+.
        assert compact.call_count == 3


PRUNE_PLACEHOLDER = "[Previous tool result removed to save context]"


def _history_with_old_tool_result(filler_len: int = 600):
    """A conversation whose oldest tool result is prunable.

    Four user turns means ``prune_messages`` (keep_last_n_turns=3) protects
    everything from the second user turn on, so only the first tool result is
    rewritten to the placeholder.
    """
    filler = "x" * filler_len
    return [
        {"role": "user", "content": f"start {filler}"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "old_1",
                    "type": "function",
                    "function": {"name": "probe", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "old_1",
            "content": f"BIG_TOOL_RESULT {filler}",
        },
        {"role": "user", "content": f"q1 {filler}"},
        {"role": "assistant", "content": f"a1 {filler}"},
        {"role": "user", "content": f"q2 {filler}"},
        {"role": "assistant", "content": f"a2 {filler}"},
        {"role": "user", "content": "now answer"},
    ]


def _has_placeholder(messages):
    return any(m.get("content") == PRUNE_PLACEHOLDER for m in messages)


class TestProactiveCompaction:
    """Proactive path: prune-first for the decision, compact the un-pruned list."""

    @patch("agentic.agent.agent.litellm")
    def test_no_compaction_when_pruning_alone_is_enough(self, mock_litellm):
        # Pruning is free — if it gets the estimate under the threshold we must
        # not pay for a full-context compaction call.
        mock_litellm.completion.side_effect = [_mock_completion_response("done")]

        def _estimate(messages):
            return 10 if _has_placeholder(messages) else 10_000_000

        with (
            patch("agentic.agent.agent.estimate_token_count", side_effect=_estimate),
            patch("agentic.agent.agent.get_context_threshold", return_value=100),
            patch("agentic.agent.agent.compact_messages") as compact,
        ):
            agent = Agent(model="gpt-4o-mini", system_prompt="test")
            output = agent.run(
                _history_with_old_tool_result(),
                tools={"probe": _looping_tool()},
                max_steps=3,
            )

        assert output.status.is_success()
        assert compact.call_count == 0

    @patch("agentic.agent.agent.litellm")
    def test_compaction_receives_unpruned_messages(self, mock_litellm):
        # Compaction replays the conversation's own prefix for the prompt
        # cache; handing it pruned messages makes that call cache-cold.
        mock_litellm.completion.side_effect = [_mock_completion_response("done")]

        est, thr = _always_over_threshold()
        with (
            est,
            thr,
            patch(
                "agentic.agent.agent.compact_messages",
                side_effect=lambda msgs, **kw: [
                    {"role": "user", "content": "summary"},
                    {"role": "user", "content": "continue"},
                ],
            ) as compact,
        ):
            agent = Agent(model="gpt-4o-mini", system_prompt="test")
            output = agent.run(
                _history_with_old_tool_result(),
                tools={"probe": _looping_tool()},
                max_steps=3,
            )

        assert output.status.is_success()
        assert compact.call_count >= 1
        sent = compact.call_args_list[0].args[0]
        assert not _has_placeholder(sent)
        assert any("BIG_TOOL_RESULT" in (m.get("content") or "") for m in sent)

    @patch("agentic.agent.agent.litellm")
    def test_proactive_compaction_not_redone_by_main_loop_check(self, mock_litellm):
        # Proactive compaction must persist into state.messages, otherwise the
        # Phase 5 check rebuilds from the uncompacted history and compacts the
        # very same conversation a second time in one step.
        mock_litellm.completion.side_effect = [
            _mock_tool_call_response("probe", '{"i": 0}', call_id="c0"),
            _mock_completion_response("done"),
        ]

        with (
            patch("agentic.agent.agent.get_context_threshold", return_value=100),
            patch(
                "agentic.agent.agent.compact_messages",
                side_effect=lambda msgs, **kw: [
                    {"role": "user", "content": "summary"},
                    {"role": "user", "content": "continue"},
                ],
            ) as compact,
        ):
            agent = Agent(model="gpt-4o-mini", system_prompt="test")
            output = agent.run(
                _history_with_old_tool_result(),
                tools={"probe": _looping_tool()},
                max_steps=3,
            )

        assert output.status.is_success()
        assert compact.call_count == 1


class TestCompactionKwargsAtEveryCallSite:
    """`model` and `tools` must be pinned at ALL THREE compaction sites.

    The existing coverage drives only the proactive site, so dropping the
    routing or the tool schemas at the reactive or Phase-5 sites was invisible.
    Those are exactly the sites that fire under context pressure — when the
    history is largest and a cache-cold full-context call costs the most.

    Both kwargs are prefix content: tool definitions lead the cached prefix,
    and the routed model name selects the API surface the cache is keyed on.
    Sending either differently from the real call makes the compaction call
    fully cache-cold.
    """

    # A reasoning model with effort set, so the routed model string differs
    # from the raw one (`openai/responses/gpt-5.4` vs `gpt-5.4`). On a
    # non-reasoning model routing is a no-op and dropping it would be
    # invisible — the exact hole in the earlier coverage.
    _REASONING_MODEL = "gpt-5.4"
    _ROUTED = maybe_route_through_responses("gpt-5.4", "medium")

    @patch("agentic.agent.agent.litellm")
    def test_reactive_site_forwards_model_and_tools(self, mock_litellm):
        assert self._ROUTED != self._REASONING_MODEL  # guards the no-op trap
        mock_litellm.completion.side_effect = [
            Exception("prompt is too long: 200000 tokens > 100000 maximum"),
            _mock_completion_response("ok"),
        ]
        with patch(
            "agentic.agent.agent.compact_messages",
            side_effect=lambda msgs, **kw: [{"role": "user", "content": "s"}],
        ) as compact:
            agent = Agent(
                model=self._REASONING_MODEL,
                system_prompt="test",
                reasoning_effort="medium",
            )
            output = agent.run("hello", tools={"probe": _looping_tool()}, max_steps=3)

        assert output.status.is_success()
        assert compact.call_count >= 1
        kw = compact.call_args_list[0].kwargs
        assert kw["model"] == self._ROUTED
        assert kw["tools"], "tool schemas must reach the reactive compaction call"
        assert {t["function"]["name"] for t in kw["tools"]} == {"probe"}

    @patch("agentic.agent.agent.litellm")
    def test_phase5_site_forwards_model_and_tools(self, mock_litellm):
        assert self._ROUTED != self._REASONING_MODEL
        # Under threshold at the proactive site, over it at Phase 5, so the
        # only compaction call in the run is Phase 5's.
        mock_litellm.completion.side_effect = [
            _mock_tool_call_response("probe", '{"i": 0}', call_id="c0"),
            _mock_completion_response("done"),
        ]
        thresholds = itertools.count()

        def fake_threshold(model, max_output_tokens=None):
            # site 1 asks first, then site 3, per step.
            return 10**9 if next(thresholds) % 2 == 0 else 0

        with (
            patch(
                "agentic.agent.agent.get_context_threshold", side_effect=fake_threshold
            ),
            patch(
                "agentic.agent.agent.compact_messages",
                side_effect=lambda msgs, **kw: [{"role": "user", "content": "s"}],
            ) as compact,
        ):
            agent = Agent(
                model=self._REASONING_MODEL,
                system_prompt="test",
                reasoning_effort="medium",
            )
            output = agent.run("hello", tools={"probe": _looping_tool()}, max_steps=3)

        assert output.status.is_success()
        assert compact.call_count >= 1
        kw = compact.call_args_list[0].kwargs
        assert kw["model"] == self._ROUTED
        assert kw["tools"], "tool schemas must reach the Phase 5 compaction call"
        assert {t["function"]["name"] for t in kw["tools"]} == {"probe"}
