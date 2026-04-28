"""Tests for the Agent ReAct loop with tool calling."""

import threading
from types import SimpleNamespace
from unittest.mock import patch

from agentic import Agent
from agentic.agent.tools import BuiltinTool
from agentic.execution.context import ExecutionContext


def _mock_completion_response(content, finish_reason="stop"):
    """Build a mock litellm response."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    role="assistant",
                    tool_calls=None,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )


def _mock_tool_call_response(tool_name, arguments, call_id="call_1"):
    """Build a mock litellm response with a tool call."""
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
                                name=tool_name,
                                arguments=arguments,
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )


def _mock_streaming_response(
    content: str = "",
    reasoning: str = "",
    tool_calls: list[dict] | None = None,
    finish_reason: str = "stop",
):
    """Streaming variant of _mock_completion_response.

    Returns an iterator of FakeChunk objects matching what
    ``litellm.completion(stream=True)`` yields.
    """
    from tests.fixtures.streams import (
        content_chunks,
        fake_stream,
        finish_chunk,
        reasoning_chunks,
        role_chunk,
        tool_call_chunks,
        usage_chunk,
    )

    chunks = [role_chunk()]
    if reasoning:
        chunks.extend(reasoning_chunks(reasoning, fragments=3))
    if content:
        chunks.extend(content_chunks(content, fragments=3))
    if tool_calls:
        for i, tc in enumerate(tool_calls):
            chunks.extend(
                tool_call_chunks(
                    name=tc["name"],
                    args=tc.get("args", {}),
                    index=i,
                    frag_count=2,
                )
            )
    chunks.append(finish_chunk(finish_reason))
    chunks.append(usage_chunk(prompt_tokens=10, completion_tokens=5))
    return fake_stream(*chunks)


class TestReactLoopNoTools:
    """Without tools, the loop should run one iteration (backward compat)."""

    @patch("agentic.agent.agent.litellm")
    def test_single_step_no_tools(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_completion_response("Hello!")
        agent = Agent(model="gpt-4o-mini", system_prompt="You are helpful.")
        output = agent.run("Hi")
        assert output.content == "Hello!"
        assert output.status.is_success()
        assert output.steps == 1
        assert output.tool_calls == []
        mock_litellm.completion.assert_called_once()


class TestReactLoopWithTools:
    """LLM calls a tool, gets result, then produces final response."""

    @patch("agentic.agent.agent.litellm")
    def test_tool_call_then_response(self, mock_litellm):
        # Step 1: LLM wants to call a tool
        tool_call_response = _mock_tool_call_response("get_time", "{}")
        # Step 2: LLM produces final response
        final_response = _mock_completion_response("The time is 3pm.")

        mock_litellm.completion.side_effect = [tool_call_response, final_response]

        tool = BuiltinTool(
            name="get_time",
            description="Get current time",
            input_schema={"type": "object", "properties": {}},
            handler=lambda args, ctx: "15:00",
        )

        agent = Agent(model="gpt-4o-mini", system_prompt="You are helpful.")
        output = agent.run("What time is it?", tools={"get_time": tool})

        assert output.content == "The time is 3pm."
        assert output.steps == 2
        assert len(output.tool_calls) == 1
        assert output.tool_calls[0].tool_name == "get_time"
        assert output.tool_calls[0].result == "15:00"
        assert mock_litellm.completion.call_count == 2


class TestReactLoopSafety:
    """Max steps and doom loop detection."""

    @patch("agentic.agent.agent.litellm")
    def test_max_steps_enforced(self, mock_litellm):
        # LLM always wants to call the tool
        tool_call_response = _mock_tool_call_response("noop", "{}")
        final_response = _mock_completion_response("I ran out of steps.")

        # 2 tool-calling steps + 1 final (max_steps=3)
        mock_litellm.completion.side_effect = [
            tool_call_response,
            tool_call_response,
            final_response,
        ]

        tool = BuiltinTool(
            name="noop",
            description="Does nothing",
            input_schema={"type": "object"},
            handler=lambda args, ctx: "ok",
        )

        agent = Agent(model="gpt-4o-mini", system_prompt="You are helpful.")
        output = agent.run("Do stuff", tools={"noop": tool}, max_steps=3)

        assert output.steps == 3
        # On the last step, tools should NOT be passed to litellm
        last_call_kwargs = mock_litellm.completion.call_args_list[-1]
        assert last_call_kwargs.kwargs.get("tools") is None

    @patch("agentic.agent.agent.litellm")
    def test_doom_loop_detection(self, mock_litellm):
        def make_tool_call():
            return _mock_tool_call_response("search", '{"q": "same"}')

        mock_litellm.completion.side_effect = [
            make_tool_call(),
            make_tool_call(),
            make_tool_call(),
        ]

        tool = BuiltinTool(
            name="search",
            description="Search",
            input_schema={
                "type": "object",
                "properties": {"q": {"type": "string"}},
            },
            handler=lambda args, ctx: "no results",
        )

        agent = Agent(model="gpt-4o-mini", system_prompt="You are helpful.")
        output = agent.run("Find it", tools={"search": tool}, max_steps=25)

        assert output.status.value == "failed"
        assert (
            "doom loop" in output.error.lower() or "identical" in output.error.lower()
        )


class TestReactLoopEvents:
    """Event emission during the ReAct loop."""

    @patch("agentic.agent.agent.litellm")
    def test_events_emitted(self, mock_litellm):
        tool_call_response = _mock_tool_call_response("ping", "{}")
        final_response = _mock_completion_response("pong")
        mock_litellm.completion.side_effect = [tool_call_response, final_response]

        events = []
        ctx = ExecutionContext(execution_id="test", on_event=lambda e: events.append(e))
        tool = BuiltinTool(
            name="ping",
            description="Ping",
            input_schema={"type": "object"},
            handler=lambda args, ctx: "pong",
        )

        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("test", context=ctx, tools={"ping": tool})

        event_types = [e["type"] for e in events]
        assert "step_started" in event_types
        assert "tool_call" in event_types
        assert "tool_result" in event_types
        assert "step_completed" in event_types

        # output.events must be populated and match the callback's event list
        assert len(output.events) > 0
        output_event_types = [e["type"] for e in output.events]
        assert "step_started" in output_event_types
        assert "tool_call" in output_event_types
        assert "tool_result" in output_event_types
        assert "step_completed" in output_event_types
        assert output.events == events


class TestAbortIntegration:
    """Abort signal integration with the ReAct loop."""

    @patch("agentic.agent.agent.litellm")
    def test_aborted_before_llm_call(self, mock_litellm):
        """When abort_signal is already set, run() returns cancelled without calling LLM."""
        signal = threading.Event()
        signal.set()  # Already aborted
        ctx = ExecutionContext(execution_id="test", abort_signal=signal)

        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("test", context=ctx)

        assert output.status.value in ("cancelled", "failed")
        mock_litellm.completion.assert_not_called()


class TestStreamingKillSwitch:
    """Regression guard: AGENT_LLM_STREAMING_ENABLED=false preserves today's behavior."""

    @patch("agentic.agent.agent.litellm")
    def test_streaming_disabled_via_env_preserves_today_behavior(
        self, mock_litellm, monkeypatch
    ):
        """Kill-switch off -> behavior identical to pre-PR. Regression guard.

        With AGENT_LLM_STREAMING_ENABLED=false, agent.run() must:
          - Use the non-streaming litellm.completion path (no stream=True kwarg).
          - Return AgentOutput.content matching the mock response exactly.
          - Emit zero content_delta / reasoning_delta events via on_event.
        """
        # Conftest already pins to "false"; explicit here for clarity / belt-and-braces.
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "false")

        mock_litellm.completion.return_value = _mock_completion_response("hi")

        events = []
        ctx = ExecutionContext(execution_id="test", on_event=lambda e: events.append(e))

        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("Hello", context=ctx)

        # AgentOutput shape matches today's non-streaming behavior.
        assert output.content == "hi"
        assert output.status.is_success()
        assert output.steps == 1
        assert output.tool_calls == []

        # litellm was called in non-streaming mode.
        mock_litellm.completion.assert_called_once()
        call_kwargs = mock_litellm.completion.call_args.kwargs
        assert not call_kwargs.get(
            "stream"
        ), "kill-switch off must not pass stream=True"

        # No streaming-specific events were emitted.
        emitted_types = [e["type"] for e in events]
        assert "content_delta" not in emitted_types
        assert "reasoning_delta" not in emitted_types
        assert output.events == events


class TestStreamingContentDeltaEmission:
    """Streaming on -> content_delta events fire; line ~563 reasoning emit is suppressed.

    Q1 resolution (issue #106): when streaming is enabled, the LLM's intermediary
    prose alongside tool calls is already streamed via content_delta to the chat
    bubble — re-emitting it as a terminal reasoning event would duplicate.
    """

    @patch("agentic.agent.agent.litellm")
    def test_streaming_emits_content_delta_events(self, mock_litellm, monkeypatch):
        """Streaming on -> content_delta events fire for each fragment, no
        duplicate reasoning event from line ~563 (Q1 fix)."""
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")

        # Step 1: streaming response with content + tool call (triggers
        # has_tool_calls=True AND assistant_msg.content non-empty path).
        # Step 2: streaming response with content only (loop terminates).
        step1 = _mock_streaming_response(
            content="thinking out loud",
            tool_calls=[{"name": "search_kb", "args": {"q": "x"}}],
            finish_reason="tool_calls",
        )
        step2 = _mock_streaming_response(content="done", finish_reason="stop")
        mock_litellm.completion.side_effect = [step1, step2]

        tool = BuiltinTool(
            name="search_kb",
            description="Search the KB",
            input_schema={
                "type": "object",
                "properties": {"q": {"type": "string"}},
            },
            handler=lambda args, ctx: "result",
        )

        events = []
        ctx = ExecutionContext(execution_id="test", on_event=lambda e: events.append(e))

        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("What's in the KB?", context=ctx, tools={"search_kb": tool})

        assert output.status.is_success()

        # 1. content_delta events fire for each fragment.
        content_deltas = [e for e in events if e.get("type") == "content_delta"]
        assert len(content_deltas) > 0, "expected at least one content_delta event"

        # 2. Concatenated content_delta deltas equal the prose from step 1.
        # (step 2's "done" deltas also flow, so check that step 1's prose is a
        # contiguous substring of the concatenation, OR sum to the full string.)
        joined = "".join(e["delta"] for e in content_deltas)
        assert (
            "thinking out loud" in joined
        ), f"expected step-1 prose in joined deltas, got: {joined!r}"

        # 3. No reasoning event WITHOUT a "source" field (line ~563 must not fire
        # when streaming is on). The reasoning emit at line ~552 (with
        # source="thinking") would not fire either since the mock has no
        # reasoning_content.
        bare_reasoning_events = [
            e for e in events if e.get("type") == "reasoning" and "source" not in e
        ]
        assert bare_reasoning_events == [], (
            f"line ~563 reasoning emit must be gated off when streaming is on; "
            f"got: {bare_reasoning_events}"
        )

    @patch("agentic.agent.agent.litellm")
    def test_line_506_emits_when_streaming_disabled(self, mock_litellm, monkeypatch):
        """Kill switch off -> line ~563 still fires for has_tool_calls + content
        steps. Regression guard for today's behavior."""
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "false")

        # Non-streaming response with BOTH content and a tool call.
        step1 = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="let me search for that",
                        role="assistant",
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                type="function",
                                function=SimpleNamespace(
                                    name="search_kb",
                                    arguments='{"q": "x"}',
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )
        step2 = _mock_completion_response("done")
        mock_litellm.completion.side_effect = [step1, step2]

        tool = BuiltinTool(
            name="search_kb",
            description="Search the KB",
            input_schema={
                "type": "object",
                "properties": {"q": {"type": "string"}},
            },
            handler=lambda args, ctx: "result",
        )

        events = []
        ctx = ExecutionContext(execution_id="test", on_event=lambda e: events.append(e))

        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("What's in the KB?", context=ctx, tools={"search_kb": tool})

        assert output.status.is_success()

        # Line ~563 must fire: a reasoning event with NO source field, content =
        # the assistant's prose alongside the tool call.
        bare_reasoning_events = [
            e for e in events if e.get("type") == "reasoning" and "source" not in e
        ]
        assert (
            len(bare_reasoning_events) == 1
        ), f"expected exactly one bare reasoning event, got: {bare_reasoning_events}"
        assert bare_reasoning_events[0]["content"] == "let me search for that"


class TestMidStreamErrorSyntheticEvents:
    """T10/M5: when accumulate_stream raises StreamPartialError mid-iteration,
    agent.py emits synthetic terminal events (chunk + reasoning + error) so
    the FE retains partial context and the route layer can persist them."""

    @patch("agentic.agent.agent.litellm")
    def test_mid_stream_error_emits_chunk_and_reasoning_then_error(
        self, mock_litellm, monkeypatch
    ):
        """M5/M2: stream raises mid-iteration with both partial content and
        partial reasoning -> synthetic terminal events emitted in order:
        chunk, reasoning (source=thinking), error."""
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")

        from tests.fixtures.streams import (
            content_chunks,
            reasoning_chunks,
            role_chunk,
        )

        def failing_stream():
            yield role_chunk()
            yield from reasoning_chunks("thinking hard", fragments=2)
            yield from content_chunks("partial answer", fragments=2)
            raise ConnectionError("upstream connection dropped")

        mock_litellm.completion.return_value = failing_stream()

        events = []
        ctx = ExecutionContext(execution_id="test", on_event=lambda e: events.append(e))
        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("Hello", context=ctx)

        # Outer except catches the re-raised StreamPartialError -> FAILED
        assert output.status.value == "failed"

        # Live deltas fired before the error
        content_deltas = [e for e in events if e.get("type") == "content_delta"]
        reasoning_deltas = [e for e in events if e.get("type") == "reasoning_delta"]
        assert len(content_deltas) > 0, "expected content_delta events before error"
        assert len(reasoning_deltas) > 0, "expected reasoning_delta events before error"

        # Find the synthetic terminal events fired AFTER the deltas
        # (they live alongside step_started/etc., so filter by type and
        # check ordering by index in the events list).
        synth_chunk = [e for e in events if e.get("type") == "chunk"]
        synth_reasoning = [
            e
            for e in events
            if e.get("type") == "reasoning" and e.get("source") == "thinking"
        ]
        error_events = [e for e in events if e.get("type") == "error"]

        assert (
            len(synth_chunk) == 1
        ), f"expected exactly 1 synthetic chunk event, got: {synth_chunk}"
        assert (
            len(synth_reasoning) == 1
        ), f"expected exactly 1 synthetic reasoning event, got: {synth_reasoning}"
        assert (
            len(error_events) == 1
        ), f"expected exactly 1 error event, got: {error_events}"

        # Synthetic chunk content == joined content fragments
        assert synth_chunk[0]["content"] == "partial answer"
        # Synthetic reasoning content == joined reasoning fragments, source=thinking
        assert synth_reasoning[0]["content"] == "thinking hard"
        assert synth_reasoning[0]["source"] == "thinking"

        # Order: chunk -> reasoning -> error
        chunk_idx = events.index(synth_chunk[0])
        reasoning_idx = events.index(synth_reasoning[0])
        error_idx = events.index(error_events[0])
        assert chunk_idx < reasoning_idx < error_idx, (
            f"expected chunk -> reasoning -> error order, got indices "
            f"chunk={chunk_idx}, reasoning={reasoning_idx}, error={error_idx}"
        )

    @patch("agentic.agent.agent.litellm")
    def test_mid_stream_error_only_content(self, mock_litellm, monkeypatch):
        """Variant: stream errors after content but before reasoning ever fires.
        Expect synthetic chunk + error, NO synthetic reasoning."""
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")

        from tests.fixtures.streams import content_chunks, role_chunk

        def failing_stream():
            yield role_chunk()
            yield from content_chunks("hello world", fragments=2)
            raise ConnectionError("upstream closed")

        mock_litellm.completion.return_value = failing_stream()

        events = []
        ctx = ExecutionContext(execution_id="test", on_event=lambda e: events.append(e))
        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("Hi", context=ctx)

        assert output.status.value == "failed"

        synth_chunk = [e for e in events if e.get("type") == "chunk"]
        synth_reasoning = [
            e
            for e in events
            if e.get("type") == "reasoning" and e.get("source") == "thinking"
        ]
        error_events = [e for e in events if e.get("type") == "error"]

        assert len(synth_chunk) == 1, f"expected synthetic chunk, got: {synth_chunk}"
        assert synth_chunk[0]["content"] == "hello world"
        assert (
            synth_reasoning == []
        ), f"expected NO synthetic reasoning, got: {synth_reasoning}"
        assert len(error_events) == 1

    @patch("agentic.agent.agent.litellm")
    def test_mid_stream_error_only_reasoning(self, mock_litellm, monkeypatch):
        """Variant: stream errors after reasoning but before any content fragment.
        Expect synthetic reasoning + error, NO synthetic chunk (m3 v3 edge case)."""
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")

        from tests.fixtures.streams import reasoning_chunks, role_chunk

        def failing_stream():
            yield role_chunk()
            yield from reasoning_chunks("just thinking", fragments=2)
            raise ConnectionError("upstream closed")

        mock_litellm.completion.return_value = failing_stream()

        events = []
        ctx = ExecutionContext(execution_id="test", on_event=lambda e: events.append(e))
        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("Hi", context=ctx)

        assert output.status.value == "failed"

        synth_chunk = [e for e in events if e.get("type") == "chunk"]
        synth_reasoning = [
            e
            for e in events
            if e.get("type") == "reasoning" and e.get("source") == "thinking"
        ]
        error_events = [e for e in events if e.get("type") == "error"]

        assert synth_chunk == [], f"expected NO synthetic chunk, got: {synth_chunk}"
        assert (
            len(synth_reasoning) == 1
        ), f"expected synthetic reasoning, got: {synth_reasoning}"
        assert synth_reasoning[0]["content"] == "just thinking"
        assert synth_reasoning[0]["source"] == "thinking"
        assert len(error_events) == 1
