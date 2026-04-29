"""Tests for agentic.llm.streaming.accumulate_stream."""

import json
import threading

import pytest

from tests.fixtures.streams import (
    FakeChoice,
    FakeChunk,
    FakeDelta,
    FakeFunction,
    FakeToolCallDelta,
    content_chunks,
    fake_stream,
    finish_chunk,
    reasoning_chunks,
    role_chunk,
    tool_call_chunks,
    usage_chunk,
)


def test_empty_stream():
    from agentic.llm.streaming import accumulate_stream

    msg, finish_reason, usage = accumulate_stream(fake_stream())

    assert msg.content == ""
    assert msg.reasoning_content == ""
    assert msg.tool_calls == []
    assert finish_reason is None
    assert usage is None


def test_content_only_stream_invokes_callback_per_fragment():
    from agentic.llm.streaming import accumulate_stream

    received: list[str] = []
    msg, finish_reason, _ = accumulate_stream(
        fake_stream(
            role_chunk(),
            *content_chunks("hello world", fragments=3),
            finish_chunk("stop"),
        ),
        on_content_delta=received.append,
    )

    assert msg.content == "hello world"
    assert finish_reason == "stop"
    # Each non-empty fragment triggered the callback
    assert "".join(received) == "hello world"
    assert all(p != "" for p in received)


def test_reasoning_only_stream_invokes_reasoning_callback():
    from agentic.llm.streaming import accumulate_stream

    received: list[str] = []
    msg, finish_reason, _ = accumulate_stream(
        fake_stream(
            role_chunk(),
            *reasoning_chunks("thinking thoughts", fragments=2),
            finish_chunk("stop"),
        ),
        on_reasoning_delta=received.append,
    )

    assert msg.reasoning_content == "thinking thoughts"
    assert msg.content == ""
    assert "".join(received) == "thinking thoughts"


def test_no_callbacks_provided_works():
    from agentic.llm.streaming import accumulate_stream

    msg, _, _ = accumulate_stream(
        fake_stream(
            role_chunk(), *content_chunks("hi", fragments=1), finish_chunk("stop")
        ),
    )

    assert msg.content == "hi"  # accumulated even without callbacks


def test_finish_reason_propagated_from_last_chunk():
    """M3 v3: finish_reason must be returned (output_recovery depends on it)."""
    from agentic.llm.streaming import accumulate_stream

    _, finish_reason, _ = accumulate_stream(
        fake_stream(
            role_chunk(),
            *content_chunks("partial", fragments=1),
            finish_chunk("length"),
        ),
    )

    assert finish_reason == "length"


def test_provider_variance_no_reasoning_field():
    """Stream missing reasoning_content (e.g. GPT-4o) → empty reasoning, no errors."""
    from agentic.llm.streaming import accumulate_stream

    msg, _, _ = accumulate_stream(
        fake_stream(role_chunk(), *content_chunks("hello"), finish_chunk("stop")),
    )

    assert msg.reasoning_content == ""


def test_single_tool_call_fragmented_args():
    """B1 v3 fix: tc.function.name and tc.function.arguments are ATTRIBUTE access
    (not dict subscript); arguments is a JSON STRING (not a parsed dict) — matches
    LiteLLM's normal response shape so agent.py's tool-dispatch path works."""
    from agentic.llm.streaming import accumulate_stream

    msg, finish_reason, _ = accumulate_stream(
        fake_stream(
            role_chunk(),
            *tool_call_chunks(
                name="search_kb", args={"query": "what is X"}, frag_count=4
            ),
            finish_chunk("tool_calls"),
        ),
    )

    assert finish_reason == "tool_calls"
    assert len(msg.tool_calls) == 1
    tc = msg.tool_calls[0]
    assert tc.id == "call_0"
    assert tc.function.name == "search_kb"  # attribute access
    assert isinstance(tc.function.arguments, str)  # JSON string, NOT parsed dict
    assert json.loads(tc.function.arguments) == {"query": "what is X"}


def test_parallel_tool_calls():
    from agentic.llm.streaming import accumulate_stream

    msg, _, _ = accumulate_stream(
        fake_stream(
            role_chunk(),
            *tool_call_chunks(
                name="search_kb", args={"q": "first"}, index=0, frag_count=2
            ),
            *tool_call_chunks(name="lookup", args={"id": 42}, index=1, frag_count=2),
            finish_chunk("tool_calls"),
        ),
    )

    assert len(msg.tool_calls) == 2
    by_name = {
        tc.function.name: json.loads(tc.function.arguments) for tc in msg.tool_calls
    }
    assert by_name == {"search_kb": {"q": "first"}, "lookup": {"id": 42}}


def test_mixed_content_reasoning_toolcall():
    from agentic.llm.streaming import accumulate_stream

    received_content: list[str] = []
    received_reasoning: list[str] = []
    msg, _, _ = accumulate_stream(
        fake_stream(
            role_chunk(),
            *reasoning_chunks("Let me search.", fragments=2),
            *content_chunks("Searching...", fragments=2),
            *tool_call_chunks(name="search_kb", args={"q": "x"}, frag_count=2),
            finish_chunk("tool_calls"),
        ),
        on_content_delta=received_content.append,
        on_reasoning_delta=received_reasoning.append,
    )

    assert msg.content == "Searching..."
    assert msg.reasoning_content == "Let me search."
    assert len(msg.tool_calls) == 1
    assert "".join(received_content) == "Searching..."
    assert "".join(received_reasoning) == "Let me search."


def test_truncated_tool_call_args_raises_truncation_error():
    """M4 fix: malformed JSON in args → StreamTruncationError."""
    from agentic.llm.streaming import StreamTruncationError, accumulate_stream

    # Manually craft chunks where args ends mid-JSON ('{"query":')
    bad_chunks = [
        role_chunk(),
        FakeChunk(
            choices=[
                FakeChoice(
                    delta=FakeDelta(
                        tool_calls=[
                            FakeToolCallDelta(
                                index=0,
                                id="call_0",
                                function=FakeFunction(name="search_kb", arguments=""),
                            )
                        ]
                    )
                )
            ]
        ),
        FakeChunk(
            choices=[
                FakeChoice(
                    delta=FakeDelta(
                        tool_calls=[
                            FakeToolCallDelta(
                                index=0, function=FakeFunction(arguments='{"query":')
                            )
                        ]
                    )
                )
            ]
        ),
        finish_chunk("tool_calls"),
    ]

    with pytest.raises(StreamTruncationError) as excinfo:
        accumulate_stream(fake_stream(*bad_chunks))

    # Truncation error should carry partial buffers (empty in this case — no content streamed)
    assert excinfo.value.partial_content == ""


def test_abort_mid_stream():
    """abort_signal mid-iteration → AbortedError."""
    from agentic.llm.streaming import AbortedError, accumulate_stream

    abort = threading.Event()

    def aborting_stream():
        yield role_chunk()
        # Simulate an external abort here
        abort.set()
        yield content_chunks("never seen")[0]

    with pytest.raises(AbortedError):
        accumulate_stream(aborting_stream(), abort_signal=abort)


def test_usage_captured_from_final_chunk():
    """Q5: final chunk's usage propagates to the returned tuple."""
    from agentic.llm.streaming import accumulate_stream

    _, _, usage = accumulate_stream(
        fake_stream(
            role_chunk(),
            *content_chunks("hi"),
            finish_chunk("stop"),
            usage_chunk(prompt_tokens=10, completion_tokens=5),
        ),
    )

    assert usage is not None
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 5
    assert usage["total_tokens"] == 15


def test_stream_partial_error_carries_buffers():
    """M5: a generic mid-stream exception propagates as StreamPartialError
    with the buffers populated up to the failure point."""
    from agentic.llm.streaming import StreamPartialError, accumulate_stream

    def failing_stream():
        yield role_chunk()
        yield from content_chunks("hello world", fragments=2)
        # Simulate provider connection drop
        raise ConnectionError("upstream closed")

    with pytest.raises(StreamPartialError) as excinfo:
        accumulate_stream(failing_stream())

    assert "hello world" in excinfo.value.partial_content
    # Reasoning was empty in this stream
    assert excinfo.value.partial_reasoning == ""
