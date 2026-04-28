"""Tests for agentic.llm.streaming.accumulate_stream."""

from tests.fixtures.streams import (
    content_chunks,
    fake_stream,
    finish_chunk,
    reasoning_chunks,
    role_chunk,
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
