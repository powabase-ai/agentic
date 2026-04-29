"""Tests that accumulate_stream captures thinking_blocks and provider_specific_fields."""

from __future__ import annotations

from types import SimpleNamespace

from agentic.llm.streaming import accumulate_stream


def _delta(**kwargs):
    return SimpleNamespace(**kwargs)


def _chunk(*, delta=None, finish_reason=None, usage=None):
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_captures_thinking_blocks_across_chunks():
    """Anthropic thinking_blocks come as delta.thinking_blocks per chunk; the
    accumulator combines them by index."""
    chunks = [
        _chunk(
            delta=_delta(
                content=None,
                reasoning_content=None,
                tool_calls=None,
                thinking_blocks=[
                    {"index": 0, "type": "thinking", "thinking": "I should "}
                ],
                provider_specific_fields=None,
            )
        ),
        _chunk(
            delta=_delta(
                content=None,
                reasoning_content=None,
                tool_calls=None,
                thinking_blocks=[
                    {"index": 0, "type": "thinking", "thinking": "search."}
                ],
                provider_specific_fields=None,
            )
        ),
        _chunk(
            delta=_delta(
                content=None,
                reasoning_content=None,
                tool_calls=None,
                thinking_blocks=[{"index": 0, "signature": "s1"}],
                provider_specific_fields=None,
            )
        ),
        _chunk(
            delta=_delta(
                content="answer",
                reasoning_content=None,
                tool_calls=None,
                thinking_blocks=None,
                provider_specific_fields=None,
            ),
            finish_reason="stop",
        ),
    ]
    msg, _, _ = accumulate_stream(iter(chunks))
    assert msg.thinking_blocks == [
        {"type": "thinking", "thinking": "I should search.", "signature": "s1"}
    ]
    assert msg.content == "answer"


def test_captures_provider_specific_fields_lists_concat():
    """OpenAI Responses encrypted_content_items accumulate as a list."""
    chunks = [
        _chunk(
            delta=_delta(
                content=None,
                reasoning_content=None,
                tool_calls=None,
                thinking_blocks=None,
                provider_specific_fields={"encrypted_content_items": [{"id": "a"}]},
            )
        ),
        _chunk(
            delta=_delta(
                content=None,
                reasoning_content=None,
                tool_calls=None,
                thinking_blocks=None,
                provider_specific_fields={"encrypted_content_items": [{"id": "b"}]},
            )
        ),
        _chunk(
            delta=_delta(
                content="x",
                reasoning_content=None,
                tool_calls=None,
                thinking_blocks=None,
                provider_specific_fields=None,
            ),
            finish_reason="stop",
        ),
    ]
    msg, _, _ = accumulate_stream(iter(chunks))
    assert msg.provider_specific_fields == {
        "encrypted_content_items": [{"id": "a"}, {"id": "b"}]
    }


def test_captures_provider_specific_fields_scalars_last_wins():
    """Non-list fields use last-write-wins."""
    chunks = [
        _chunk(
            delta=_delta(
                content=None,
                reasoning_content=None,
                tool_calls=None,
                thinking_blocks=None,
                provider_specific_fields={"some_count": 5},
            )
        ),
        _chunk(
            delta=_delta(
                content=None,
                reasoning_content=None,
                tool_calls=None,
                thinking_blocks=None,
                provider_specific_fields={"some_count": 7},
            )
        ),
        _chunk(
            delta=_delta(
                content="x",
                reasoning_content=None,
                tool_calls=None,
                thinking_blocks=None,
                provider_specific_fields=None,
            ),
            finish_reason="stop",
        ),
    ]
    msg, _, _ = accumulate_stream(iter(chunks))
    assert msg.provider_specific_fields["some_count"] == 7


def test_message_dataclass_has_extended_fields():
    """Smoke check: the Message dataclass exposes thinking_blocks and provider_specific_fields."""
    chunks = [
        _chunk(
            delta=_delta(
                content="hi",
                reasoning_content=None,
                tool_calls=None,
                thinking_blocks=None,
                provider_specific_fields=None,
            ),
            finish_reason="stop",
        ),
    ]
    msg, _, _ = accumulate_stream(iter(chunks))
    assert hasattr(msg, "thinking_blocks")
    assert hasattr(msg, "provider_specific_fields")
    assert msg.thinking_blocks == []
    assert msg.provider_specific_fields == {}


def test_combine_thinking_blocks_handles_no_index():
    """If LiteLLM emits thinking_blocks without explicit index, treat as index=0."""
    chunks = [
        _chunk(
            delta=_delta(
                content=None,
                reasoning_content=None,
                tool_calls=None,
                thinking_blocks=[{"type": "thinking", "thinking": "x"}],
                provider_specific_fields=None,
            )
        ),
        _chunk(
            delta=_delta(
                content=None,
                reasoning_content=None,
                tool_calls=None,
                thinking_blocks=[{"thinking": "y"}],
                provider_specific_fields=None,
            )
        ),
        _chunk(
            delta=_delta(
                content="ok",
                reasoning_content=None,
                tool_calls=None,
                thinking_blocks=None,
                provider_specific_fields=None,
            ),
            finish_reason="stop",
        ),
    ]
    msg, _, _ = accumulate_stream(iter(chunks))
    assert msg.thinking_blocks == [{"type": "thinking", "thinking": "xy"}]
