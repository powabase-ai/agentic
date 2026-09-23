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
    accumulator combines them in sequence."""
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
    """List-valued provider_specific_fields concatenate across chunks."""
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
    """Fragments without an index join the open block in sequence."""
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


def _bare_delta(**kwargs):
    fields = dict(
        content=None,
        reasoning_content=None,
        tool_calls=None,
        thinking_blocks=None,
        provider_specific_fields=None,
    )
    fields.update(kwargs)
    return _delta(**fields)


def test_captures_reasoning_items_from_delta():
    """LiteLLM's Responses bridge emits every reasoning item of a response on
    one delta; stream_chunk_builder drops them, so the accumulator must."""
    item_a = {
        "id": "rs_a",
        "type": "reasoning",
        "encrypted_content": "A",
        "summary": [],
    }
    item_b = {
        "id": "rs_b",
        "type": "reasoning",
        "encrypted_content": "B",
        "summary": [],
    }
    chunks = [
        _chunk(delta=_bare_delta(content="hi")),
        _chunk(delta=_bare_delta(content="", reasoning_items=[item_a, item_b])),
        _chunk(delta=_bare_delta(), finish_reason="stop"),
    ]
    msg, _, _ = accumulate_stream(iter(chunks))
    assert msg.reasoning_items == [item_a, item_b]
    assert msg.content == "hi"


def test_reasoning_items_default_to_empty():
    msg, _, _ = accumulate_stream(iter([]))
    assert msg.reasoning_items == []


def test_redacted_thinking_block_keeps_its_data():
    """A redacted block is opaque: its whole payload is `data`. Rebuilding it
    as {type, thinking, signature} would make it malformed on replay. The
    deltas carry LiteLLM's real shapes: no `index`, the signature on a delta of
    its own with empty thinking text."""
    chunks = [
        _chunk(
            delta=_bare_delta(
                thinking_blocks=[
                    {"type": "thinking", "thinking": "plan", "signature": ""}
                ]
            )
        ),
        _chunk(
            delta=_bare_delta(
                thinking_blocks=[
                    {"type": "thinking", "thinking": "", "signature": "s0"}
                ]
            )
        ),
        _chunk(
            delta=_bare_delta(
                thinking_blocks=[{"type": "redacted_thinking", "data": "OPAQUE"}]
            )
        ),
        _chunk(delta=_bare_delta(content="answer"), finish_reason="stop"),
    ]
    msg, _, _ = accumulate_stream(iter(chunks))
    assert msg.thinking_blocks == [
        {"type": "thinking", "thinking": "plan", "signature": "s0"},
        {"type": "redacted_thinking", "data": "OPAQUE"},
    ]


def _thinking_chunks(*deltas):
    return [_chunk(delta=_bare_delta(thinking_blocks=[d])) for d in deltas] + [
        _chunk(delta=_bare_delta(content="answer"), finish_reason="stop")
    ]


def test_two_thinking_blocks_stay_separate():
    """LiteLLM's thinking deltas carry no index, so blocks are told apart by
    sequence: a signature closes the block it signs."""
    chunks = _thinking_chunks(
        {"type": "thinking", "thinking": "first", "signature": ""},
        {"type": "thinking", "thinking": "", "signature": "SA"},
        {"type": "thinking", "thinking": "second", "signature": ""},
        {"type": "thinking", "thinking": "", "signature": "SB"},
    )
    msg, _, _ = accumulate_stream(iter(chunks))
    assert msg.thinking_blocks == [
        {"type": "thinking", "thinking": "first", "signature": "SA"},
        {"type": "thinking", "thinking": "second", "signature": "SB"},
    ]


def test_redacted_before_thinking():
    chunks = _thinking_chunks(
        {"type": "redacted_thinking", "data": "OPAQUE"},
        {"type": "thinking", "thinking": "plan", "signature": ""},
        {"type": "thinking", "thinking": "", "signature": "s1"},
    )
    msg, _, _ = accumulate_stream(iter(chunks))
    assert msg.thinking_blocks == [
        {"type": "redacted_thinking", "data": "OPAQUE"},
        {"type": "thinking", "thinking": "plan", "signature": "s1"},
    ]


def test_blocks_parsed_by_litellms_anthropic_stream_parser_stay_separate():
    """Drive LiteLLM's own Anthropic SSE parser, so the test pins the delta
    shapes a real stream produces rather than our idea of them."""
    from litellm.llms.anthropic.chat.handler import ModelResponseIterator

    def start(i, block):
        return {"type": "content_block_start", "index": i, "content_block": block}

    def delta(i, d):
        return {"type": "content_block_delta", "index": i, "delta": d}

    def stop(i):
        return {"type": "content_block_stop", "index": i}

    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude",
                "usage": {"input_tokens": 5, "output_tokens": 1},
            },
        },
        start(0, {"type": "thinking", "thinking": ""}),
        delta(0, {"type": "thinking_delta", "thinking": "first"}),
        delta(0, {"type": "signature_delta", "signature": "SIG0"}),
        stop(0),
        start(1, {"type": "redacted_thinking", "data": "OPAQUE"}),
        stop(1),
        start(2, {"type": "thinking", "thinking": ""}),
        delta(2, {"type": "thinking_delta", "thinking": "second"}),
        delta(2, {"type": "signature_delta", "signature": "SIG2"}),
        stop(2),
        start(3, {"type": "text", "text": ""}),
        delta(3, {"type": "text_delta", "text": "answer"}),
        stop(3),
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 9},
        },
        {"type": "message_stop"},
    ]
    parser = ModelResponseIterator(
        streaming_response=None, sync_stream=True, json_mode=False
    )
    chunks = [c for c in (parser.chunk_parser(e) for e in events) if c is not None]

    msg, _, _ = accumulate_stream(iter(chunks))

    assert msg.thinking_blocks == [
        {"type": "thinking", "thinking": "first", "signature": "SIG0"},
        {"type": "redacted_thinking", "data": "OPAQUE"},
        {"type": "thinking", "thinking": "second", "signature": "SIG2"},
    ]
    assert msg.content == "answer"


def test_stream_cut_off_mid_thinking_replays_no_unsigned_block():
    """A response truncated mid-thinking leaves its last block unsigned. The
    record keeps it; replay must not send it back."""
    from agentic.agent.message import reasoning_replay_fields
    from agentic.llm.reasoning_extractor import extract_reasoning_artifact

    def thinking_delta(text="", signature=""):
        return _chunk(
            delta=_delta(
                content=None,
                reasoning_content=text or None,
                tool_calls=None,
                thinking_blocks=[
                    {"type": "thinking", "thinking": text, "signature": signature}
                ],
                provider_specific_fields=None,
            )
        )

    chunks = [
        thinking_delta("first "),
        thinking_delta("block"),
        thinking_delta(signature="sig1"),
        thinking_delta("second, cut "),
        thinking_delta("off"),
        _chunk(
            delta=_delta(
                content=None,
                reasoning_content=None,
                tool_calls=None,
                thinking_blocks=None,
                provider_specific_fields=None,
            ),
            finish_reason="length",
        ),
    ]
    msg, finish_reason, _ = accumulate_stream(iter(chunks))
    assert finish_reason == "length"
    artifact = extract_reasoning_artifact(
        model="anthropic/claude-opus-4-8",
        assembled_message=msg,
        final_response=SimpleNamespace(usage=None),
        requested_effort="high",
    )
    assert artifact.thinking_blocks == [
        {"type": "thinking", "thinking": "first block", "signature": "sig1"},
        {"type": "thinking", "thinking": "second, cut off"},
    ]
    assert reasoning_replay_fields(artifact) == {
        "thinking_blocks": [
            {"type": "thinking", "thinking": "first block", "signature": "sig1"}
        ]
    }
