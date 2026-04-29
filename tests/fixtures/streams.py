"""Test fixtures for LiteLLM-style streaming chunk shapes.

These dataclasses mirror what `litellm.completion(stream=True)` yields, so
accumulator tests can construct realistic streams without mocking nested
attribute graphs.

Use the helper functions for common patterns (content-only, reasoning-only,
tool calls). Compose with `fake_stream(*chunks)` to feed `accumulate_stream`.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeFunction:
    name: str | None = None
    arguments: str | None = None


@dataclass
class FakeToolCallDelta:
    index: int = 0
    id: str | None = None
    type: str = "function"
    function: FakeFunction | None = None


@dataclass
class FakeDelta:
    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[FakeToolCallDelta] | None = None


@dataclass
class FakeChoice:
    delta: FakeDelta
    finish_reason: str | None = None
    index: int = 0


@dataclass
class FakeChunk:
    choices: list[FakeChoice] = field(default_factory=list)
    usage: Any | None = None


def fake_stream(*chunks: FakeChunk) -> Iterator[FakeChunk]:
    """Yield the given chunks. Use as the input to `accumulate_stream` in tests."""
    yield from chunks


def role_chunk() -> FakeChunk:
    """First chunk in a stream — establishes the assistant role."""
    return FakeChunk(choices=[FakeChoice(delta=FakeDelta(role="assistant"))])


def content_chunks(text: str, fragments: int = 4) -> list[FakeChunk]:
    """Split `text` into `fragments` content-bearing chunks."""
    if fragments < 1 or not text:
        return []
    step = max(1, len(text) // fragments)
    pieces = [text[i : i + step] for i in range(0, len(text), step)]
    return [FakeChunk(choices=[FakeChoice(delta=FakeDelta(content=p))]) for p in pieces]


def reasoning_chunks(text: str, fragments: int = 4) -> list[FakeChunk]:
    """Split `text` into `fragments` reasoning_content-bearing chunks."""
    if fragments < 1 or not text:
        return []
    step = max(1, len(text) // fragments)
    pieces = [text[i : i + step] for i in range(0, len(text), step)]
    return [
        FakeChunk(choices=[FakeChoice(delta=FakeDelta(reasoning_content=p))])
        for p in pieces
    ]


def tool_call_chunks(
    name: str, args: dict[str, Any], index: int = 0, frag_count: int = 3
) -> list[FakeChunk]:
    """Build chunks for a single tool call: id+name first, then arg fragments.

    args is serialized to JSON and split into `frag_count` pieces.
    """
    import json

    args_json = json.dumps(args)
    if frag_count < 1:
        frag_count = 1
    step = max(1, len(args_json) // frag_count)
    arg_pieces = [args_json[i : i + step] for i in range(0, len(args_json), step)]

    chunks: list[FakeChunk] = []
    # First chunk: id + name, no args
    chunks.append(
        FakeChunk(
            choices=[
                FakeChoice(
                    delta=FakeDelta(
                        tool_calls=[
                            FakeToolCallDelta(
                                index=index,
                                id=f"call_{index}",
                                function=FakeFunction(name=name, arguments=""),
                            )
                        ]
                    )
                )
            ]
        )
    )
    # Subsequent chunks: arg fragments only
    for piece in arg_pieces:
        chunks.append(
            FakeChunk(
                choices=[
                    FakeChoice(
                        delta=FakeDelta(
                            tool_calls=[
                                FakeToolCallDelta(
                                    index=index, function=FakeFunction(arguments=piece)
                                )
                            ]
                        )
                    )
                ]
            )
        )
    return chunks


def finish_chunk(reason: str = "stop") -> FakeChunk:
    """Last content-bearing chunk — has finish_reason set."""
    return FakeChunk(choices=[FakeChoice(delta=FakeDelta(), finish_reason=reason)])


def usage_chunk(prompt_tokens: int, completion_tokens: int) -> FakeChunk:
    """Final usage chunk (OpenAI with stream_options={'include_usage': True}).

    Note: this chunk has empty `choices`, only `usage`.
    """

    @dataclass
    class _Usage:
        prompt_tokens: int
        completion_tokens: int
        total_tokens: int

    return FakeChunk(
        choices=[],
        usage=_Usage(
            prompt_tokens, completion_tokens, prompt_tokens + completion_tokens
        ),
    )
