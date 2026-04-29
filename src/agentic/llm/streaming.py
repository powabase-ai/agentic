"""LiteLLM streaming response accumulator.

Provides accumulate_stream() which iterates a streaming response and assembles
a Message-like object compatible with what the ReAct loop reads off
response.choices[0].message. Callbacks fire per non-empty fragment for
content and reasoning_content; tool calls accumulate by index and JSON-decode
the assembled arguments at end-of-stream.

Live-only delta events should be emitted by the caller via callbacks; the
utility itself doesn't know about events. This keeps it provider-agnostic
and unit-testable without ExecutionContext or threading.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

# ===== Exceptions =====


class AbortedError(Exception):
    """Raised when the abort_signal fires mid-stream."""


class StreamPartialError(Exception):
    """Raised when an exception occurs during stream iteration.

    Carries `partial_content` and `partial_reasoning` from the buffers at the
    moment of failure, so the caller can persist what was streamed.
    """

    def __init__(
        self,
        message: str,
        partial_content: str = "",
        partial_reasoning: str = "",
    ):
        super().__init__(message)
        self.partial_content = partial_content
        self.partial_reasoning = partial_reasoning


class StreamTruncationError(StreamPartialError):
    """Raised when accumulated tool-call arguments fail to JSON-decode at stream end.

    Indicates an upstream truncation within a tool_call's argument fragment.
    """


# ===== Assembled message =====


@dataclass
class _Function:
    """Tool-call function shape — name + JSON-string arguments.

    CRITICAL: `arguments` is a JSON STRING, not a parsed dict. This matches
    LiteLLM's normal (non-streaming) response shape. agent.py accesses
    `tc.function.name` and `tc.function.arguments` (attribute access) and
    re-serializes `arguments` into msg_dict for the next LLM round-trip —
    a parsed dict here would silently break that round-trip.
    """

    name: str
    arguments: str


@dataclass
class _ToolCall:
    """Minimal tool_call shape matching what the ReAct loop reads.

    agent.py:475-486 and tool dispatch use attribute access:
        tc.function.name
        tc.function.arguments  # JSON string
    Hence _Function is a dataclass, NOT a dict.
    """

    id: str
    type: str = "function"
    function: _Function | None = None


@dataclass
class Message:
    """Minimal assistant message shape matching response.choices[0].message."""

    role: str = "assistant"
    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[_ToolCall] = field(default_factory=list)
    thinking_blocks: list[dict] = field(default_factory=list)
    provider_specific_fields: dict = field(default_factory=dict)


# ===== Accumulator =====


def _combine_thinking_blocks(blocks: list[dict]) -> list[dict]:
    """Combine streamed thinking-block deltas into final form, grouping by index.

    Anthropic emits content_block_start (type=thinking), content_block_delta
    (type=thinking_delta) chunks of partial thinking text, then signature_delta,
    then content_block_stop. LiteLLM normalizes to delta.thinking_blocks per
    chunk that share an `index` per logical block.

    Local implementation rather than coupling to LiteLLM internals — equivalent
    algorithm, fewer cross-version surprises (verified at litellm/main.py for
    processor.get_combined_thinking_content).
    """
    by_index: dict[int, dict] = {}
    for block in blocks:
        idx = block.get("index", 0)
        existing = by_index.setdefault(
            idx, {"type": block.get("type", "thinking"), "thinking": ""}
        )
        if "thinking" in block and block["thinking"]:
            existing["thinking"] += block["thinking"]
        if "signature" in block and block["signature"]:
            existing["signature"] = block["signature"]
        if block.get("type"):
            existing["type"] = block["type"]
    return [by_index[i] for i in sorted(by_index)]


def accumulate_stream(
    stream: Iterator[Any],
    on_content_delta: Callable[[str], None] | None = None,
    on_reasoning_delta: Callable[[str], None] | None = None,
    abort_signal: threading.Event | None = None,
) -> tuple[Message, str | None, dict | None]:
    """Iterate an LLM stream, invoking callbacks per non-empty fragment.

    Returns (assembled_message, finish_reason, usage_dict).
    - assembled_message: Message with .content, .reasoning_content, .tool_calls
      matching what the ReAct loop reads off `response.choices[0].message`.
    - finish_reason: from the last content-bearing chunk.
    - usage_dict: from the final usage chunk if present (None otherwise).

    Callbacks receive only the delta fragment. The caller embeds step number /
    source / delegation context via lambda closure.

    Raises AbortedError if abort_signal.is_set() between chunks (closes the
    underlying stream cleanly via stream.close() first).
    """
    content_buf = ""
    reasoning_buf = ""
    # Tool calls accumulate into a dict keyed by `tc.index`. Each entry has
    # {id, name, args_str}. JSON-decoded at end of stream; raises
    # StreamTruncationError on malformed args.
    tool_calls_acc: dict[int, dict[str, Any]] = {}
    thinking_blocks_acc: list[dict] = []
    psf_acc: dict[str, Any] = {}
    finish_reason: str | None = None
    usage: dict | None = None

    try:
        for chunk in stream:
            # Abort poll point
            if abort_signal is not None and abort_signal.is_set():
                try:
                    if hasattr(stream, "close"):
                        stream.close()
                finally:
                    raise AbortedError("Stream aborted by signal")

            # Capture usage if present (typically on the final chunk)
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = _extract_usage(chunk_usage)

            # Process choices
            choices = getattr(chunk, "choices", []) or []
            if not choices:
                continue

            choice = choices[0]
            delta = getattr(choice, "delta", None)
            chunk_finish_reason = getattr(choice, "finish_reason", None)
            if chunk_finish_reason is not None:
                finish_reason = chunk_finish_reason

            if delta is None:
                continue

            # Content fragment
            content_frag = getattr(delta, "content", None)
            if content_frag:
                content_buf += content_frag
                if on_content_delta is not None:
                    on_content_delta(content_frag)

            # Reasoning fragment
            reasoning_frag = getattr(delta, "reasoning_content", None)
            if reasoning_frag:
                reasoning_buf += reasoning_frag
                if on_reasoning_delta is not None:
                    on_reasoning_delta(reasoning_frag)

            # Tool calls
            tool_call_deltas = getattr(delta, "tool_calls", None) or []
            for tcd in tool_call_deltas:
                idx = getattr(tcd, "index", 0)
                entry = tool_calls_acc.setdefault(
                    idx, {"id": None, "name": None, "args_str": ""}
                )
                tcd_id = getattr(tcd, "id", None)
                if tcd_id and entry["id"] is None:
                    entry["id"] = tcd_id
                tcd_function = getattr(tcd, "function", None)
                if tcd_function is not None:
                    tcd_name = getattr(tcd_function, "name", None)
                    if tcd_name and entry["name"] is None:
                        entry["name"] = tcd_name
                    tcd_args = getattr(tcd_function, "arguments", None)
                    if tcd_args:
                        entry["args_str"] += tcd_args

            # Anthropic thinking_blocks delta capture (verified at
            # litellm/main.py:6350-6361). Each chunk's delta.thinking_blocks
            # is a list of partial blocks identified by index.
            chunk_thinking = getattr(delta, "thinking_blocks", None) or []
            for block in chunk_thinking:
                thinking_blocks_acc.append(block)

            # OpenAI Responses / Gemini provider_specific_fields delta capture.
            # List-valued fields concat; scalar fields last-write-wins.
            chunk_psf = getattr(delta, "provider_specific_fields", None) or {}
            for k, v in chunk_psf.items():
                if isinstance(v, list):
                    psf_acc.setdefault(k, []).extend(v)
                else:
                    psf_acc[k] = v
    except (AbortedError, StreamPartialError):
        # Already wrapped — propagate as-is
        raise
    except Exception as e:
        raise StreamPartialError(
            f"Stream iteration failed: {e}",
            partial_content=content_buf,
            partial_reasoning=reasoning_buf,
        ) from e

    # Finalize tool calls — VALIDATE the JSON parses (raise on truncation)
    # but keep `arguments` as a JSON STRING in the result (B1 v3: agent.py
    # expects a string here matching LiteLLM's normal response shape).
    tool_calls: list[_ToolCall] = []
    for idx in sorted(tool_calls_acc):
        entry = tool_calls_acc[idx]
        args_str = entry["args_str"] or "{}"
        # Validate JSON (raises on mid-arg truncation) but DON'T replace the value
        try:
            json.loads(args_str)
        except json.JSONDecodeError as e:
            raise StreamTruncationError(
                f"Tool call args at index {idx} failed to decode: {e}",
                partial_content=content_buf,
                partial_reasoning=reasoning_buf,
            ) from e
        tool_calls.append(
            _ToolCall(
                id=entry["id"] or f"call_{idx}",
                function=_Function(name=entry["name"] or "", arguments=args_str),
            )
        )

    return (
        Message(
            content=content_buf,
            reasoning_content=reasoning_buf,
            tool_calls=tool_calls,
            thinking_blocks=_combine_thinking_blocks(thinking_blocks_acc),
            provider_specific_fields=psf_acc,
        ),
        finish_reason,
        usage,
    )


def _extract_usage(chunk_usage: Any) -> dict:
    """Normalize the usage chunk's shape to a plain dict."""
    return {
        "prompt_tokens": getattr(chunk_usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(chunk_usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(chunk_usage, "total_tokens", 0) or 0,
    }
