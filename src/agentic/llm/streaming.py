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
import logging
import re
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _extract_reasoning_from_delta(delta) -> str | None:
    """Robust extraction tolerating field-name variations and known leakage bugs.

    Defends against:
    - Field-name variation (LiteLLM #21386): some providers use `reasoning`
      instead of `reasoning_content`.
    - <think> tags leaked into content (LiteLLM #26326, Fireworks AI bug).
    - THOUGHT: prefix in content (google-genai #2121, Gemini 2.5 bug).

    Returns None when no reasoning is detected.
    """
    rc = getattr(delta, "reasoning_content", None)
    if rc:
        return rc
    r = getattr(delta, "reasoning", None)
    if r:
        return r
    content = getattr(delta, "content", "") or ""
    if "<think>" in content:
        match = _THINK_TAG_RE.search(content)
        if match:
            return match.group(1)
    if content.startswith("THOUGHT:"):
        return content[len("THOUGHT:") :].strip()
    return None


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
    *,
    model: str | None = None,
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
            reasoning_frag = _extract_reasoning_from_delta(delta)
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

    # Workaround for LiteLLM's Anthropic streaming usage gap: in
    # non-streaming responses, `AnthropicConfig.calculate_usage` counts
    # reasoning_tokens from the reasoning_content text. In streaming mode,
    # `ModelResponseIterator._handle_usage` calls `calculate_usage` with
    # `reasoning_content=None`, so reasoning_tokens always lands as 0 on
    # completion_tokens_details — even when the model actually emitted
    # thinking blocks. Same gap shows up for any provider whose streaming
    # path forwards the raw usage without re-counting. Best-effort fallback:
    # if we accumulated reasoning text but usage says 0, count the tokens
    # ourselves using the same tokenizer the provider would use.
    if (
        model is not None
        and reasoning_buf
        and usage is not None
        and not usage.get("reasoning_tokens")
    ):
        try:
            import litellm

            counted = litellm.token_counter(model=model, text=reasoning_buf)
            if counted > 0:
                usage["reasoning_tokens"] = counted
        except Exception:
            # Best-effort — never fail the stream over a token count. Log at
            # debug so a missing tokenizer mapping is grep-able when reasoning
            # token counts stay zero in production for a particular model.
            logger.debug(
                "reasoning_tokens backfill failed for model=%s",
                model,
                exc_info=True,
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
    """Normalize the usage chunk's shape to a plain dict.

    Mirrors Agent._extract_usage in agent.py: captures `reasoning_tokens` and
    `cached_tokens` from their nested details objects so the streaming path
    surfaces them for reasoning-capable models. Without this, streaming runs
    persisted `reasoning_tokens=0` even when the model clearly used reasoning.
    Naming variation across APIs:
      - Chat Completions:  completion_tokens_details.reasoning_tokens,
                           prompt_tokens_details.cached_tokens
      - Responses API:     output_tokens_details.reasoning_tokens,
                           input_tokens_details.cached_tokens
    """

    def _get(obj: Any, key: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    prompt = _get(chunk_usage, "prompt_tokens")
    if prompt is None:
        prompt = _get(chunk_usage, "input_tokens")
    completion = _get(chunk_usage, "completion_tokens")
    if completion is None:
        completion = _get(chunk_usage, "output_tokens")
    total = _get(chunk_usage, "total_tokens")

    out: dict[str, int] = {
        "prompt_tokens": prompt or 0,
        "completion_tokens": completion or 0,
        "total_tokens": total or 0,
    }

    for details_key in ("completion_tokens_details", "output_tokens_details"):
        details = _get(chunk_usage, details_key)
        reasoning = _get(details, "reasoning_tokens")
        if reasoning is not None:
            out["reasoning_tokens"] = reasoning
            break

    for details_key in ("prompt_tokens_details", "input_tokens_details"):
        details = _get(chunk_usage, details_key)
        cached = _get(details, "cached_tokens")
        if cached is not None:
            out["cached_tokens"] = cached
            break

    return out
