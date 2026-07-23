"""Context compaction — summarize conversation history when context is full."""

from __future__ import annotations

import logging
import re
from typing import Any

import litellm

from agentic.agent.model_registry import resolve_context_window
from agentic.agent.normalization import normalize_messages

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4

_SUMMARY_MAX_TOKENS = 8000

# Compaction is a normal provider call on the same model as the real one, so it
# gets the same retry budget (see `Agent._run` — `num_retries: 3`). Without it a
# transient 529 is a hard compaction failure on the exact path that exists to
# rescue an over-long context.
_COMPACTION_NUM_RETRIES = 3
# litellm's `request_timeout` default is 6000.0s, so an unbounded compaction
# call can block the agent for 100 minutes. Generous enough for a full-context
# summarization (the prompt is large, but the output is capped at 8k).
_COMPACTION_TIMEOUT_SECONDS = 600.0

# Whitespace inside the angle brackets is tolerated throughout: models emit
# `<summary >` and `< analysis >` often enough that a strict pattern turns an
# ordinary formatting wobble into a compaction failure (`<summary >` used to
# yield "" — no progress, forever) or a scratchpad leak (`< analysis >` used to
# fall through to the whole-text fallback).
# All four tag matchers accept an optional attribute list (`\b[^>]*`): a model
# that emits `<analysis confidence="low">` or `<summary lang="de">` is ordinary
# output, and a bare-tag regex silently fails to match it — for the analysis
# tag that means the scratchpad is never stripped and leaks into the summary,
# the exact class of failure this guard exists to prevent. `\b` keeps it from
# matching `<analysistext>`.
_SUMMARY_RE = re.compile(
    r"<\s*summary\b[^>]*>(.*?)<\s*/\s*summary\s*>", re.DOTALL | re.IGNORECASE
)
_SUMMARY_OPEN_RE = re.compile(r"<\s*summary\b[^>]*>", re.IGNORECASE)
_ANALYSIS_OPEN_RE = re.compile(r"<\s*analysis\b[^>]*>", re.IGNORECASE)
_ANALYSIS_BLOCK_RE = re.compile(
    r"<\s*analysis\b[^>]*>.*?<\s*/\s*analysis\s*>", re.DOTALL | re.IGNORECASE
)

# finish_reason values that mean the model was cut off mid-output.
_TRUNCATED_FINISH_REASONS = {"length", "max_tokens"}

COMPACTION_INSTRUCTION = """Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions. This summary will replace the full conversation history, so it must capture every fact, decision, and piece of context needed to continue the work without loss — whatever the domain (research, data analysis, writing, legal analysis, software, operations, etc.).

Before your final summary, wrap your analysis in <analysis> tags. In your analysis:
1. Chronologically analyze each message and section. For each, identify:
   - The user's explicit requests and intents
   - Your approach to addressing them
   - Key decisions, concepts, methods, and domain-specific details
   - Specific identifiers and content: file names, document/record IDs, URLs, source titles, citations, data values, function signatures, exact quotes, edits made
2. Double-check for accuracy and completeness.

Your summary must include these sections:
1. Primary Request and Intent: all of the user's explicit requests and intents, in detail.
2. Key Concepts: the important concepts, methods, technologies, and domain knowledge involved.
3. Materials, Sources, and Findings: specific files, documents, retrieved sources, tool results, citations, and data examined or produced — with their identifiers and the exact content that matters (quotes, values, snippets), plus why each is important.
4. Problem Solving: problems solved and any ongoing troubleshooting.
5. Pending Tasks: tasks you have explicitly been asked to work on.
6. Current Work: precisely what was being worked on immediately before this summary, with exact identifiers and content from the most recent messages.
7. Optional Next Step: the next step, ONLY if directly in line with the user's explicit request and the work in progress. If the last task concluded, list a next step only if explicitly requested; do not drift into tangential work.
8. If there is a next step, include direct verbatim quotes from the most recent messages showing exactly what you were working on and where you left off, to prevent drift.

Output your analysis in <analysis> tags, then the summary in <summary> tags."""

CONTINUATION_NUDGE = (
    "You are mid-task — the above is a summary of work already in progress, not a "
    "new request. Do not acknowledge, greet, or recap the summary. Continue the "
    "task exactly where you left off, taking the next action directly."
)


def _strip_analysis(text: str) -> str:
    """Return ``text`` with the ``<analysis>`` scratchpad removed.

    The COMPACTION_INSTRUCTION above *tells* the model to wrap its output in
    ``<summary>`` tags, so a model restating that plan mid-analysis ("2. I will
    now wrap the result in <summary> tags as instructed") is ordinary output,
    not an exotic shape. Searching the raw text for the literal tag treats that
    mention as the start of the summary and drags the analysis tail — which for
    these agents carries confidence hedges the user must never see — into the
    conversation history.

    Closed ``<analysis>...</analysis>`` spans are removed outright. An *unclosed*
    opening tag means the model was cut off mid-scratchpad, so everything from
    that tag onward is scratchpad too and is dropped; without this, the
    truncated variant of the same shape defeats the guard entirely and installs
    the private scratchpad as the whole conversation history.
    """
    stripped = _ANALYSIS_BLOCK_RE.sub("", text)
    unclosed = _ANALYSIS_OPEN_RE.search(stripped)
    if unclosed:
        stripped = stripped[: unclosed.start()]
    return stripped


def _summary_started(text: str) -> bool:
    """Whether a real ``<summary>`` block opened outside the analysis scratchpad.

    Shared with ``_extract_summary`` so the finish_reason guard and the
    extraction path cannot disagree about what counts as a summary.
    """
    return bool(_SUMMARY_OPEN_RE.search(_strip_analysis(text)))


def _extract_summary(text: str | None) -> str:
    """Return the <summary>...</summary> content, or the whole text if untagged.

    If a closing </summary> tag is missing (e.g. output truncated by
    max_tokens) but an opening <summary> tag is present, return everything
    after the opening tag instead of falling back to the whole text — which
    would otherwise leak the <analysis> reasoning block into the summary.

    If there is no <summary> tag at all *and* an <analysis> block appears
    anywhere in the output, the model ran out of
    output budget before it ever started the summary. Returning the raw text
    there would install the model's internal scratchpad as the entire
    conversation history, so return "" — callers treat an empty summary as a
    compaction failure and keep the original messages. Genuinely untagged
    plain-text output still falls back to the whole text (intentional).

    The <analysis> check uses ``search``, not an anchored ``match``: models
    routinely emit a conversational preamble ("Okay, let me analyze...")
    before the tag, and an anchored check lets that shape through with the
    scratchpad attached.

    The summary is searched for only *outside* the analysis scratchpad — see
    ``_strip_analysis``. The final ``<analysis>`` check still runs against the
    raw text, so an analysis-only output is still reported as a failure.
    """
    if not text:
        return ""
    outside_analysis = _strip_analysis(text)
    m = _SUMMARY_RE.search(outside_analysis)
    if m:
        return m.group(1).strip()
    open_m = _SUMMARY_OPEN_RE.search(outside_analysis)
    if open_m:
        return outside_analysis[open_m.end() :].strip()
    if _ANALYSIS_OPEN_RE.search(text):
        return ""
    return text.strip()


# Floor for an image block whose payload is a short remote URL: the URL costs
# almost no characters, but the image it resolves to still costs the provider
# real tokens.
_IMAGE_BLOCK_MIN_CHARS = 1000


def _block_char_len(item: Any) -> int:
    """Measured size of one multimodal content block.

    Every branch measures the block's *actual* payload. A flat proxy weight
    (the previous ``+= 1000`` for images) under-measured an inline base64 data
    URL by ~2000x — a 6MB image estimated at 752 tokens against a real
    ~1,500,000 — so ``truncate_messages``' "did we get under target"
    postcondition passed on a megabyte history and the whole compaction
    subsystem was inoperative for image-mode knowledge bases. This is
    production-reachable: the host's context handler resolves image storage
    refs to inline base64 data URLs, which reach us as message content arrays.

    Unrecognized block types are measured rather than counted as 0 for the same
    reason: a provider-specific block (``tool_result``, ``document``, ...)
    carrying megabytes of content must not be invisible to the budget.
    """
    if not isinstance(item, dict):
        return len(str(item))
    kind = item.get("type")
    if kind == "text":
        return len(item.get("text", ""))
    if kind == "image_url":
        url = item.get("image_url")
        if isinstance(url, dict):
            url = url.get("url", "")
        return max(len(str(url or "")), _IMAGE_BLOCK_MIN_CHARS)
    return len(str(item))


def _content_char_len(content: Any) -> int:
    """Return approximate character length of a message content field.

    Handles both ``str`` content and multimodal ``list[dict]`` content blocks
    (e.g. ``[{"type": "text", "text": "..."}, {"type": "image_url", ...}]``).
    """
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(_block_char_len(item) for item in content)
    return len(str(content)) if content else 0


def _tool_calls_char_len(msg: dict[str, Any]) -> int:
    """Approximate character length of a message's ``tool_calls`` payload.

    Assistant tool-call messages routinely carry ``content=None`` and put the
    entire payload in ``tool_calls[].function.arguments``. Counting only
    ``content`` measured such a message at 0 tokens no matter how large it
    was, which made ``truncate_messages``' "did we get under target"
    postcondition unenforceable: it would report success having shrunk
    nothing, and the identical oversized request would be re-sent.
    """
    calls = msg.get("tool_calls")
    if not isinstance(calls, list):
        return 0
    total = 0
    for tc in calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        if not isinstance(fn, dict):
            continue
        total += len(str(fn.get("name") or ""))
        total += len(str(fn.get("arguments") or ""))
    return total


def _message_char_len(msg: dict[str, Any]) -> int:
    """Total measured size of a message: content plus any tool-call payload."""
    return _content_char_len(msg.get("content", "") or "") + _tool_calls_char_len(msg)


def estimate_token_count(messages: list[dict[str, Any]]) -> int:
    """Rough estimate of token count for a message list."""
    total_chars = sum(_message_char_len(m) for m in messages)
    return total_chars // CHARS_PER_TOKEN


def compact_messages(
    messages: list[dict[str, Any]],
    model: str,
    api_key: str | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Summarize the conversation in-context on ``model`` and rebuild a short history.

    Appends a single summary instruction to the existing message prefix (so the
    cached history prefix is reused), extracts the ``<summary>`` block, and
    returns ``[system?, {user: summary}, {user: CONTINUATION_NUDGE}]``.

    ``model`` must be the same model string the real agent call routes to (see
    ``Agent._compaction_model_for``) and ``tools`` the same tool schemas it
    sends. The caching this relies on is the *automatic* prefix caching offered
    by the OpenAI-family, DeepSeek and OpenRouter endpoints, which matches on
    an exact leading prefix of the serialized request — tool definitions come
    first, so omitting or altering them breaks the match at the very first
    block and makes this (full-context) call entirely cache-cold, defeating the
    whole reason we feed compaction the un-pruned history. Anthropic caching is
    NOT in play: it requires explicit ``cache_control`` breakpoints, which this
    codebase does not set anywhere. ``tool_choice="none"`` keeps the model
    summarizing instead of trying to call one of them.

    Messages are run through ``normalize_messages`` first: not every call site
    hands us already-normalized history, and non-standard bookkeeping keys
    (e.g. ``_injected``) or orphan ``tool`` messages make OpenAI-compatible
    endpoints reject the request with a 400 — which the broad ``except`` below
    would swallow, silently turning compaction into a no-op.

    Reasoning kwargs are NOT forwarded: the agent's ``extra_body`` /
    ``reasoning_effort`` settings do not reach this call, so compaction always
    runs at the provider's default reasoning effort regardless of how the agent
    is configured. That is current behavior, documented rather than changed.

    On any failure the original ``messages`` list object is returned unchanged;
    callers detect "no progress" with an identity check (``result is messages``).
    """
    sanitized = normalize_messages(messages)

    if len(sanitized) <= 2:
        logger.warning(
            "Compaction skipped: too few messages to summarize "
            "(%d after normalization), keeping original messages",
            len(sanitized),
        )
        return messages

    if sanitized[-1].get("tool_calls"):
        logger.warning("Compaction skipped: last message has unanswered tool_calls")
        return messages

    system_msg = sanitized[0] if sanitized[0].get("role") == "system" else None
    summarize_request = sanitized + [
        {"role": "user", "content": COMPACTION_INSTRUCTION}
    ]
    call_kwargs: dict[str, Any] = {
        "model": model,
        "messages": summarize_request,
        "stream": False,
        "max_tokens": _SUMMARY_MAX_TOKENS,
        "num_retries": _COMPACTION_NUM_RETRIES,
        "timeout": _COMPACTION_TIMEOUT_SECONDS,
    }
    if api_key is not None:
        call_kwargs["api_key"] = api_key
    if tools:
        call_kwargs["tools"] = tools
        call_kwargs["tool_choice"] = "none"

    try:
        response = litellm.completion(**call_kwargs)
        choice = response.choices[0]
        content = choice.message.content
        finish_reason = getattr(choice, "finish_reason", None)
    except Exception as e:  # best-effort; keep original on failure
        logger.warning("Compaction failed, keeping original messages: %s", e)
        return messages

    # Cut off before the summary ever opened: whatever came back is the
    # <analysis> scratchpad, not a summary. Installing it as the conversation
    # history would be worse than not compacting at all.
    if finish_reason in _TRUNCATED_FINISH_REASONS and not _summary_started(
        content or ""
    ):
        logger.warning(
            "Compaction output truncated (finish_reason=%s) before <summary> "
            "opened, keeping original messages",
            finish_reason,
        )
        return messages

    summary = _extract_summary(content)

    if not summary.strip():
        logger.warning(
            "Compaction produced an empty summary, keeping original messages"
        )
        return messages

    result: list[dict[str, Any]] = []
    if system_msg:
        result.append(system_msg)
    result.append({"role": "user", "content": summary})
    result.append({"role": "user", "content": CONTINUATION_NUDGE})
    return result


_MAX_OUTPUT_TOKENS = 8000
_COMPACT_BUFFER = 13000
_COMPACT_BUFFER_FRACTION = 0.08
_COMPACT_BUFFER_MAX_FRACTION = 0.25
# Minimum share of the window kept available for input, as a fraction. The
# output reservation is capped so this much always survives: reserve plus
# buffer can exceed a small window outright (an 8k model against an 8k output
# reservation), and a non-positive threshold makes `token_estimate > threshold`
# unconditionally true — compaction then fires before the first request and
# replaces the user's question with a summary of nothing.
_MIN_INPUT_FRACTION = 0.25

_warned: set[str] = set()


def _warn_once(key: str, msg: str, *args: object) -> None:
    """Emit ``msg`` at WARNING the first time ``key`` is seen.

    ``get_context_threshold`` runs 2-3x per agent step, so an undeduped line
    here is log spam rather than a signal. Mirrors ``model_registry``'s dedupe.
    """
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(msg, *args)


def _compact_buffer(window: int) -> int:
    """Safety margin below the window, proportional to it and bounded by it.

    ``estimate_token_count`` is a crude ``chars // 4`` heuristic that can
    undercount by several percent on non-English or dense technical text. A
    flat 13k buffer was a ~10% margin at 128k windows but only ~1.3% at 1M,
    so scale it with the window and keep 13k as the floor (which preserves
    the historical value for windows up to 162,500).

    The 13k floor is also capped at 25% of the window: on small models
    (gpt-4's 8192, gpt-3.5-turbo's 16385) a flat 13k buffer is most or all of
    the context. Large windows are unaffected — 8% is below 25% everywhere.
    """
    return min(
        max(_COMPACT_BUFFER, int(window * _COMPACT_BUFFER_FRACTION)),
        int(window * _COMPACT_BUFFER_MAX_FRACTION),
    )


def get_context_threshold(model: str, max_output_tokens: int | None = None) -> int:
    """Compaction threshold: the model's window minus room for output + a buffer.

    ``max_output_tokens`` is the agent's configured ``max_tokens`` — the value
    actually sent to the provider. Reserving a hardcoded ``_MAX_OUTPUT_TOKENS``
    while the caller asks for more output puts the threshold *above* the real
    input ceiling, so compaction never fires and the call 400s with the exact
    ``prompt_too_long`` this function exists to prevent. Reserve whichever is
    larger.

    The reservation is taken from ``effective_max_output_tokens``, which caps
    it to what the window can actually serve, so
    ``threshold + reserve + buffer <= window`` holds unconditionally and the
    threshold is always positive. There is deliberately no floor on the
    *result*: an unconditional floor fires whenever the computed threshold is
    merely small, and a large ``max_tokens`` makes it legitimately small. A 200k
    window reserving 128k of output computes a correct 56k threshold; flooring
    that to 100k shipped a threshold whose own output request overran the
    window by 28k. Since ``estimate_token_count`` is a ``chars // 4``
    undercount, a zero-or-negative margin means compaction declines to fire and
    the provider returns the ``prompt_too_long`` this function exists to
    prevent.
    """
    window = resolve_context_window(model)
    reserve = effective_max_output_tokens(model, max_output_tokens)
    return window - reserve - _compact_buffer(window)


def effective_max_output_tokens(
    model: str, max_output_tokens: int | None = None
) -> int:
    """The output reservation ``get_context_threshold`` subtracts from the window.

    Reserving more output than the window can give back leaves no room for
    input, so this caps the reservation to keep ``_MIN_INPUT_FRACTION`` of the
    window (plus the compaction buffer) available. Windows large enough to
    serve the request are untouched — a 200k window reserves a 128k request in
    full.

    This bounds only the *reservation*, i.e. the point at which compaction
    fires. It is deliberately **not** applied to the real call's ``max_tokens``:
    doing so would silently shorten the caller's requested output, and it is
    unsafe on input-only models (e.g. GPT-5, where ``resolve_context_window``
    returns the input ceiling and output is billed against a *separate* budget
    — capping output against the input window there would truncate legitimate
    output). Consequently, if a caller sets ``max_tokens`` above this ceiling,
    ``threshold + real_max_tokens`` can still exceed the window and the provider
    may reject; that residual is tied to the unresolved input-vs-total window
    question and is left for that decision rather than papered over here.
    """
    window = resolve_context_window(model)
    buffer = _compact_buffer(window)
    requested = max(max_output_tokens or 0, _MAX_OUTPUT_TOKENS)
    ceiling = window - buffer - int(window * _MIN_INPUT_FRACTION)
    if requested <= ceiling:
        return requested
    # Surfacing this here names the cause. Downstream it only shows up as a
    # truncated response or "postcondition violated", which name the symptom.
    _warn_once(
        f"output_capped:{model}:{window}:{requested}",
        "context_output_request_capped model=%s window=%d requested=%d "
        "buffer=%d; capping output request to %d to keep room for input",
        model,
        window,
        requested,
        buffer,
        ceiling,
    )
    return ceiling


_TRUNCATION_MARKER = "\n[... truncated ...]"


def _truncate_content(content: Any, max_chars: int) -> Any:
    """Shrink a message ``content`` field to at most ``max_chars`` characters.

    Mirrors ``_content_char_len``, so the result's measured length is really
    ``<= max_chars`` for both ``str`` and multimodal ``list[dict]`` content.
    """
    if max_chars <= 0:
        return ""
    if isinstance(content, str):
        if len(content) <= max_chars:
            return content
        if max_chars <= len(_TRUNCATION_MARKER):
            return content[:max_chars]
        return content[: max_chars - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER
    if isinstance(content, list):
        out: list[Any] = []
        remaining = max_chars
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                text = item.get("text", "")
                if remaining <= 0:
                    continue
                kept_text = _truncate_content(text, remaining)
                remaining -= len(kept_text)
                out.append({**item, "text": kept_text})
            else:
                # Images and provider-specific blocks are opaque: they cannot
                # be clipped mid-payload, so they are kept whole or dropped.
                # The cost charged here is the one `_content_char_len` charges,
                # which is what keeps the postcondition enforceable.
                cost = _block_char_len(item)
                if cost > remaining:
                    continue
                remaining -= cost
                out.append(item)
        return out
    return _truncate_content(str(content) if content else "", max_chars)


def _fit_to_budget(
    messages: list[dict[str, Any]], target_tokens: int
) -> list[dict[str, Any]]:
    """Cap message contents so the whole list estimates at <= ``target_tokens``.

    Uses water-filling: find the largest per-message character cap under which
    the total fits, then truncate only the messages that exceed it. Dropping
    whole messages cannot help here — the caller has already retained the
    minimum viable tail, and a single oversized message (a giant tool result is
    the common case) must still be brought under target.

    Only ``content`` is truncated — ``tool_calls`` arguments are structured
    JSON that a provider parses, so clipping them mid-string would produce an
    invalid request. Their cost is instead subtracted from the budget up
    front, so the budget the water-filling works against is the room actually
    left for content. When tool-call payloads alone blow the budget the
    remainder goes non-positive, all content is dropped, and the caller's
    postcondition correctly reports that the target could not be met.
    """
    fixed_chars = sum(_tool_calls_char_len(m) for m in messages)
    char_budget = max(target_tokens, 0) * CHARS_PER_TOKEN - fixed_chars
    lengths = [_content_char_len(m.get("content", "") or "") for m in messages]
    if sum(lengths) <= char_budget:
        return messages

    lo, hi, cap = 0, max(lengths), 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if sum(min(x, mid) for x in lengths) <= char_budget:
            cap = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return [
        {**m, "content": _truncate_content(m.get("content"), cap)} if n > cap else m
        for m, n in zip(messages, lengths, strict=True)
    ]


def truncate_messages(
    messages: list[dict[str, Any]], target_tokens: int
) -> list[dict[str, Any]]:
    """Drop oldest messages until the history fits under ``target_tokens``.

    Deterministic last-resort fallback for when ``compact_messages`` makes no
    progress: it performs no LLM call, so it cannot raise. Keeps the leading
    system message (if any) plus as many of the most recent messages as fit,
    then truncates the retained contents so the result really lands under
    ``target_tokens``.

    Two postconditions are checked before returning, because the
    reactive-compaction call site persists this result as the conversation and
    cannot retry afterwards:

    1. The result never collapses to the system message alone when the input
       had real content. A retained tail that starts with a ``tool`` result
       would be discarded by ``normalize_messages`` as an orphan, so the
       boundary is walked back to include the originating assistant
       ``tool_calls`` message first.
    2. The result estimates at or under ``target_tokens``. Retaining one
       oversized message whole would leave the very next request over the
       provider's limit with no recovery left.

    These are explicit ``if`` checks rather than ``assert``s: ``python -O``
    strips assertions, which would silently reinstate the destroyed-conversation
    bug on a path that has no retry left. On a violation the failure is logged
    and the original ``messages`` list object is returned unchanged, matching
    ``compact_messages``' no-progress contract — callers detect it with an
    identity check (``result is messages``).
    """
    if not messages:
        return []

    try:
        return _truncate_messages_checked(messages, target_tokens)
    except Exception as e:
        # The docstring promises this cannot raise, and the reactive call site
        # relies on that — but it is not self-evidently true: normalize_messages
        # indexes `msg["role"]` unguarded, so a message without a role raises
        # KeyError. Fail closed to the no-progress contract.
        logger.warning(
            "truncate_messages failed unexpectedly (%s: %s); "
            "returning original messages unchanged",
            type(e).__name__,
            e,
        )
        return messages


def _truncate_messages_checked(
    messages: list[dict[str, Any]], target_tokens: int
) -> list[dict[str, Any]]:
    """Body of ``truncate_messages``; see there for the contract."""
    system_msg = messages[0] if messages[0].get("role") == "system" else None
    rest = messages[1:] if system_msg else list(messages)

    if not rest:
        # System-only input. This still goes through the postconditions below
        # rather than returning early: a lone oversized system prompt is
        # exactly the case that used to slip out 500x over target, unfitted
        # and unwarned.
        result = normalize_messages(list(messages))
    else:
        budget = target_tokens - (
            estimate_token_count([system_msg]) if system_msg else 0
        )

        # Newest-first walk; always retain at least the newest message.
        start = len(rest) - 1
        used = estimate_token_count([rest[start]])
        for i in range(len(rest) - 2, -1, -1):
            cost = estimate_token_count([rest[i]])
            if used + cost > budget:
                break
            start = i
            used += cost

        # Repair the boundary: a leading `tool` result is an orphan unless the
        # assistant message that issued its tool_call comes with it.
        #
        # Walk back onto that parent only while its *untruncatable* cost still
        # fits. `_fit_to_budget` shrinks `content` but never `tool_calls`
        # arguments (structured JSON a provider parses), so an oversized
        # `tool_calls` payload re-imported here can never be shed afterwards:
        # the result overshoots, the postcondition below fires, and the whole
        # truncation bails with no recovery left. Dropping the orphaned `tool`
        # result instead keeps the walk able to reach target. Content size is
        # deliberately not counted — it is always recoverable by truncation.
        fixed_used = sum(_tool_calls_char_len(m) for m in rest[start:])
        while start > 0 and rest[start].get("role") == "tool":
            parent_fixed = _tool_calls_char_len(rest[start - 1])
            if (fixed_used + parent_fixed) // CHARS_PER_TOKEN > budget:
                break
            start -= 1
            fixed_used += parent_fixed
        kept = rest[start:]
        while kept and kept[0].get("role") == "tool":
            kept = kept[1:]  # no affordable originating assistant — drop orphan

        result = normalize_messages(([system_msg] if system_msg else []) + kept)

        if not [m for m in result if m.get("role") != "system"]:
            # Everything was dropped as unpairable. Preserve the newest content
            # as a plain user message rather than a bare system prompt.
            newest = rest[-1]
            result = ([system_msg] if system_msg else []) + [
                {"role": "user", "content": newest.get("content") or ""}
            ]

    # Fit only the non-system tail. The system message is reserved whole in the
    # budget walk above, and it carries the agent's operating and safety rules;
    # water-filling caps uniformly, so including it here truncated a long
    # system prompt mid-sentence and the reactive path then persisted that as
    # the conversation's rules.
    if result and result[0].get("role") == "system":
        head, tail = result[:1], result[1:]
    else:
        head, tail = [], result
    result = head + _fit_to_budget(tail, target_tokens - estimate_token_count(head))

    input_has_content = any(m.get("role") != "system" for m in messages)
    if input_has_content and not [m for m in result if m.get("role") != "system"]:
        logger.warning(
            "truncate_messages postcondition violated (system-only result): "
            "input_messages=%d result_messages=%d target_tokens=%d; "
            "returning original messages unchanged",
            len(messages),
            len(result),
            target_tokens,
        )
        return messages

    result_tokens = estimate_token_count(result)
    if result_tokens > target_tokens:
        logger.warning(
            "truncate_messages postcondition violated (overshot target): "
            "input_messages=%d input_tokens=%d result_messages=%d "
            "result_tokens=%d target_tokens=%d; returning original messages "
            "unchanged",
            len(messages),
            estimate_token_count(messages),
            len(result),
            result_tokens,
            target_tokens,
        )
        return messages

    return result


def prune_messages(
    messages: list[dict[str, Any]], keep_last_n_turns: int = 3
) -> list[dict[str, Any]]:
    if not messages:
        return messages
    user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if keep_last_n_turns > 0 and len(user_indices) > keep_last_n_turns:
        protect_from = user_indices[-keep_last_n_turns]
    else:
        protect_from = len(messages)
    result = []
    for i, msg in enumerate(messages):
        if msg["role"] == "tool" and i < protect_from:
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": "[Previous tool result removed to save context]",
                }
            )
        else:
            result.append(msg)
    return result
