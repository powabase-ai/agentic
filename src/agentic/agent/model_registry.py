"""Model metadata — context-window resolution across provider registries.

Resolution order for a model's context window:
1. OpenRouter registry — GET /api/v1/models, cached. Consulted for ``openrouter/*``
   ids directly, and for bare ids via a provider-qualified lookup (``gpt-5`` ->
   ``openai/gpt-5``). Its ``context_length`` is the normalized TOTAL window.
2. litellm.get_model_info (with the ``openrouter/`` prefix stripped as a retry).
   Its ``max_input_tokens`` is input-only for models that bill output separately
   (GPT-5), so OpenRouter's total is preferred when available.
3. A conservative default, with a warning so unresolved models are visible.

Windows are reported here as published by the source, with no output room
deducted. Room for the response is reserved exactly once, downstream in
``compaction.get_context_threshold``, from the ``max_tokens`` the agent
actually requests. Deducting anything here as well would double-count it.
"""

from __future__ import annotations

import logging
import threading
import time

import httpx
import litellm

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_WINDOW = 128_000

_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_OPENROUTER_TTL_SECONDS = 6 * 60 * 60  # 6h
# Backoff after a failed fetch. Without it an empty cache means every call
# re-issues a 5s-timeout HTTP GET; get_context_threshold runs twice per agent
# step, so an OpenRouter outage would add ~10s of latency to every step.
_OPENROUTER_FAILURE_TTL_SECONDS = 60

_openrouter_cache: dict[str, int] = {}
_openrouter_cache_ts: float = 0.0
_openrouter_failure_ts: float = 0.0
# Serializes the refresh. The backoff above does NOT prevent a thundering
# herd: it only rate-limits *sequential* retries. Agents run concurrently via
# ThreadPoolExecutor (see orchestration/strategies.py), so on a cold cache
# every thread evaluates the staleness check before any of them writes, and
# all of them issue a simultaneous 5s GET.
_registry_lock = threading.Lock()

# Models already warned about, so a 30-step run does not emit 60-90 identical
# lines (get_context_threshold runs 2-3x per step).
_warned: set[str] = set()


def _fetch_openrouter_registry() -> dict[str, int]:
    """Fetch ``{model_id: context_window}`` from OpenRouter.

    Returns ``{}`` on any failure (network, non-200, malformed body) so callers
    degrade gracefully. ``top_provider.context_length`` wins over the root
    ``context_length`` when present.

    The raw total is stored as published. ``top_provider.max_completion_tokens``
    is deliberately NOT subtracted: it is the ceiling on what a caller *may*
    request, not what we do request. Output room is reserved exactly once, in
    ``compaction.get_context_threshold``, and is based on the ``max_tokens`` we
    actually send. Subtracting here as well double-counted it — DeepSeek
    publishes ``max_completion_tokens=384_000`` against a 1,048,576 window, so
    the conversion threw away a third of the usable context and made compaction
    fire far too early.
    """
    out: dict[str, int] = {}
    try:
        resp = httpx.get(_OPENROUTER_MODELS_URL, timeout=5.0)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        # The parse loop lives INSIDE the try on purpose. It walks
        # third-party data we do not control; anything that escapes here
        # propagates through resolve_context_window -> get_context_threshold
        # into the agent loop, which has no handler — so one malformed row
        # would fail every agent run in the fleet.
        for m in data:
            mid = None
            try:
                mid = m.get("id")
                if not mid or not isinstance(mid, str):
                    continue
                top = m.get("top_provider") or {}
                window = top.get("context_length") or m.get("context_length")
                if not window:
                    continue
                out[mid] = int(window)
            except (TypeError, ValueError, AttributeError) as e:
                # One bad row must not discard the whole catalogue. This
                # covers every exception the `.get()`-based access above can
                # actually raise on malformed JSON; KeyError is deliberately
                # not included — everything here goes through dict.get(),
                # never bracket indexing, so it cannot raise it.
                logger.warning("openrouter_registry_row_skipped id=%s: %s", mid, e)
                continue
    except Exception as e:  # best-effort; never raise into the loop
        logger.warning("openrouter_registry_fetch_failed: %s", e)
        return {}
    return out


def _openrouter_registry() -> dict[str, int]:
    """TTL-cached OpenRouter registry. Only replaces the cache on a good fetch.

    A failed fetch starts a short backoff so the next call is served from
    whatever is cached (possibly ``{}``) instead of re-hitting the network.

    The refresh is serialized under ``_registry_lock`` and the staleness check
    is re-evaluated inside it, so concurrent agent threads arriving on a cold
    cache produce exactly one HTTP GET rather than one per thread.
    """
    global _openrouter_cache, _openrouter_cache_ts, _openrouter_failure_ts

    def _needs_refresh(now: float) -> bool:
        is_stale = (
            not _openrouter_cache
            or (now - _openrouter_cache_ts) > _OPENROUTER_TTL_SECONDS
        )
        in_backoff = (now - _openrouter_failure_ts) < _OPENROUTER_FAILURE_TTL_SECONDS
        return is_stale and not in_backoff

    if _needs_refresh(time.time()):
        with _registry_lock:
            now = time.time()
            if _needs_refresh(now):  # another thread may have refreshed
                fetched: dict[str, int] = {}
                # `finally`, not `else`: if a fetch ever raises despite the
                # guards above, the backoff must still be armed. Assigning the
                # failure timestamp only on a normal return would leave every
                # subsequent call re-fetching and re-raising.
                try:
                    fetched = _fetch_openrouter_registry()
                finally:
                    if fetched:
                        _openrouter_cache = fetched
                        _openrouter_cache_ts = now
                        _openrouter_failure_ts = 0.0
                    else:
                        _openrouter_failure_ts = now
    return _openrouter_cache


def _openrouter_window_for_bare(model: str) -> int | None:
    """Total window for a non-``openrouter/``-prefixed model, via OpenRouter.

    OpenRouter's ``context_length`` is the model's **total** window (input and
    output must both fit within it), normalized across every provider. litellm's
    ``max_input_tokens`` is a raw provider passthrough: it equals the total for
    shared-budget models (gpt-4o, Claude — input and output draw from one pool)
    but is only the *input* ceiling for input-only models, where output is
    billed against a separate budget. GPT-5 reports ``max_input_tokens=272_000``
    against a documented 400_000 total; reserving output downstream against the
    272k ceiling throws away ~half the usable context and compacts far too soon.

    Preferring OpenRouter's total fixes that without a curated per-model map:
    the id is derived from litellm's own provider resolution (``gpt-5`` ->
    ``openai/gpt-5``). Any miss — provider unresolved, id absent, name-format
    mismatch (Anthropic's dashed litellm names vs OpenRouter's dotted ones) —
    returns ``None`` and the caller falls back to litellm, i.e. today's
    behavior. Since OpenRouter's total is always >= litellm's ``max_input``
    (equal for shared models, larger for input-only ones), this only ever
    widens resolution, never narrows it. Shared-budget models that miss lose
    nothing: their ``max_input`` already *is* the total.
    """
    try:
        _, provider, _, _ = litellm.get_llm_provider(model)
    except Exception:  # unresolvable provider (e.g. a bare open-weight name)
        return None
    if not provider:
        return None
    window = _openrouter_registry().get(f"{provider}/{model}")
    if window and window > 0:
        return window
    return None


def _litellm_window(model: str) -> int | None:
    try:
        info = litellm.get_model_info(model)
    except Exception:  # litellm raises for unmapped models
        return None
    window = info.get("max_input_tokens")
    if window is None:
        return None
    try:
        # Mirrors the int-coercion `_fetch_openrouter_registry` already does
        # (see :85 there) so both sources are symmetric: a non-numeric value
        # here would otherwise make `resolve_context_window` return a `str`
        # despite its `-> int` annotation, and the caller has no handler for
        # the resulting `TypeError`.
        return int(window)
    except (TypeError, ValueError):
        return None


def _warn_once(key: str, msg: str, *args: object) -> None:
    """Emit ``msg`` at WARNING the first time ``key`` is seen.

    ``get_context_threshold`` runs 2-3x per agent step, so an undeduped line
    would be emitted 60-90 times over a 30-step run.
    """
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(msg, *args)


def resolve_context_window(model: str) -> int:
    """Resolve a model's context window. See module docstring for order."""
    is_openrouter = model.startswith("openrouter/")
    if is_openrouter:
        stripped = model[len("openrouter/") :]
        window = _openrouter_registry().get(stripped)
        if window and window > 0:
            return window
    else:
        # Bare name: prefer OpenRouter's normalized TOTAL window over litellm's
        # input-only max_input_tokens (see _openrouter_window_for_bare). A miss
        # returns None and drops through to litellm — today's behavior.
        window = _openrouter_window_for_bare(model)
        if window:
            return window
    window = _litellm_window(model)
    if window is None and is_openrouter:
        window = _litellm_window(model[len("openrouter/") :])
    if window and window > 0:
        if is_openrouter:
            # The primary source (the OpenRouter registry) did not answer, so
            # this value is a degraded estimate. litellm's table under-reports
            # some openrouter ids badly — 163_840 against a true 1_048_576 for
            # DeepSeek — which is the exact production bug this module exists
            # to fix. Silently returning it would hide the regression.
            _warn_once(
                f"fallback:{model}",
                "context_window_fallback model=%s source=litellm window=%d "
                "(openrouter registry unavailable or missing this id)",
                model,
                window,
            )
        return window
    if model not in _warned:
        _warned.add(model)
        logger.warning(
            "context_window_unresolved model=%s using_default=%d",
            model,
            DEFAULT_CONTEXT_WINDOW,
        )
    return DEFAULT_CONTEXT_WINDOW
