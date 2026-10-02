"""Per-call routing helpers for the OpenAI Responses API bridge.

The litellm 1.83.14 mechanism for routing through the Responses API (which is
the only OpenAI endpoint that returns reasoning content) is the responses/
model prefix — verified at litellm/main.py (responses_api_bridge_check).
Models matching `(openai|azure)/responses/<model>` are stripped of the prefix
and routed via the Responses bridge.

We transform the model at Agent call time. Gemini returns reasoning_content
natively on Chat Completions when reasoning_effort is passed; no transformation
needed. Anthropic also stays on Chat Completions, but newer models hide the
reasoning summary by default, so reasoning_call_kwargs opts them into
`display: "summarized"` (see _ANTHROPIC_OMITTED_DISPLAY_FAMILIES below).

A0 verification (Task 1) revealed a LiteLLM 1.83.14 quirk: when going through
the Responses bridge, passing `reasoning_effort` at the top level is silently
dropped from the outgoing request. The verified working pattern is to pack
the `effort` into `extra_body['reasoning']` and to NOT pass top-level
reasoning_effort on the Responses path. `reasoning_call_kwargs` returns the
right shape per route.

Reasoning *summaries* are NOT requested by default. OpenAI only returns
reasoning summaries (`reasoning.summary`) for org-verified accounts; an
unverified org gets HTTP 400 "Your organization must be verified to generate
reasoning summaries" — which fails the entire call for EVERY OpenAI reasoning
model (o-series and the gpt-5 families) whenever reasoning_effort is set.
Since callers may run multi-tenant BYOK deployments where most end-user OpenAI
orgs are unverified, we omit the summary request by default. Deployments
whose OpenAI org is verified can opt back in (to surface reasoning summaries
in the UI) by setting the env var ``OPENAI_REASONING_SUMMARY=1``.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass

import litellm
from litellm.router_utils.reasoning_effort_capability import (
    nearest_declared_reasoning_effort,
    resolve_supported_reasoning_efforts,
)
from litellm.utils import get_optional_params

logger = logging.getLogger(__name__)


# Every effort level any provider accepts. "none" is the off switch, the rest
# a strength ladder, weakest first.
_STRENGTHS = ("minimal", "low", "medium", "high", "xhigh", "max")
_EFFORTS = ("none", *_STRENGTHS)

# OpenRouter's Kimi K3 entries declare no levels (litellm's own Moonshot and
# Fireworks K3 entries declare low/high/max, which litellm's resolver reads);
# Kimi K3 always reasons and takes low, high and max only.
_OPENROUTER_KIMI_K3 = (
    "openrouter/moonshotai/kimi-k3",
    "openrouter/moonshotai/kimi-k3:batch",
)
_KIMI_K3_EFFORTS = frozenset({"low", "high", "max"})


@contextmanager
def _quiet_litellm():
    """litellm prints a "Provider List" banner to stdout for every model it
    can't resolve; a capability probe on an unmapped model is not an error."""
    previous = litellm.suppress_debug_info
    litellm.suppress_debug_info = True
    try:
        yield
    finally:
        litellm.suppress_debug_info = previous


@dataclass(frozen=True)
class _ReasoningCapability:
    efforts: frozenset[str] | None
    """Levels the model accepts; None when that isn't known (pass through)."""
    always_reasons: bool
    """True when the model cannot switch reasoning off."""


def _reasoning_capability(model: str) -> _ReasoningCapability | None:
    """What ``model`` accepts as a reasoning effort, or None if it doesn't
    reason (or isn't known to litellm).

    Starts from litellm's own resolver (``resolve_supported_reasoning_
    efforts``), which reads the per-level ``supports_*_reasoning_effort``
    flags, the bare-model twin of a provider-prefixed entry, and an entry's
    declared ``reasoning_effort_levels``. Then applies what the resolver
    can't know about the routes this module builds:

    - Adaptive Claude (4.6+): the effort goes out as ``output_config.effort``,
      where neither ``none`` nor ``minimal`` is a level; with no flags, every
      other level passes. Models flagged ``thinking_always_on`` (Claude 5.5,
      Fable) can't switch thinking off.
    - Pre-adaptive Claude: unknown (pass through) -- the effort goes out as
      top-level ``reasoning_effort`` and litellm maps every level, ``none``
      included, onto a thinking budget.
    - OpenAI / Azure: a reasoning model with no flags (o-series) takes low,
      medium and high. Without a ``none`` level it can't switch reasoning off.
    - Gemini: litellm maps none through high onto the model's thinking config.
    - OpenRouter: maps any strength to the model's nearest level itself, but
      rejects ``none`` where the model always reasons, so with no flags every
      strength passes and ``none`` sends nothing.
    - OpenRouter's Kimi K3: low/high/max, always reasoning.
    """
    if model.lower() in _OPENROUTER_KIMI_K3:
        return _ReasoningCapability(_KIMI_K3_EFFORTS, always_reasons=True)
    lookup = model.replace("/responses/", "/", 1)
    try:
        with _quiet_litellm():
            _, provider, _, _ = litellm.get_llm_provider(lookup)
            if not litellm.supports_reasoning(model=lookup):
                return None
            info = dict(litellm.get_model_info(lookup))
    except Exception:
        logger.debug(
            "reasoning_capability_unknown", extra={"model": model}, exc_info=True
        )
        return None
    resolved = resolve_supported_reasoning_efforts(info, deployment_is_mapped=True)
    efforts = frozenset(resolved) if resolved is not None else None

    if "claude" in model.lower() and provider in ("anthropic", "vertex_ai", "bedrock"):
        if not (
            info.get("supports_adaptive_thinking") is True
            or _anthropic_reasoning_needs_summarized(model)
        ):
            return _ReasoningCapability(None, always_reasons=False)
        ladder = efforts if efforts is not None else frozenset(_STRENGTHS)
        return _ReasoningCapability(
            ladder - {"none", "minimal"},
            always_reasons=info.get("thinking_always_on") is True,
        )
    if provider in ("openai", "azure"):
        if efforts is None:
            efforts = frozenset({"low", "medium", "high"})
        return _ReasoningCapability(efforts, always_reasons="none" not in efforts)
    if provider == "gemini" or (provider == "vertex_ai" and "gemini" in model.lower()):
        return _ReasoningCapability(
            frozenset({"none", "minimal", "low", "medium", "high"}),
            always_reasons=False,
        )
    if provider == "openrouter" and efforts is None:
        # OpenRouter maps a strength a model lacks to its nearest level, but
        # rejects "none" on a model that always reasons.
        efforts = frozenset(_STRENGTHS)
    return _ReasoningCapability(efforts, always_reasons=False)


def supported_reasoning_efforts(model: str) -> frozenset[str] | None:
    """The effort levels ``model`` accepts, or None when that isn't known."""
    capability = _reasoning_capability(model)
    return capability.efforts if capability is not None else None


def normalize_reasoning_effort(reasoning_effort: str | None, model: str) -> str | None:
    """The effort to actually send to ``model`` for a requested level.

    - A level the model accepts is sent as is.
    - Any other strength moves to the weakest accepted level at least as
      strong, or the strongest one when it asks for more than the model has
      (litellm's ``nearest_declared_reasoning_effort``): ``medium`` on Kimi
      K3 becomes ``high``, ``max`` on o3 becomes ``high``.
    - ``none`` is the off switch, never rounded onto the ladder: on a model
      without a ``none`` level nothing is sent, which leaves reasoning off
      where the provider defaults it off. Only a model that cannot switch
      reasoning off (Claude 5.5 / Fable, Kimi K3, OpenAI reasoning models
      without ``none``) gets its weakest level instead, since sending
      nothing there means its (stronger) default.
    - A value outside every provider's vocabulary is dropped: sending no
      effort is accepted everywhere.
    - With no capability data for the model, the value passes through.
    """
    if reasoning_effort is None:
        return None
    effort = (
        reasoning_effort.strip().lower() if isinstance(reasoning_effort, str) else None
    )
    if effort not in _EFFORTS:
        logger.warning(
            "reasoning_effort_unrecognized",
            extra={"model": model, "requested_effort": reasoning_effort},
        )
        return None
    capability = _reasoning_capability(model)
    if capability is None or capability.efforts is None:
        return effort
    supported = capability.efforts
    if effort in supported:
        return effort
    strengths = [level for level in _STRENGTHS if level in supported]
    if effort == "none":
        sent = strengths[0] if capability.always_reasons and strengths else None
    elif strengths:
        sent = nearest_declared_reasoning_effort(effort, strengths)
    else:
        sent = None
    logger.info(
        "reasoning_effort_adjusted",
        extra={"model": model, "requested_effort": effort, "sent_effort": sent},
    )
    return sent


def sampling_call_kwargs(
    model: str, temperature: float | None, reasoning_effort: str | None
) -> dict:
    """``{"temperature": temperature}`` when ``model`` takes it at this
    reasoning effort, ``{}`` when it would reject it.

    An agent's temperature is set once and kept across model changes, but
    newer models refuse it: Claude 4.7+/5.x and Fable take no sampling
    parameters, GPT-5.x/6 take a non-default temperature only when the
    effort resolves to ``none``, and budget-based Claude thinking is
    incompatible with a temperature. litellm knows each model's rule, so the
    question is put to its own ``get_optional_params`` with ``drop_params``
    -- for the temperature alone, so nothing else in the request (tools,
    ``response_format``) can be dropped silently. When litellm can't answer,
    the temperature is kept, as before.
    """
    if temperature is None:
        return {}
    effort = normalize_reasoning_effort(reasoning_effort, model)
    lookup = model.replace("/responses/", "/", 1)
    try:
        with _quiet_litellm():
            bare_model, provider, _, _ = litellm.get_llm_provider(lookup)
            accepted = get_optional_params(
                model=bare_model,
                custom_llm_provider=provider,
                temperature=temperature,
                drop_params=True,
                **({"reasoning_effort": effort} if effort is not None else {}),
            )
    except Exception:
        logger.debug(
            "sampling_capability_unknown", extra={"model": model}, exc_info=True
        )
        return {"temperature": temperature}
    thinking = accepted.get("thinking")
    budget_thinking = isinstance(thinking, dict) and thinking.get("type") == "enabled"
    if "temperature" in accepted and not budget_thinking:
        return {"temperature": temperature}
    logger.info(
        "temperature_dropped",
        extra={"model": model, "temperature": temperature, "reasoning_effort": effort},
    )
    return {}


def maybe_route_through_responses(model: str, reasoning_effort: str | None) -> str:
    """For OpenAI/Azure reasoning models with reasoning_effort set, route via
    the Responses bridge by inserting `responses/` after the provider prefix.

    Provider is resolved via ``litellm.get_llm_provider`` so bare model names
    (e.g. ``gpt-5.4``, what the agent UI stores) are handled the same as
    prefixed forms (``openai/gpt-5.4``).

    Examples:
        gpt-5.4                  + medium → openai/responses/gpt-5.4   (bare → resolved to openai)
        openai/gpt-5.4           + medium → openai/responses/gpt-5.4
        azure/my-deploy          + medium → azure/responses/my-deploy
        claude-opus-4-7          + medium → unchanged (Anthropic returns reasoning natively)
        anthropic/...            + medium → unchanged
        gpt-4o                   + medium → unchanged (no reasoning support)
        openai/responses/gpt-5.4 + medium → unchanged (already routed)
    """
    if reasoning_effort is None:
        return model
    if "/responses/" in model:
        return model
    if normalize_reasoning_effort(reasoning_effort, model) in (None, "none"):
        # No reasoning to bring back, and on the Responses route the effort
        # travels in extra_body, where litellm can't see that it resolves to
        # "none" -- so it would refuse a temperature the model accepts.
        return model

    # Resolve provider for bare or prefixed model names.
    try:
        _, provider, _, _ = litellm.get_llm_provider(model)
    except Exception:
        return model

    if provider not in ("openai", "azure"):
        return model

    try:
        if not litellm.supports_reasoning(model=model):
            return model
    except Exception:
        return model

    # Insert /responses/ after the provider prefix, or synthesize one for
    # bare model names.
    if "/" in model:
        prefix, rest = model.split("/", 1)
        return f"{prefix}/responses/{rest}"
    return f"{provider}/responses/{model}"


# Anthropic models whose adaptive-thinking `display` defaults to "omitted":
# opus 4.7/4.8, opus 5/5.5, sonnet 5, fable 5, mythos 5/preview. On these,
# thinking fires but the reasoning summary text AND the reasoning token count
# are suppressed unless we ask for `display: "summarized"` explicitly (a
# silent change from opus 4.6, where "summarized" was the default). We opt in
# so reasoning is visible for eval debugging and the reasoning-display UI.
# The "opus-5" entry below matches both bare `claude-opus-5` and
# `claude-opus-5-5` -- intentional, since Opus 5's display also defaults to
# omitted, not just 5.5's.
#
# Deliberately NOT matched: opus-4-6 / sonnet-4-6 (already default to
# "summarized", so litellm's reasoning_effort path surfaces reasoning fine) and
# pre-adaptive models (opus-4-5, sonnet-4-5, claude-3-*), which reject
# `thinking.type: "adaptive"` with a 400 and must keep the reasoning_effort
# path (litellm maps effort to the right per-model shape). Matching by family
# fails safe: an unknown/new model just falls back to reasoning_effort.
_ANTHROPIC_OMITTED_DISPLAY_FAMILIES = (
    "opus-4-7",
    "opus-4-8",
    "opus-5",
    "sonnet-5",
    "fable-5",
    "mythos-5",
    "mythos-preview",
)


def _anthropic_reasoning_needs_summarized(model: str) -> bool:
    """True for Anthropic adaptive models that hide reasoning by default and
    therefore need an explicit `display: "summarized"` request."""
    m = model.lower()
    return any(family in m for family in _ANTHROPIC_OMITTED_DISPLAY_FAMILIES)


def _reasoning_summary_enabled() -> bool:
    """Whether to request an OpenAI Responses reasoning *summary*.

    Off by default — requesting a summary from an OpenAI org that isn't
    verified fails the whole call with HTTP 400 ("Your organization must be
    verified to generate reasoning summaries"). Verified-org deployments can
    opt in via ``OPENAI_REASONING_SUMMARY`` (1/true/yes/on).
    """
    return os.getenv("OPENAI_REASONING_SUMMARY", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def reasoning_call_kwargs(reasoning_effort: str | None, model: str) -> dict:
    """Return the kwargs dict to merge into litellm.completion(...) for
    requesting reasoning from this model.

    For Responses paths:          {"extra_body": {"reasoning": {"effort": effort}}}
                                  (plus "summary": "detailed" iff opt-in — see
                                  _reasoning_summary_enabled)
    For Anthropic omitted-display: {"thinking": {"type": "adaptive",
                                    "display": "summarized"},
                                    "output_config": {"effort": effort}}
    For other non-Responses paths: {"reasoning_effort": effort}

    The Responses-path packing is required because litellm 1.83.14 silently
    drops top-level `reasoning_effort` when the call routes through the
    Responses bridge. Verified empirically in A0.1. The summary is omitted by
    default to avoid the unverified-org 400 (see module docstring).

    The Anthropic branch requests `display: "summarized"` for models that
    otherwise hide reasoning (see _ANTHROPIC_OMITTED_DISPLAY_FAMILIES). Unlike
    OpenAI's reasoning summaries, Anthropic imposes no org-verification gate on
    summarized thinking, so this is always on (no opt-in flag). litellm
    forwards `thinking`/`output_config` to the Anthropic wire unchanged
    (verified offline via get_optional_params on 1.90.1).
    """
    reasoning_effort = normalize_reasoning_effort(reasoning_effort, model)
    if reasoning_effort is None:
        return {}
    if "/responses/" in model:
        reasoning: dict[str, str] = {"effort": reasoning_effort}
        if _reasoning_summary_enabled():
            reasoning["summary"] = "detailed"
        return {"extra_body": {"reasoning": reasoning}}
    if _anthropic_reasoning_needs_summarized(model):
        return {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": reasoning_effort},
        }
    return {"reasoning_effort": reasoning_effort}


def loop_reasoning_call_kwargs(reasoning_effort: str | None, model: str) -> dict:
    """``reasoning_call_kwargs`` plus what replaying reasoning between steps needs.

    On a Responses route the reply's reasoning items come back with their
    encrypted content only when asked for. With it, the items can be handed
    back on the next step whether or not the provider stores responses — the
    only form that works for a zero-data-retention organization. Only the
    OpenAI Responses route asks: reasoning items are extracted and replayed
    for the ``openai`` provider alone, so another Responses route (Azure)
    would fetch the payload for nothing. Every other route already returns
    what replay needs, so its kwargs are unchanged.

    For the agent loop and its compaction call; one-shot callers, which have
    nothing to replay, keep using ``reasoning_call_kwargs``.
    """
    kwargs = reasoning_call_kwargs(reasoning_effort, model)
    if kwargs and model.startswith("openai/responses/"):
        extra_body = dict(kwargs.get("extra_body") or {})
        extra_body["include"] = ["reasoning.encrypted_content"]
        kwargs["extra_body"] = extra_body
    return kwargs
