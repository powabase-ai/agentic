"""LiteLLM cost accumulator scoped to a Celery indexing task.

Aggregates per-call cost + token counts from LiteLLM completions and groups
them by stage (e.g. "tree", "enrichment"), so a downstream caller can persist
the totals on the indexed_source row at end-of-job.

The accumulator is scoped via a ContextVar, mirroring `_reasoning_effort_var`
in agentic/src/agentic/knowledge/indexing/_pageindex_lib/utils.py. Setting
the var BEFORE `asyncio.run(...)` causes each asyncio.run snapshot to inherit
the same accumulator object, so mutations from inside multiple sequential
asyncio.run blocks all land on the same totals.

Usage from a Celery task body:

    from agentic.llm.cost_accumulator import install, init_accumulator
    install()                  # idempotent global LiteLLM callback registration
    acc = init_accumulator()   # scope to this task's thread
    asyncio.run(stage_one())
    asyncio.run(stage_two())
    stats["llm_costs"] = acc.to_dict()

Each call site that should be counted must pass `metadata={"stage": "<name>"}`
into `litellm.acompletion(...)`. Calls without metadata land in stage "unknown".
"""
from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass, field

import litellm

logger = logging.getLogger(__name__)

_acc_var: contextvars.ContextVar[CostAccumulator | None] = contextvars.ContextVar(
    "llm_cost_accumulator", default=None
)


@dataclass
class CostAccumulator:
    by_stage: dict[str, dict] = field(default_factory=dict)

    def add(
        self,
        stage: str,
        model: str | None,
        effort: str | None,
        usage,
        cost: float | None,
    ) -> None:
        if usage is None:
            return
        bucket = self.by_stage.setdefault(
            stage,
            {
                "model": model,
                "reasoning_effort": effort,
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "cost_usd": 0.0,
            },
        )
        bucket["calls"] += 1
        bucket["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        bucket["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            bucket["reasoning_tokens"] += getattr(details, "reasoning_tokens", 0) or 0
        if cost is not None:
            bucket["cost_usd"] += cost

    def to_dict(self) -> dict:
        total = sum(b["cost_usd"] for b in self.by_stage.values())
        return {
            "total_cost_usd": round(total, 6),
            "by_stage": self.by_stage,
        }


def init_accumulator() -> CostAccumulator:
    """Set a fresh accumulator on the current context and return it."""
    acc = CostAccumulator()
    _acc_var.set(acc)
    return acc


def _extract_metadata(kwargs: dict) -> dict:
    """Pull caller-supplied metadata out of the callback kwargs.

    LiteLLM relocates the user-passed `metadata` kwarg into
    `kwargs["litellm_params"]["metadata"]` before invoking the callback (see
    litellm_core_utils/litellm_logging.py:_init_kwargs_for_logging). Some
    callback paths may also leave metadata at the top level. Read both,
    preferring litellm_params since that's the canonical location.
    """
    lp = kwargs.get("litellm_params") or {}
    return lp.get("metadata") or kwargs.get("metadata") or {}


def _extract_effort(kwargs: dict) -> str | None:
    """reasoning_effort lives in different places on each provider path:
    - Responses bridge: kwargs["extra_body"]["reasoning"]["effort"]
    - Anthropic/Gemini direct: kwargs["reasoning_effort"]
    """
    extra_body = kwargs.get("extra_body") or {}
    reasoning = extra_body.get("reasoning") or {}
    effort = reasoning.get("effort")
    if effort is not None:
        return effort
    return kwargs.get("reasoning_effort")


async def _async_cb(kwargs, response, start_time, end_time):
    """LiteLLM async_success_callback. Reads the active accumulator from the
    ContextVar and adds the call's usage + cost. Silent no-op if no accumulator
    is set for this context, so unrelated LiteLLM calls (agents, retrieval,
    etc.) don't interfere.
    """
    try:
        acc = _acc_var.get()
        if acc is None:
            return
        metadata = _extract_metadata(kwargs)
        stage = metadata.get("stage", "unknown")
        cost = (getattr(response, "_hidden_params", None) or {}).get("response_cost")
        effort = _extract_effort(kwargs)
        acc.add(
            stage=stage,
            model=kwargs.get("model"),
            effort=effort,
            usage=getattr(response, "usage", None),
            cost=cost,
        )
    except Exception:
        # Telemetry must never break the indexing pipeline.
        logger.exception("cost_accumulator callback failed")


def install() -> None:
    """Register the cost-tracking callback once. Safe to call repeatedly.

    Routes through litellm's logging_callback_manager which auto-detects the
    async callable and stores it on the internal _async_success_callback list.
    LiteLLM's manager dedupes identical callable references.
    """
    if _async_cb in (litellm._async_success_callback or []):
        return
    litellm.logging_callback_manager.add_litellm_async_success_callback(_async_cb)
