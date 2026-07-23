"""Hook execution engine for the tool pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from agentic.agent.rules import evaluate_rules

logger = logging.getLogger(__name__)

# Canonical hook vocabulary — the single source of truth for CRUD validation
# (routes) and dispatch (run_hooks / agent.py fire sites).
HOOK_TYPES: frozenset[str] = frozenset({"http", "rule", "approval"})
HOOK_EVENTS: frozenset[str] = frozenset(
    {
        "OnRunStart",
        "PreToolUse",
        "PostToolUse",
        "OnDelegation",
        "PreResponse",
        "OnRunComplete",
    }
)
# Events that cannot block per the documented contract: PostToolUse may only
# transform (modified_output); OnRunComplete is fire-and-forget. The agent
# executor discards `blocked` at both sites, so a block there is inert — we
# neither record it nor short-circuit the chain on it.
NON_BLOCKING_HOOK_EVENTS: frozenset[str] = frozenset({"PostToolUse", "OnRunComplete"})
# Which modification field each event's executor actually reads. Two axes matter,
# and getting either wrong makes the audit trail claim an effect that never
# happened:
#   - Event axis: an event absent from this map discards modifications entirely
#     (OnRunStart and OnDelegation consult only `blocked`; OnRunComplete is
#     fire-and-forget).
#   - Field axis: the PreResponse call site reads ONLY `modified_output`, so a
#     hook returning just `modified_input` changes nothing there — yet a
#     field-agnostic flag would trip `_preresponse_edited()` in the routes and
#     swap the persisted answer for one that drops every earlier step.
CONSUMED_MODIFICATION_FIELD: dict[str, str] = {
    "PreToolUse": "modified_input",
    "PostToolUse": "modified_output",
    "PreResponse": "modified_output",
}
# Events that carry a tool name. `matcher` filters by tool, so pairing it with
# any other event yields a hook that can never match (those events dispatch with
# tool_name="") — and an unmatched hook produces no audit record at all.
TOOL_SCOPED_HOOK_EVENTS: frozenset[str] = frozenset(
    {"PreToolUse", "PostToolUse", "OnDelegation"}
)
# Hook types whose ONLY effect is blocking — pairing them with a non-blocking
# event yields a hook that can never do anything (rejected at CRUD).
BLOCKING_ONLY_HOOK_TYPES: frozenset[str] = frozenset({"rule", "approval"})


@dataclass
class HookConfig:
    event: str  # PreToolUse, PostToolUse, PreResponse, etc.
    type: str  # "http" | "rule" | "approval"
    config: dict[str, Any]  # Type-specific config
    matcher: str | None = None  # Tool name filter (None = all)
    enabled: bool = True
    id: str | None = None
    position: int | None = None


@dataclass
class HookExecution:
    hook_id: str | None
    position: int | None
    event: str
    status: str  # "succeeded" | "failed" | "timed_out"
    latency_ms: int
    modified: bool
    blocked: bool
    message: str | None


@dataclass
class HookResult:
    blocked: bool = False
    message: str | None = None
    modified_input: dict[str, Any] | None = None
    modified_output: str | None = None
    status: str | None = None
    executions: list[HookExecution] = field(default_factory=list)


def run_hooks(
    event: str,
    tool_name: str,
    data: dict[str, Any],
    output: str | None,
    hooks: list[HookConfig],
    context: Any = None,
) -> HookResult:
    """Execute matching hooks for an event. Returns aggregated result."""
    result = HookResult()

    for hook in hooks:
        if not hook.enabled:
            continue
        if hook.event != event:
            continue
        if hook.matcher and hook.matcher != tool_name:
            continue

        # Per-hook fail-open boundary: one misconfigured hook (e.g. a non-dict
        # `config` from CRUD) must not crash the run or discard the executions
        # already recorded for hooks that DID fire.
        start = time.perf_counter()
        try:
            if hook.type == "rule":
                hook_result = _execute_rule_hook(
                    hook.config, data if output is None else {"output": output}
                )
                status = hook_result.status or "succeeded"
            elif hook.type == "http":
                hook_result = _execute_http_hook(
                    hook.config, event, tool_name, data, output, context
                )
                status = hook_result.status or "succeeded"
            elif hook.type == "approval":
                hook_result = _execute_approval_hook(
                    hook.config, tool_name, data, context
                )
                status = hook_result.status or "succeeded"
            else:
                logger.warning(
                    "Skipping hook with unknown type %r (event=%s)", hook.type, event
                )
                hook_result = HookResult(
                    status="failed", message=f"unknown hook type: {hook.type!r}"
                )
                status = "failed"
        except Exception as e:
            # Gating hooks (rule/approval) fail CLOSED on a dispatch error — a
            # malformed compliance gate must deny, not silently allow. http
            # stays fail-open (its own request failures are already handled
            # internally with fail-open; only a pre-request raise reaches here).
            fail_closed = hook.type in ("rule", "approval")
            logger.warning(
                "Hook execution raised for event=%s type=%s (%s): %s",
                event,
                hook.type,
                "fail-closed" if fail_closed else "fail-open",
                e,
            )
            hook_result = HookResult(
                status="failed", message=str(e), blocked=fail_closed
            )
            status = "failed"
        latency_ms = int((time.perf_counter() - start) * 1000)

        # A block on a non-blocking event is inert (the executor discards it),
        # so don't claim it in the audit trail and don't short-circuit the chain.
        blocked = hook_result.blocked
        if blocked and event in NON_BLOCKING_HOOK_EVENTS:
            logger.warning(
                "Hook %s returned a block on non-blocking event %s; the executor "
                "ignores it — recording blocked=False and continuing the chain.",
                hook.id,
                event,
            )
            blocked = False

        # Same honesty rule as `blocked` above, applied to modifications — but
        # resolved against the ONE field this event's call site consumes.
        returned_modification = (
            hook_result.modified_output is not None
            or hook_result.modified_input is not None
        )
        consumed_field = CONSUMED_MODIFICATION_FIELD.get(event)
        applied_value = getattr(hook_result, consumed_field) if consumed_field else None
        # `not blocked`: when a hook blocks, every call site uses the block and
        # discards the modification, so claiming one would be another audit lie.
        modification_applies = applied_value is not None and not blocked
        if returned_modification and not modification_applies:
            logger.warning(
                "Hook %s returned a modification that event %s does not consume "
                "(it reads %s) — recording modified=False and dropping it.",
                hook.id,
                event,
                consumed_field or "nothing",
            )

        result.executions.append(
            HookExecution(
                hook_id=hook.id,
                position=hook.position,
                event=event,
                status=status,
                latency_ms=latency_ms,
                modified=returned_modification and modification_applies,
                blocked=blocked,
                message=hook_result.message,
            )
        )

        if blocked:
            result.blocked = True
            result.message = hook_result.message
            return result
        # `is not None`, not truthiness: a hook may legitimately return a falsy
        # modification (`""` = redact the answer entirely, `{}` = strip all
        # tool args). Truthiness would drop those while the audit record above
        # still reports modified=True — an audit trail that claims a redaction
        # the caller never received. Absent keys arrive as None and are skipped.
        if modification_applies:
            setattr(result, consumed_field, applied_value)

    return result


def _execute_rule_hook(config: dict, data: dict) -> HookResult:
    rules = [config] if "condition" in config else config.get("rules", [])
    if not rules:
        # No evaluable rules — the gate allows everything. Report it as failed so
        # a typo'd config ("conditon") is distinguishable from a genuine pass in
        # the audit trail. Deliberately NOT blocked: an empty rule set is a
        # config-semantics question, not a crashed gate, and flipping it to deny
        # would silently convert an allowing hook into a denying one on deploy.
        # New configs of this shape are rejected at CRUD.
        return HookResult(status="failed", message="rule hook has no evaluable rules")
    allowed, reason = evaluate_rules(data, rules)
    if not allowed:
        return HookResult(blocked=True, message=reason)
    return HookResult()


def _execute_http_hook(
    config: dict,
    event: str,
    tool_name: str,
    data: dict,
    output: str | None,
    context: Any = None,
) -> HookResult:
    url = config.get("url")
    if not url:
        return HookResult(status="failed", message="hook url not configured")

    timeout = config.get("timeout_seconds", 5)
    headers = config.get("headers", {})

    payload = {"event": event, "tool_name": tool_name, "data": data}
    if output is not None:
        payload["output"] = output
    if config.get("include_run_context") is True and context is not None:
        payload["run"] = {
            "orchestration_run_id": getattr(context, "orchestration_run_id", None),
            "execution_id": getattr(context, "execution_id", None),
            # No parent_run_id: it is only set by ExecutionContext.child_context(),
            # and hooks never fire on sub-agent contexts (they are supervisor-only
            # and not propagated downward). Emitting an always-null field would
            # invite tenant webhooks to branch on something that can never occur.
        }

    try:
        from agentic.agent.url_validation import SSRFError, validate_url

        try:
            validate_url(url)
        except SSRFError as e:
            # Fail-open like every other http failure: the SSRF guard already
            # prevented the request, so the security win is preserved — aborting
            # the whole run on top of that would let a URL typo brick every run.
            logger.warning("Hook URL blocked by SSRF protection (fail-open): %s", e)
            return HookResult(
                blocked=False, message=f"Hook URL blocked: {e}", status="failed"
            )

        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if response.status_code == 200:
            body = response.json()
            action = body.get("action", "allow")
            if action == "deny":
                # Carry any supplied replacement through. On a blocking event
                # the block wins and this is ignored; on a non-blocking event
                # (where the block is inert per NON_BLOCKING_HOOK_EVENTS) it
                # degrades the deny into the redaction the tenant asked for,
                # instead of dropping both.
                return HookResult(
                    blocked=True,
                    message=body.get("message", "Blocked by webhook"),
                    modified_input=body.get("modified_input"),
                    modified_output=body.get("modified_output"),
                    status="succeeded",
                )
            return HookResult(
                modified_input=body.get("modified_input"),
                modified_output=body.get("modified_output"),
                status="succeeded",
            )
        else:
            logger.warning(
                "Hook HTTP returned status %s (event=%s, fail-open)",
                response.status_code,
                event,
            )
            return HookResult(status="failed", message=f"HTTP {response.status_code}")
    except requests.exceptions.Timeout as e:
        logger.warning("Hook HTTP call timed out (event=%s, fail-open): %s", event, e)
        return HookResult(status="timed_out", message=f"{type(e).__name__}: {e}")
    except Exception as e:
        logger.warning("Hook HTTP call failed (event=%s, fail-open): %s", event, e)
        return HookResult(status="failed", message=f"{type(e).__name__}: {e}")


def _execute_approval_hook(
    config: dict,
    tool_name: str,
    data: dict,
    context: Any,
) -> HookResult:
    if context is None:
        return HookResult(
            blocked=True,
            message="Approval requires ExecutionContext",
            status="failed",
        )

    context.reset_approval()

    context.emit_event(
        {
            "type": "approval_requested",
            "tool_name": tool_name,
            "tool_input": data,
            "message": config.get("message", "Approval required"),
        }
    )

    # Check if a decision was already set before we even asked
    decision = context.get_approval_decision()
    if decision is None:
        approval_event = context.get_approval_event()
        timeout = config.get("timeout", 300)
        approval_event.wait(timeout=timeout)
        decision = context.get_approval_decision()
    if decision is None:
        return HookResult(
            blocked=True, message="Approval timed out", status="timed_out"
        )
    if not decision.get("approved"):
        return HookResult(
            blocked=True,
            message=f"Denied by user: {decision.get('reason', '')}",
            status="succeeded",
        )
    return HookResult(blocked=False, status="succeeded")
