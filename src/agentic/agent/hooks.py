"""Hook execution engine for the tool pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import requests

from agentic.agent.rules import evaluate_rules

logger = logging.getLogger(__name__)


@dataclass
class HookConfig:
    event: str  # PreToolUse, PostToolUse, PreResponse, etc.
    type: str  # "http" | "rule"
    config: dict[str, Any]  # Type-specific config
    matcher: str | None = None  # Tool name filter (None = all)
    enabled: bool = True


@dataclass
class HookResult:
    blocked: bool = False
    message: str | None = None
    modified_input: dict[str, Any] | None = None
    modified_output: str | None = None


def run_hooks(
    event: str,
    tool_name: str,
    data: dict[str, Any],
    output: str | None,
    hooks: list[HookConfig],
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

        if hook.type == "rule":
            hook_result = _execute_rule_hook(
                hook.config, data if output is None else {"output": output}
            )
        elif hook.type == "http":
            hook_result = _execute_http_hook(
                hook.config, event, tool_name, data, output
            )
        else:
            continue

        if hook_result.blocked:
            return hook_result
        if hook_result.modified_input:
            result.modified_input = hook_result.modified_input
        if hook_result.modified_output:
            result.modified_output = hook_result.modified_output

    return result


def _execute_rule_hook(config: dict, data: dict) -> HookResult:
    rules = [config] if "condition" in config else config.get("rules", [])
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
) -> HookResult:
    url = config.get("url")
    if not url:
        return HookResult()

    timeout = config.get("timeout_seconds", 5)
    headers = config.get("headers", {})

    payload = {
        "event": event,
        "tool_name": tool_name,
        "data": data,
    }
    if output is not None:
        payload["output"] = output

    try:
        from agentic.agent.url_validation import SSRFError, validate_url

        try:
            validate_url(url)
        except SSRFError as e:
            logger.error("Hook URL blocked by SSRF protection: %s", e)
            return HookResult(blocked=True, message=f"Hook URL blocked: {e}")

        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if response.status_code == 200:
            body = response.json()
            action = body.get("action", "allow")
            if action == "deny":
                return HookResult(
                    blocked=True, message=body.get("message", "Blocked by webhook")
                )
            return HookResult(
                modified_input=body.get("modified_input"),
                modified_output=body.get("modified_output"),
            )
        else:
            logger.warning(
                "Hook HTTP returned status %s (fail-open)", response.status_code
            )
    except Exception as e:
        logger.warning("Hook HTTP call failed (fail-open): %s", e)

    return HookResult()  # Fail-open
