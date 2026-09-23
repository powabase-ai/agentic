"""Prompt-cache helpers: deterministic tool ordering and explicit breakpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import litellm

if TYPE_CHECKING:
    from agentic.agent.tools import ToolDefinition

_TYPE_PRIORITY = {
    "BuiltinTool": 0,
    "KnowledgeSearchTool": 1,
    "CustomTool": 2,
    "McpTool": 3,
    "DelegateTool": 4,
}

# Providers that accept litellm's `cache_control` on every Claude model.
# Bedrock (where Converse emits `cachePoint` blocks) supports prompt caching
# only on some Claude models, so it is gated on litellm's model map instead.
_EXPLICIT_BREAKPOINT_PROVIDERS = frozenset({"anthropic", "vertex_ai"})
_MAX_BREAKPOINTS = 4
_EPHEMERAL = {"type": "ephemeral"}


def sort_tools_for_cache(tools: dict[str, ToolDefinition]) -> list[ToolDefinition]:
    """Sort tools deterministically for prompt cache stability."""

    def sort_key(tool: ToolDefinition) -> tuple[int, str]:
        type_name = type(tool).__name__
        priority = _TYPE_PRIORITY.get(type_name, 99)
        return (priority, tool.name)

    return sorted(tools.values(), key=sort_key)


def uses_explicit_cache_breakpoints(model: str) -> bool:
    """True when ``model`` is Claude on a route where explicit ``cache_control``
    breakpoints are known to work: Anthropic and Vertex AI for any Claude
    model, Bedrock for the Claude models litellm's model map lists with prompt
    caching.

    Everything else is left unmarked and pays the uncached rate: providers
    that cache automatically (OpenAI, DeepSeek, Gemini), and Claude routes
    not verified with these breakpoints (OpenRouter, Azure AI, a LiteLLM
    proxy, Bedrock ids the model map does not know, such as inference-profile
    ARNs). Fails closed: a model litellm cannot resolve is never marked.
    """
    if "claude" not in model.lower():
        return False
    try:
        _, provider, _, _ = litellm.get_llm_provider(model)
        if provider == "bedrock":
            return bool(litellm.utils.supports_prompt_caching(model=model))
    except Exception:
        return False
    return provider in _EXPLICIT_BREAKPOINT_PROVIDERS


def add_cache_breakpoints(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    mark_last_message: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Return copies of ``messages`` and ``tools`` carrying cache breakpoints.

    The provider caches the request prefix — tools, then system, then
    messages — up to each breakpoint, and allows at most four per request,
    including any the caller already set inside message content. In priority
    order this spends up to three of what is left:

    1. the last message — it moves every step, so each call reads the
       history the previous call wrote;
    2. the leading system message — large, fixed for the run, and a read
       point that survives a rewritten history;
    3. the last tool definition — tools open the prefix, though a read at the
       system breakpoint already covers them.

    ``mark_last_message=False`` skips (1), for a request whose tail nothing
    reads back — compaction's summary instruction.

    A spot that already carries a breakpoint keeps it, TTL included, at no
    cost. The inputs are never mutated: a breakpoint belongs to one request,
    and left in stored history it would still be there next step. Models
    without explicit breakpoints get the inputs back unchanged.
    """
    if not uses_explicit_cache_breakpoints(model):
        return messages, tools

    budget = _MAX_BREAKPOINTS - _count_breakpoints(messages, tools)
    marked = list(messages)
    targets = []
    if marked and mark_last_message:
        targets.append(len(marked) - 1)
    if len(marked) > 1 and marked[0].get("role") == "system":
        targets.append(0)
    for i in targets:
        if "cache_control" in _breakpoint_spot(marked[i]):
            continue
        if budget <= 0:
            break
        marked[i] = _with_breakpoint(marked[i])
        budget -= 1

    if tools and budget > 0 and "cache_control" not in tools[-1]:
        tools = [*tools[:-1], {**tools[-1], "cache_control": dict(_EPHEMERAL)}]
    return marked, tools


def _count_breakpoints(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
) -> int:
    """Breakpoints the request already carries, wherever litellm reads them:
    on a message, a content block, a tool, or a tool's ``function``."""
    count = 0
    for message in messages:
        if "cache_control" in message:
            count += 1
        content = message.get("content")
        if isinstance(content, list):
            count += sum(
                1
                for block in content
                if isinstance(block, dict) and "cache_control" in block
            )
    for tool in tools or []:
        if "cache_control" in tool or "cache_control" in (tool.get("function") or {}):
            count += 1
    return count


def _breakpoint_spot(message: dict[str, Any]) -> dict[str, Any]:
    """The dict litellm reads this message's breakpoint from.

    litellm moves a message-level ``cache_control`` onto the provider block
    for string content and for tool results; list content needs it on the
    last block.
    """
    content = message.get("content")
    if (
        message.get("role") == "tool"
        or not isinstance(content, list)
        or not content
        or not isinstance(content[-1], dict)
    ):
        return message
    return content[-1]


def _with_breakpoint(message: dict[str, Any]) -> dict[str, Any]:
    """Copy of ``message`` with a breakpoint at its ``_breakpoint_spot``."""
    if _breakpoint_spot(message) is message:
        return {**message, "cache_control": dict(_EPHEMERAL)}
    content = message["content"]
    last = {**content[-1], "cache_control": dict(_EPHEMERAL)}
    return {**message, "content": [*content[:-1], last]}
