"""MCP client — tool discovery and execution via JSON-RPC over HTTP."""

from __future__ import annotations

import itertools
import logging
from typing import Any

import requests

from agentic.mcp.types import McpToolInfo

logger = logging.getLogger(__name__)
_request_id_counter = itertools.count(1)


def _next_id() -> int:
    return next(_request_id_counter)


def _jsonrpc_request(method: str, params: dict | None = None) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": _next_id(),
        "method": method,
        **({"params": params} if params else {}),
    }


def discover_mcp_tools(
    server_url: str,
    server_headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> list[McpToolInfo]:
    """Query an MCP server for available tools. Returns empty list on error."""
    try:
        resp = requests.post(
            server_url,
            json=_jsonrpc_request("tools/list"),
            headers=server_headers or {},
            timeout=timeout,
        )
        data = resp.json()
        if "error" in data:
            logger.warning("MCP tools/list error: %s", data["error"])
            return []
        tools_data = data.get("result", {}).get("tools", [])
        return [
            McpToolInfo(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {"type": "object"}),
                read_only_hint=t.get("annotations", {}).get("readOnlyHint", False),
                destructive_hint=t.get("annotations", {}).get("destructiveHint", False),
                open_world_hint=t.get("annotations", {}).get("openWorldHint", False),
            )
            for t in tools_data
        ]
    except Exception as e:
        logger.warning("MCP discover failed for %s: %s", server_url, e)
        return []


def call_mcp_tool(
    server_url: str,
    server_headers: dict[str, str] | None = None,
    tool_name: str = "",
    arguments: dict[str, Any] | None = None,
    timeout: int = 30,
) -> str:
    """Call an MCP tool and return the text result. Returns error string on failure."""
    try:
        resp = requests.post(
            server_url,
            json=_jsonrpc_request(
                "tools/call",
                {"name": tool_name, "arguments": arguments or {}},
            ),
            headers=server_headers or {},
            timeout=timeout,
        )
        data = resp.json()
        if "error" in data:
            return f"Error: {data['error'].get('message', 'Unknown MCP error')}"
        content_blocks = data.get("result", {}).get("content", [])
        text_parts = [
            b.get("text", "") for b in content_blocks if b.get("type") == "text"
        ]
        return "".join(text_parts) or "(empty response)"
    except Exception as e:
        return f"Error calling MCP tool: {e}"
