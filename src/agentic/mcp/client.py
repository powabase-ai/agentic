"""MCP client — tool discovery and execution via JSON-RPC over HTTP."""

from __future__ import annotations

import itertools
import json
from typing import Any

import requests

from agentic.mcp.types import McpToolInfo

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


_MCP_ACCEPT = "application/json, text/event-stream"


class McpError(RuntimeError):
    """An MCP request failed at the transport or protocol level."""


def _parse_rpc_response(resp) -> dict:
    """Return the JSON-RPC envelope from a JSON or SSE-framed response.

    Streamable HTTP servers may answer a POST with either a JSON body or an
    SSE stream. The envelope is carried in the stream's first `data:` frame.
    """
    content_type = (resp.headers.get("content-type") or "").lower()
    if "text/event-stream" in content_type:
        for line in resp.text.splitlines():
            if line.startswith("data:"):
                payload = line[len("data:") :].strip()
                try:
                    return json.loads(payload)
                except ValueError as e:
                    raise McpError(
                        f"MCP server sent an unparseable SSE data frame: {e}"
                    ) from e
        raise McpError("MCP server returned an SSE stream containing no data frame")
    try:
        return resp.json()
    except ValueError as e:
        raise McpError(f"MCP server returned a non-JSON response: {e}") from e


def _post_rpc(
    server_url: str,
    server_headers: dict[str, str] | None,
    method: str,
    params: dict | None = None,
    timeout: int = 30,
) -> dict:
    """POST a JSON-RPC request to an MCP server and return the decoded envelope.

    The Accept header is set last and deliberately overrides any caller value:
    the Streamable HTTP transport requires both content types, so a caller
    override could only break the request.
    """
    headers = {**(server_headers or {}), "Accept": _MCP_ACCEPT}
    try:
        resp = requests.post(
            server_url,
            json=_jsonrpc_request(method, params),
            headers=headers,
            timeout=timeout,
        )
    except Exception as e:
        raise McpError(f"MCP request to {server_url} failed: {e}") from e
    if resp.status_code >= 400:
        snippet = (resp.text or "")[:300]
        raise McpError(f"MCP server returned HTTP {resp.status_code}: {snippet}")
    return _parse_rpc_response(resp)


def discover_mcp_tools(
    server_url: str,
    server_headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> list[McpToolInfo]:
    """Query an MCP server for the tools it advertises.

    Raises:
        McpError: on transport failure, an HTTP error status, an unparseable
            body, or a JSON-RPC error member in the response.
    """
    data = _post_rpc(server_url, server_headers, "tools/list", timeout=timeout)
    if "error" in data:
        message = data["error"].get("message", "unknown error")
        raise McpError(f"MCP tools/list failed: {message}")
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


def call_mcp_tool(
    server_url: str,
    server_headers: dict[str, str] | None = None,
    tool_name: str = "",
    arguments: dict[str, Any] | None = None,
    timeout: int = 30,
) -> str:
    """Call an MCP tool and return the text result. Returns error string on failure."""
    try:
        data = _post_rpc(
            server_url,
            server_headers,
            "tools/call",
            {"name": tool_name, "arguments": arguments or {}},
            timeout,
        )
        if "error" in data:
            return f"Error: {data['error'].get('message', 'Unknown MCP error')}"
        content_blocks = data.get("result", {}).get("content", [])
        text_parts = [
            b.get("text", "") for b in content_blocks if b.get("type") == "text"
        ]
        return "".join(text_parts) or "(empty response)"
    except Exception as e:
        return f"Error calling MCP tool: {e}"
