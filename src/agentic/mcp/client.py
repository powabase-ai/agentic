"""MCP client — tool discovery and execution via JSON-RPC over HTTP."""

from __future__ import annotations

import itertools
import json
import logging
from typing import Any

import requests

from agentic.mcp.types import McpToolInfo

logger = logging.getLogger(__name__)

_request_id_counter = itertools.count(1)

_MCP_ACCEPT = "application/json, text/event-stream"


class McpError(RuntimeError):
    """An MCP request failed at the transport or protocol level."""


def _next_id() -> int:
    return next(_request_id_counter)


def _jsonrpc_request(method: str, params: dict | None = None) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": _next_id(),
        "method": method,
        **({"params": params} if params else {}),
    }


def _require_envelope_is_dict(data: Any) -> dict:
    if not isinstance(data, dict):
        raise McpError(
            "MCP server returned a JSON-RPC envelope that is not an object: "
            f"{type(data).__name__}"
        )
    return data


def _iter_sse_data_payloads(body: str):
    """Yield each SSE event's data payload, assembled per the SSE spec.

    An event's `data:` lines accumulate until a blank line dispatches the
    event; multi-line data joins with newlines. Other fields (`event:`,
    `id:`, `retry:`) and comment lines are ignored.
    """
    data_lines: list[str] = []
    for raw_line in body.split("\n"):
        line = raw_line.rstrip("\r")
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith("data:"):
            value = line[len("data:") :]
            if value.startswith(" "):
                value = value[1:]
            data_lines.append(value)
    if data_lines:
        yield "\n".join(data_lines)


def _envelopes_from_response(resp: requests.Response) -> list[dict]:
    """Decode every JSON-RPC envelope in a JSON or SSE-framed response body."""
    content_type = (resp.headers.get("content-type") or "").lower()
    if "text/event-stream" in content_type:
        # The SSE spec mandates UTF-8. Decoding resp.content directly matters:
        # a bare text/event-stream content type would make requests fall back
        # to ISO-8859-1 and mojibake every non-ASCII character.
        try:
            body = resp.content.decode("utf-8")
        except UnicodeDecodeError as e:
            raise McpError(f"MCP server sent a non-UTF-8 SSE stream: {e}") from e
        envelopes = []
        for payload in _iter_sse_data_payloads(body):
            try:
                data = json.loads(payload)
            except ValueError as e:
                raise McpError(
                    f"MCP server sent an unparseable SSE data frame: {e}"
                ) from e
            envelopes.append(_require_envelope_is_dict(data))
        if not envelopes:
            raise McpError("MCP server returned an SSE stream containing no data frame")
        return envelopes
    try:
        data = resp.json()
    except ValueError as e:
        raise McpError(f"MCP server returned a non-JSON response: {e}") from e
    return [_require_envelope_is_dict(data)]


def _select_response_envelope(envelopes: list[dict], request_id: int) -> dict:
    """Pick the envelope that answers `request_id`.

    A stream may interleave notifications — and server-to-client requests —
    before the response; both carry a `method` key and are skipped. The
    response is the envelope whose id matches the request's, or an id-less
    error envelope, which JSON-RPC permits when the server could not read
    the request id.
    """
    for env in envelopes:
        if "method" in env:
            continue
        env_id = env.get("id")
        if env_id == request_id or str(env_id) == str(request_id):
            if "result" not in env and "error" not in env:
                raise McpError(
                    "MCP server returned a response envelope with neither "
                    "result nor error"
                )
            return env
        if env_id is None and "error" in env:
            return env
    raise McpError(
        f"MCP server response contained no envelope answering request id "
        f"{request_id} ({len(envelopes)} envelope(s) received)"
    )


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
    request = _jsonrpc_request(method, params)
    try:
        headers = {**(server_headers or {}), "Accept": _MCP_ACCEPT}
        resp = requests.post(
            server_url,
            json=request,
            headers=headers,
            timeout=timeout,
        )
    except Exception as e:
        raise McpError(f"MCP request to {server_url} failed: {e}") from e
    if resp.status_code >= 400:
        snippet = (resp.text or "")[:300]
        raise McpError(f"MCP server returned HTTP {resp.status_code}: {snippet}")
    return _select_response_envelope(_envelopes_from_response(resp), request["id"])


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
        try:
            message = data["error"].get("message", "unknown error")
        except (KeyError, AttributeError, TypeError) as e:
            raise McpError(
                f"MCP tools/list returned a malformed error member: {data['error']!r}"
            ) from e
        raise McpError(f"MCP tools/list failed: {message}")
    try:
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
    except (KeyError, AttributeError, TypeError) as e:
        raise McpError(
            f"MCP tools/list returned a malformed tools/list result: {e}"
        ) from e


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
            timeout=timeout,
        )
        if "error" in data:
            try:
                message = data["error"].get("message", "Unknown MCP error")
            except (KeyError, AttributeError, TypeError):
                message = f"malformed error member: {data['error']!r}"
            logger.warning(
                "MCP tools/call %r on %s failed: %s", tool_name, server_url, message
            )
            return f"Error: {message}"
        content_blocks = data.get("result", {}).get("content", [])
        text_parts = [
            b.get("text", "")
            for b in content_blocks
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "".join(text_parts) or "(empty response)"
    except Exception as e:
        logger.warning("MCP tools/call %r on %s failed: %s", tool_name, server_url, e)
        return f"Error calling MCP tool: {e}"
