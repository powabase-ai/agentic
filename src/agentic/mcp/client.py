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


def _is_response_to(env: dict, request_id: int) -> bool:
    env_id = env.get("id")
    return env_id == request_id or str(env_id) == str(request_id)


def _require_result_or_error(env: dict) -> dict:
    if "result" not in env and "error" not in env:
        raise McpError(
            "MCP server returned a response envelope with neither result nor error"
        )
    return env


def _response_envelope(resp: requests.Response, request_id: int) -> dict:
    """Extract the JSON-RPC envelope answering `request_id` from the body.

    A stream may interleave notifications — and server-to-client requests —
    before the response; both carry a `method` key and are skipped. Frames
    are decoded lazily and scanning stops at the exact-id match, so a
    truncated or junk trailing frame (what a dropped connection looks like)
    cannot destroy an answer that already arrived. An id-less error
    envelope, which JSON-RPC permits when the server could not read the
    request id, is kept only as a fallback — an exact match anywhere in the
    stream beats it.
    """
    content_type = (resp.headers.get("content-type") or "").lower()
    if "text/event-stream" not in content_type:
        try:
            data = resp.json()
        except ValueError as e:
            raise McpError(f"MCP server returned a non-JSON response: {e}") from e
        env = _require_envelope_is_dict(data)
        if "method" not in env and _is_response_to(env, request_id):
            return _require_result_or_error(env)
        if "method" not in env and env.get("id") is None and "error" in env:
            return env
        raise McpError(
            f"MCP server response contained no envelope answering request id "
            f"{request_id}"
        )

    # The SSE spec mandates UTF-8. Decoding resp.content directly matters: a
    # bare text/event-stream content type would make requests fall back to
    # ISO-8859-1 and mojibake every non-ASCII character.
    try:
        body = resp.content.decode("utf-8").lstrip("﻿")
    except UnicodeDecodeError as e:
        raise McpError(f"MCP server sent a non-UTF-8 SSE stream: {e}") from e

    saw_frame = False
    first_parse_error: ValueError | None = None
    idless_error_env: dict | None = None
    for payload in _iter_sse_data_payloads(body):
        saw_frame = True
        try:
            data = json.loads(payload)
        except ValueError as e:
            if first_parse_error is None:
                first_parse_error = e
            continue
        if not isinstance(data, dict) or "method" in data:
            continue
        if _is_response_to(data, request_id):
            return _require_result_or_error(data)
        if data.get("id") is None and "error" in data and idless_error_env is None:
            idless_error_env = data
    if idless_error_env is not None:
        return idless_error_env
    if not saw_frame:
        raise McpError("MCP server returned an SSE stream containing no data frame")
    if first_parse_error is not None:
        raise McpError(
            f"MCP server sent an unparseable SSE data frame: {first_parse_error}"
        ) from first_parse_error
    raise McpError(
        f"MCP server response contained no envelope answering request id {request_id}"
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
    try:
        headers = {**(server_headers or {}), "Accept": _MCP_ACCEPT}
    except TypeError as e:
        raise McpError(
            f"MCP server headers are not a mapping: {type(server_headers).__name__}"
        ) from e
    request = _jsonrpc_request(method, params)
    try:
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
    return _response_envelope(resp, request["id"])


def discover_mcp_tools(
    server_url: str,
    server_headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> list[McpToolInfo]:
    """Query an MCP server for the tools it advertises.

    `timeout` is a connect/per-read timeout, not a total deadline — a server
    that keeps the stream open with keepalives can extend a call well beyond
    it (total-deadline tracking is part of issue #15).

    Raises:
        McpError: on transport failure, an HTTP error status, an unparseable
            body, or a JSON-RPC error member in the response.
    """
    data = _post_rpc(server_url, server_headers, "tools/list", timeout=timeout)
    if "error" in data:
        try:
            message = data["error"].get("message", "unknown error")
            code = data["error"].get("code")
        except (KeyError, AttributeError, TypeError) as e:
            raise McpError(
                f"MCP tools/list returned a malformed error member: {data['error']!r}"
            ) from e
        code_note = f" (code {code})" if code is not None else ""
        raise McpError(f"MCP tools/list failed{code_note}: {message}")
    try:
        # `result: null` stays an error (AttributeError → malformed branch):
        # the key being present with a null value is a malformed response,
        # unlike the null *tool fields* guarded below.
        tools_data = data.get("result", {}).get("tools", [])
        tools = []
        for t in tools_data:
            name = t["name"]
            if not isinstance(name, str) or not name:
                raise McpError(
                    f"MCP tools/list returned a tool with an invalid name: {name!r}"
                )
            # `or`-guards, not defaults: present-but-null fields must not
            # flow through to the LLM provider as nulls, and one tool's
            # `annotations: null` must not poison the whole list.
            annotations = t.get("annotations") or {}
            tools.append(
                McpToolInfo(
                    name=name,
                    description=t.get("description") or "",
                    input_schema=t.get("inputSchema") or {"type": "object"},
                    read_only_hint=annotations.get("readOnlyHint", False),
                    destructive_hint=annotations.get("destructiveHint", False),
                    open_world_hint=annotations.get("openWorldHint", False),
                )
            )
        return tools
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
    """Call an MCP tool and return the text result. Returns error string on failure.

    `timeout` is a connect/per-read timeout, not a total deadline — a server
    that keeps the stream open with keepalives can extend a call well beyond
    it (total-deadline tracking is part of issue #15).
    """
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
                code = data["error"].get("code")
            except (KeyError, AttributeError, TypeError):
                message = f"malformed error member: {data['error']!r}"
                code = None
            code_note = f" (code {code})" if code is not None else ""
            logger.warning(
                "MCP tools/call %r on %s failed%s: %s",
                tool_name,
                server_url,
                code_note,
                message,
            )
            return f"Error{code_note}: {message}"
        result = data.get("result", {})
        content_blocks = result.get("content", [])
        text_parts = [
            b.get("text", "")
            for b in content_blocks
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        text = "".join(text_parts)
        # Per the MCP spec, tool-execution failures come back as
        # result.isError — not as JSON-RPC errors. Returning them as
        # successes would feed the ReAct loop failure text shaped like a
        # result.
        if result.get("isError"):
            logger.warning(
                "MCP tools/call %r on %s reported isError: %s",
                tool_name,
                server_url,
                text or "(no error detail)",
            )
            return f"Error: {text or 'MCP tool reported an error with no detail'}"
        return text or "(empty response)"
    except Exception as e:
        logger.warning("MCP tools/call %r on %s failed: %s", tool_name, server_url, e)
        return f"Error calling MCP tool: {e}"
