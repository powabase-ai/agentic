from unittest.mock import MagicMock, patch

import pytest
import requests

from agentic.mcp.client import McpError, call_mcp_tool, discover_mcp_tools

# The autouse id-counter reset lives in tests/conftest.py — any MCP-touching
# test file may hardcode "id": 1 in fixture bodies.


class TestDiscoverMcpTools:
    @patch("agentic.mcp.client.requests.post")
    def test_discover_returns_tools(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(
                return_value={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "tools": [
                            {
                                "name": "create_issue",
                                "description": "Create a GitHub issue",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"title": {"type": "string"}},
                                    "required": ["title"],
                                },
                                "annotations": {
                                    "readOnlyHint": False,
                                    "destructiveHint": False,
                                },
                            },
                            {
                                "name": "list_repos",
                                "description": "List repositories",
                                "inputSchema": {"type": "object", "properties": {}},
                                "annotations": {"readOnlyHint": True},
                            },
                            {
                                "name": "delete_repo",
                                "description": "Delete a repository",
                                "inputSchema": {"type": "object", "properties": {}},
                                "annotations": {"destructiveHint": True},
                            },
                        ]
                    },
                }
            ),
        )

        tools = discover_mcp_tools(
            server_url="https://mcp.example.com",
            server_headers={"Authorization": "Bearer token"},
        )

        assert len(tools) == 3
        assert tools[0].name == "create_issue"
        assert tools[0].description == "Create a GitHub issue"
        assert tools[0].input_schema["properties"]["title"]["type"] == "string"
        assert tools[0].read_only_hint is False
        assert tools[0].destructive_hint is False
        assert tools[1].name == "list_repos"
        assert tools[1].read_only_hint is True
        assert tools[1].destructive_hint is False
        assert tools[2].name == "delete_repo"
        assert tools[2].destructive_hint is True
        assert tools[2].read_only_hint is False

    @patch("agentic.mcp.client.requests.post")
    def test_discover_empty_server(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(
                return_value={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"tools": []},
                }
            ),
        )
        tools = discover_mcp_tools(server_url="https://mcp.example.com")
        assert tools == []

    @patch("agentic.mcp.client.requests.post")
    def test_discover_raises_on_transport_error(self, mock_post):
        mock_post.side_effect = Exception("Connection refused")
        with pytest.raises(McpError) as exc:
            discover_mcp_tools(server_url="https://bad.example.com")
        assert "Connection refused" in str(exc.value)


class TestCallMcpTool:
    @patch("agentic.mcp.client.requests.post")
    def test_call_returns_text_result(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(
                return_value={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "content": [{"type": "text", "text": "Issue #42 created"}],
                    },
                }
            ),
        )

        result = call_mcp_tool(
            server_url="https://mcp.example.com",
            server_headers={},
            tool_name="create_issue",
            arguments={"title": "Bug report"},
        )

        assert result == "Issue #42 created"
        call_body = mock_post.call_args.kwargs["json"]
        assert call_body["method"] == "tools/call"
        assert call_body["params"]["name"] == "create_issue"
        assert call_body["params"]["arguments"] == {"title": "Bug report"}

    @patch("agentic.mcp.client.requests.post")
    def test_call_handles_multiple_content_blocks(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(
                return_value={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "content": [
                            {"type": "text", "text": "Part 1. "},
                            {"type": "text", "text": "Part 2."},
                        ],
                    },
                }
            ),
        )
        result = call_mcp_tool(
            server_url="https://mcp.example.com",
            server_headers={},
            tool_name="tool",
            arguments={},
        )
        assert result == "Part 1. Part 2."

    @patch("agentic.mcp.client.requests.post")
    def test_call_handles_error_response(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(
                return_value={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "error": {"code": -32600, "message": "Invalid request"},
                }
            ),
        )
        result = call_mcp_tool(
            server_url="https://mcp.example.com",
            server_headers={},
            tool_name="tool",
            arguments={},
        )
        assert result.startswith("Error")
        assert "Invalid request" in result

    @patch("agentic.mcp.client.requests.post")
    def test_call_handles_connection_error(self, mock_post):
        mock_post.side_effect = Exception("Connection refused")
        result = call_mcp_tool(
            server_url="https://bad.example.com",
            server_headers={},
            tool_name="tool",
            arguments={},
        )
        assert "error" in result.lower()


class TestStreamableHttpHeaders:
    ACCEPT = "application/json, text/event-stream"

    @patch("agentic.mcp.client.requests.post")
    def test_discover_sends_accept_header(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(
                return_value={"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}
            ),
        )
        discover_mcp_tools(server_url="https://mcp.example.com")
        assert mock_post.call_args.kwargs["headers"]["Accept"] == self.ACCEPT

    @patch("agentic.mcp.client.requests.post")
    def test_call_sends_accept_header(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(
                return_value={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"content": [{"type": "text", "text": "ok"}]},
                }
            ),
        )
        call_mcp_tool(server_url="https://mcp.example.com", tool_name="t", arguments={})
        assert mock_post.call_args.kwargs["headers"]["Accept"] == self.ACCEPT

    @patch("agentic.mcp.client.requests.post")
    def test_caller_headers_preserved_but_accept_wins(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(
                return_value={"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}
            ),
        )
        discover_mcp_tools(
            server_url="https://mcp.example.com",
            server_headers={"Authorization": "Bearer token", "Accept": "text/plain"},
        )
        sent = mock_post.call_args.kwargs["headers"]
        assert sent["Authorization"] == "Bearer token"
        assert sent["Accept"] == self.ACCEPT


SSE_TOOLS_BODY = (
    "event: message\n"
    'data: {"jsonrpc": "2.0", "id": 1, "result": {"tools": [{"name": "search", '
    '"description": "Search things", "inputSchema": {"type": "object"}}]}}\n'
    "\n"
)

SSE_CALL_BODY = (
    "event: message\n"
    'data: {"jsonrpc": "2.0", "id": 1, "result": {"content": '
    '[{"type": "text", "text": "hello from sse"}]}}\n'
    "\n"
)


class TestSseFramedResponses:
    @patch("agentic.mcp.client.requests.post")
    def test_discover_parses_sse_body(self, mock_post):
        mock_post.return_value = _sse_response(SSE_TOOLS_BODY)
        tools = discover_mcp_tools(server_url="https://mcp.example.com")
        assert len(tools) == 1
        assert tools[0].name == "search"
        assert tools[0].input_schema == {"type": "object"}

    @patch("agentic.mcp.client.requests.post")
    def test_call_parses_sse_body(self, mock_post):
        mock_post.return_value = _sse_response(SSE_CALL_BODY)
        result = call_mcp_tool(
            server_url="https://mcp.example.com", tool_name="t", arguments={}
        )
        assert result == "hello from sse"


HTTP_406_BODY = (
    '{"jsonrpc": "2.0", "id": "server-error", "error": {"code": -32600, '
    '"message": "Not Acceptable: Client must accept both application/json '
    'and text/event-stream"}}'
)


class TestErrorSurfacing:
    @patch("agentic.mcp.client.requests.post")
    def test_discover_raises_on_http_error(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=406,
            headers={"content-type": "application/json"},
            text=HTTP_406_BODY,
        )
        with pytest.raises(McpError) as exc:
            discover_mcp_tools(server_url="https://mcp.example.com")
        assert "406" in str(exc.value)
        assert "Not Acceptable" in str(exc.value)

    @patch("agentic.mcp.client.requests.post")
    def test_discover_raises_on_jsonrpc_error(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(
                return_value={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "error": {"code": -32601, "message": "Method not found"},
                }
            ),
        )
        with pytest.raises(McpError) as exc:
            discover_mcp_tools(server_url="https://mcp.example.com")
        assert "Method not found" in str(exc.value)

    @patch("agentic.mcp.client.requests.post")
    def test_call_returns_error_string_on_http_error(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=406,
            headers={"content-type": "application/json"},
            text=HTTP_406_BODY,
        )
        result = call_mcp_tool(
            server_url="https://mcp.example.com", tool_name="t", arguments={}
        )
        assert "Error calling MCP tool" in result
        assert "Not Acceptable" in result

    def test_mcp_error_is_exported_from_package(self):
        from agentic.mcp import McpError as Exported

        assert Exported is McpError


class TestUnparseableResponses:
    @patch("agentic.mcp.client.requests.post")
    def test_discover_raises_on_sse_stream_without_data_frame(self, mock_post):
        mock_post.return_value = _sse_response("event: message\n\n")
        with pytest.raises(McpError) as exc:
            discover_mcp_tools(server_url="https://mcp.example.com")
        assert "no data frame" in str(exc.value)

    @patch("agentic.mcp.client.requests.post")
    def test_discover_raises_on_unparseable_sse_data_frame(self, mock_post):
        mock_post.return_value = _sse_response("event: message\ndata: not-json\n\n")
        with pytest.raises(McpError) as exc:
            discover_mcp_tools(server_url="https://mcp.example.com")
        assert "unparseable SSE data frame" in str(exc.value)

    @patch("agentic.mcp.client.requests.post")
    def test_discover_raises_on_non_json_body(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(side_effect=ValueError("Expecting value")),
        )
        with pytest.raises(McpError) as exc:
            discover_mcp_tools(server_url="https://mcp.example.com")
        assert "non-JSON response" in str(exc.value)


class TestMalformedSuccessPayloads:
    @patch("agentic.mcp.client.requests.post")
    def test_discover_raises_on_tool_entry_missing_name(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(
                return_value={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"tools": [{"description": "no name field"}]},
                }
            ),
        )
        with pytest.raises(McpError):
            discover_mcp_tools(server_url="https://mcp.example.com")

    @patch("agentic.mcp.client.requests.post")
    def test_discover_raises_on_error_member_not_a_dict(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(return_value={"jsonrpc": "2.0", "id": 1, "error": "boom"}),
        )
        with pytest.raises(McpError):
            discover_mcp_tools(server_url="https://mcp.example.com")

    @patch("agentic.mcp.client.requests.post")
    def test_discover_raises_on_tools_entry_not_a_dict(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(
                return_value={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"tools": ["nope"]},
                }
            ),
        )
        with pytest.raises(McpError):
            discover_mcp_tools(server_url="https://mcp.example.com")

    @patch("agentic.mcp.client.requests.post")
    def test_discover_raises_on_result_that_is_a_string(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(return_value={"jsonrpc": "2.0", "id": 1, "result": "oops"}),
        )
        with pytest.raises(McpError):
            discover_mcp_tools(server_url="https://mcp.example.com")

    @patch("agentic.mcp.client.requests.post")
    def test_discover_raises_on_result_that_is_a_list(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(return_value={"jsonrpc": "2.0", "id": 1, "result": ["x"]}),
        )
        with pytest.raises(McpError):
            discover_mcp_tools(server_url="https://mcp.example.com")

    @patch("agentic.mcp.client.requests.post")
    def test_discover_raises_on_result_that_is_null(self, mock_post):
        # `.get("result", {})` returns None here because the key IS present
        # with a null value — the {} default never applies.
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(return_value={"jsonrpc": "2.0", "id": 1, "result": None}),
        )
        with pytest.raises(McpError):
            discover_mcp_tools(server_url="https://mcp.example.com")


class TestNonDictEnvelopes:
    """A 2xx body that decodes to valid JSON that isn't a JSON object at all."""

    @patch("agentic.mcp.client.requests.post")
    def test_discover_raises_on_null_body(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(return_value=None),
        )
        with pytest.raises(McpError):
            discover_mcp_tools(server_url="https://mcp.example.com")

    @patch("agentic.mcp.client.requests.post")
    def test_discover_raises_on_integer_body(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(return_value=5),
        )
        with pytest.raises(McpError):
            discover_mcp_tools(server_url="https://mcp.example.com")

    @patch("agentic.mcp.client.requests.post")
    def test_discover_raises_on_list_body(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(return_value=[1, 2, 3]),
        )
        with pytest.raises(McpError):
            discover_mcp_tools(server_url="https://mcp.example.com")

    @patch("agentic.mcp.client.requests.post")
    def test_discover_raises_on_string_body(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(return_value="hello"),
        )
        with pytest.raises(McpError):
            discover_mcp_tools(server_url="https://mcp.example.com")

    @patch("agentic.mcp.client.requests.post")
    def test_discover_raises_on_null_sse_body(self, mock_post):
        # json.loads("null") returns None, so the SSE branch needs the same
        # non-dict guard as the resp.json() branch.
        mock_post.return_value = _sse_response("event: message\ndata: null\n\n")
        with pytest.raises(McpError):
            discover_mcp_tools(server_url="https://mcp.example.com")


def _sse_response(body: str, charset: str | None = None) -> requests.Response:
    """A real requests.Response carrying an SSE body, as a server would send it."""
    resp = requests.Response()
    resp.status_code = 200
    content_type = "text/event-stream"
    if charset:
        content_type += f"; charset={charset}"
    resp.headers["content-type"] = content_type
    resp._content = body.encode("utf-8")
    return resp


SSE_NOTIFICATIONS_THEN_CALL_RESPONSE = (
    "event: message\n"
    'data: {"jsonrpc": "2.0", "method": "notifications/message", '
    '"params": {"level": "info", "data": "working on it"}}\n'
    "\n"
    "event: message\n"
    'data: {"jsonrpc": "2.0", "method": "notifications/progress", '
    '"params": {"progressToken": "t", "progress": 1}}\n'
    "\n"
    "event: message\n"
    'data: {"jsonrpc": "2.0", "id": 1, "result": {"content": '
    '[{"type": "text", "text": "final answer"}]}}\n'
    "\n"
)


class TestSseResponseSelection:
    """The response is the envelope answering our request id — never just the
    first frame. Compliant servers (FastMCP among them) interleave
    notifications on the POST stream before the response."""

    @patch("agentic.mcp.client.requests.post")
    def test_call_skips_notifications_before_the_response(self, mock_post):
        mock_post.return_value = _sse_response(SSE_NOTIFICATIONS_THEN_CALL_RESPONSE)
        result = call_mcp_tool(
            server_url="https://mcp.example.com", tool_name="t", arguments={}
        )
        assert result == "final answer"

    @patch("agentic.mcp.client.requests.post")
    def test_discover_skips_notifications_before_the_response(self, mock_post):
        body = (
            "event: message\n"
            'data: {"jsonrpc": "2.0", "method": "notifications/message", '
            '"params": {"level": "info", "data": "listing"}}\n'
            "\n"
            "event: message\n"
            'data: {"jsonrpc": "2.0", "id": 1, "result": {"tools": '
            '[{"name": "search", "inputSchema": {"type": "object"}}]}}\n'
            "\n"
        )
        mock_post.return_value = _sse_response(body)
        tools = discover_mcp_tools(server_url="https://mcp.example.com")
        assert [t.name for t in tools] == ["search"]

    @patch("agentic.mcp.client.requests.post")
    def test_stream_of_only_notifications_raises(self, mock_post):
        body = (
            "event: message\n"
            'data: {"jsonrpc": "2.0", "method": "notifications/message", '
            '"params": {"level": "info", "data": "nothing else"}}\n'
            "\n"
        )
        mock_post.return_value = _sse_response(body)
        with pytest.raises(McpError) as exc:
            discover_mcp_tools(server_url="https://mcp.example.com")
        assert "no envelope answering request id" in str(exc.value)

    @patch("agentic.mcp.client.requests.post")
    def test_matching_envelope_with_neither_result_nor_error_raises(self, mock_post):
        mock_post.return_value = _sse_response(
            'event: message\ndata: {"jsonrpc": "2.0", "id": 1}\n\n'
        )
        with pytest.raises(McpError) as exc:
            discover_mcp_tools(server_url="https://mcp.example.com")
        assert "neither result nor error" in str(exc.value)

    @patch("agentic.mcp.client.requests.post")
    def test_json_body_with_neither_result_nor_error_raises(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(return_value={"jsonrpc": "2.0", "id": 1}),
        )
        with pytest.raises(McpError) as exc:
            discover_mcp_tools(server_url="https://mcp.example.com")
        assert "neither result nor error" in str(exc.value)

    @patch("agentic.mcp.client.requests.post")
    def test_request_id_less_error_envelope_is_accepted(self, mock_post):
        # JSON-RPC permits id:null on an error when the server could not
        # read the request id.
        mock_post.return_value = _sse_response(
            "event: message\n"
            'data: {"jsonrpc": "2.0", "id": null, "error": '
            '{"code": -32700, "message": "Parse error"}}\n'
            "\n"
        )
        with pytest.raises(McpError) as exc:
            discover_mcp_tools(server_url="https://mcp.example.com")
        assert "Parse error" in str(exc.value)

    @patch("agentic.mcp.client.requests.post")
    def test_multi_line_data_frame_joins_with_newlines(self, mock_post):
        # The SSE spec joins an event's data: lines with \n before dispatch.
        mock_post.return_value = _sse_response(
            "event: message\n"
            'data: {"jsonrpc": "2.0", "id": 1,\n'
            'data:  "result": {"tools": []}}\n'
            "\n"
        )
        tools = discover_mcp_tools(server_url="https://mcp.example.com")
        assert tools == []


class TestSseCharset:
    @patch("agentic.mcp.client.requests.post")
    def test_bare_event_stream_content_type_decodes_as_utf8(self, mock_post):
        # The SSE spec mandates UTF-8. Without this a bare text/event-stream
        # content type makes requests fall back to ISO-8859-1 and mojibake
        # every non-ASCII character.
        body = (
            "event: message\n"
            'data: {"jsonrpc": "2.0", "id": 1, "result": {"tools": '
            '[{"name": "café", "description": "über ✓", '
            '"inputSchema": {"type": "object"}}]}}\n'
            "\n"
        )
        mock_post.return_value = _sse_response(body)  # no charset in header
        tools = discover_mcp_tools(server_url="https://mcp.example.com")
        assert tools[0].name == "café"
        assert tools[0].description == "über ✓"

    @patch("agentic.mcp.client.requests.post")
    def test_non_utf8_sse_stream_raises(self, mock_post):
        resp = requests.Response()
        resp.status_code = 200
        resp.headers["content-type"] = "text/event-stream"
        resp._content = b"event: message\ndata: \xff\xfe\n\n"
        mock_post.return_value = resp
        with pytest.raises(McpError) as exc:
            discover_mcp_tools(server_url="https://mcp.example.com")
        assert "non-UTF-8" in str(exc.value)


class TestRequestShape:
    @patch("agentic.mcp.client.requests.post")
    def test_discover_sends_tools_list_method(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(
                return_value={"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}
            ),
        )
        discover_mcp_tools(server_url="https://mcp.example.com")
        assert mock_post.call_args.kwargs["json"]["method"] == "tools/list"

    @patch("agentic.mcp.client.requests.post")
    def test_timeout_reaches_requests_as_a_keyword(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(
                return_value={"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}
            ),
        )
        discover_mcp_tools(server_url="https://mcp.example.com", timeout=7)
        assert mock_post.call_args.kwargs["timeout"] == 7

    @patch("agentic.mcp.client.requests.post")
    def test_call_timeout_reaches_requests_as_a_keyword(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(
                return_value={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"content": [{"type": "text", "text": "ok"}]},
                }
            ),
        )
        call_mcp_tool(
            server_url="https://mcp.example.com",
            tool_name="t",
            arguments={},
            timeout=9,
        )
        assert mock_post.call_args.kwargs["timeout"] == 9


class TestCallErrorPathRobustness:
    @patch("agentic.mcp.client.requests.post")
    def test_call_survives_error_member_that_is_a_string(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(return_value={"jsonrpc": "2.0", "id": 1, "error": "boom"}),
        )
        result = call_mcp_tool(
            server_url="https://mcp.example.com", tool_name="t", arguments={}
        )
        assert result.startswith("Error")
        assert "has no attribute" not in result
        assert "boom" in result

    @patch("agentic.mcp.client.requests.post")
    def test_call_ignores_non_dict_content_blocks(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(
                return_value={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "content": ["rogue string", {"type": "text", "text": "kept"}]
                    },
                }
            ),
        )
        result = call_mcp_tool(
            server_url="https://mcp.example.com", tool_name="t", arguments={}
        )
        assert result == "kept"

    @patch("agentic.mcp.client.requests.post")
    def test_call_error_return_is_logged(self, mock_post, caplog):
        mock_post.side_effect = Exception("Connection refused")
        with caplog.at_level("WARNING", logger="agentic.mcp.client"):
            result = call_mcp_tool(
                server_url="https://bad.example.com", tool_name="t", arguments={}
            )
        assert result.startswith("Error")
        assert any("Connection refused" in r.message for r in caplog.records)

    @patch("agentic.mcp.client.requests.post")
    def test_call_jsonrpc_error_is_logged(self, mock_post, caplog):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(
                return_value={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "error": {"code": -32600, "message": "Invalid request"},
                }
            ),
        )
        with caplog.at_level("WARNING", logger="agentic.mcp.client"):
            call_mcp_tool(
                server_url="https://mcp.example.com", tool_name="t", arguments={}
            )
        assert any("Invalid request" in r.message for r in caplog.records)


class TestToolExecutionErrors:
    """Per the MCP spec, tool-execution failures are reported as
    result.isError — not as JSON-RPC errors. FastMCP and the official SDK
    report every tool exception this way."""

    @patch("agentic.mcp.client.requests.post")
    def test_is_error_result_returns_an_error_string(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(
                return_value={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "content": [
                            {"type": "text", "text": "the database is on fire"}
                        ],
                        "isError": True,
                    },
                }
            ),
        )
        result = call_mcp_tool(
            server_url="https://mcp.example.com", tool_name="t", arguments={}
        )
        assert result.startswith("Error")
        assert "the database is on fire" in result

    @patch("agentic.mcp.client.requests.post")
    def test_is_error_with_empty_content_is_not_a_success(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(
                return_value={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"content": [], "isError": True},
                }
            ),
        )
        result = call_mcp_tool(
            server_url="https://mcp.example.com", tool_name="t", arguments={}
        )
        assert result.startswith("Error")
        assert result != "(empty response)"

    @patch("agentic.mcp.client.requests.post")
    def test_is_error_false_stays_a_success(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(
                return_value={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "content": [{"type": "text", "text": "fine"}],
                        "isError": False,
                    },
                }
            ),
        )
        result = call_mcp_tool(
            server_url="https://mcp.example.com", tool_name="t", arguments={}
        )
        assert result == "fine"


class TestLazyFrameDecoding:
    """A frame after the answer must not destroy a result that already
    arrived — a truncated trailing frame is what a dropped connection
    looks like."""

    @patch("agentic.mcp.client.requests.post")
    def test_truncated_frame_after_the_answer_is_ignored(self, mock_post):
        mock_post.return_value = _sse_response(
            "event: message\n"
            'data: {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}\n'
            "\n"
            "event: message\n"
            'data: {"jsonrpc": "2.0", "method": "notif\n'
            "\n"
        )
        assert discover_mcp_tools(server_url="https://mcp.example.com") == []

    @patch("agentic.mcp.client.requests.post")
    def test_junk_frame_before_the_answer_is_skipped(self, mock_post):
        mock_post.return_value = _sse_response(
            "event: message\n"
            "data: not-json\n"
            "\n"
            "event: message\n"
            'data: {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}\n'
            "\n"
        )
        assert discover_mcp_tools(server_url="https://mcp.example.com") == []

    @patch("agentic.mcp.client.requests.post")
    def test_junk_only_stream_still_reports_the_parse_failure(self, mock_post):
        mock_post.return_value = _sse_response("event: message\ndata: not-json\n\n")
        with pytest.raises(McpError) as exc:
            discover_mcp_tools(server_url="https://mcp.example.com")
        assert "unparseable SSE data frame" in str(exc.value)

    @patch("agentic.mcp.client.requests.post")
    def test_utf8_bom_is_tolerated(self, mock_post):
        resp = requests.Response()
        resp.status_code = 200
        resp.headers["content-type"] = "text/event-stream"
        resp._content = "﻿".encode() + (
            b'event: message\ndata: {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}\n\n'
        )
        mock_post.return_value = resp
        assert discover_mcp_tools(server_url="https://mcp.example.com") == []


class TestIdCorrelationRejects:
    """Pin the predicate in the rejecting direction — a matcher replaced
    with `True` must fail these."""

    @patch("agentic.mcp.client.requests.post")
    def test_wrong_id_before_the_right_one_is_skipped(self, mock_post):
        mock_post.return_value = _sse_response(
            "event: message\n"
            'data: {"jsonrpc": "2.0", "id": 99, "result": {"tools": '
            '[{"name": "wrong", "inputSchema": {"type": "object"}}]}}\n'
            "\n"
            "event: message\n"
            'data: {"jsonrpc": "2.0", "id": 1, "result": {"tools": '
            '[{"name": "right", "inputSchema": {"type": "object"}}]}}\n'
            "\n"
        )
        tools = discover_mcp_tools(server_url="https://mcp.example.com")
        assert [t.name for t in tools] == ["right"]

    @patch("agentic.mcp.client.requests.post")
    def test_lone_wrong_id_envelope_raises(self, mock_post):
        mock_post.return_value = _sse_response(
            "event: message\n"
            'data: {"jsonrpc": "2.0", "id": 99, "result": {"tools": []}}\n'
            "\n"
        )
        with pytest.raises(McpError) as exc:
            discover_mcp_tools(server_url="https://mcp.example.com")
        assert "no envelope answering request id" in str(exc.value)

    @patch("agentic.mcp.client.requests.post")
    def test_exact_id_match_beats_an_earlier_idless_error(self, mock_post):
        # An id:null error is a fallback for when the server could not read
        # our id — it must not preempt a real answer later in the stream.
        mock_post.return_value = _sse_response(
            "event: message\n"
            'data: {"jsonrpc": "2.0", "id": null, "error": '
            '{"code": -32700, "message": "Parse error"}}\n'
            "\n"
            "event: message\n"
            'data: {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}\n'
            "\n"
        )
        assert discover_mcp_tools(server_url="https://mcp.example.com") == []


class TestNullToolFields:
    """Present-but-null fields on a tool entry must not poison discovery
    or flow through to the LLM provider as nulls."""

    @patch("agentic.mcp.client.requests.post")
    def test_null_annotations_on_one_tool_does_not_kill_the_list(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(
                return_value={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "tools": [
                            {
                                "name": "t1",
                                "inputSchema": {"type": "object"},
                                "annotations": None,
                            }
                        ]
                    },
                }
            ),
        )
        tools = discover_mcp_tools(server_url="https://mcp.example.com")
        assert tools[0].read_only_hint is False
        assert tools[0].destructive_hint is False

    @patch("agentic.mcp.client.requests.post")
    def test_null_description_and_schema_get_defaults(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(
                return_value={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "tools": [
                            {"name": "t1", "description": None, "inputSchema": None}
                        ]
                    },
                }
            ),
        )
        tools = discover_mcp_tools(server_url="https://mcp.example.com")
        assert tools[0].description == ""
        assert tools[0].input_schema == {"type": "object"}

    @pytest.mark.parametrize("bad_name", [None, "", 7])
    @patch("agentic.mcp.client.requests.post")
    def test_invalid_name_raises(self, mock_post, bad_name):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(
                return_value={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "tools": [{"name": bad_name, "inputSchema": {"type": "object"}}]
                    },
                }
            ),
        )
        with pytest.raises(McpError):
            discover_mcp_tools(server_url="https://mcp.example.com")


class TestErrorDetails:
    @patch("agentic.mcp.client.requests.post")
    def test_discover_error_includes_the_jsonrpc_code(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(
                return_value={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "error": {"code": -32601, "message": "Method not found"},
                }
            ),
        )
        with pytest.raises(McpError) as exc:
            discover_mcp_tools(server_url="https://mcp.example.com")
        assert "-32601" in str(exc.value)

    @patch("agentic.mcp.client.requests.post")
    def test_call_error_includes_the_jsonrpc_code(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(
                return_value={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "error": {"code": -32602, "message": "Invalid params"},
                }
            ),
        )
        result = call_mcp_tool(
            server_url="https://mcp.example.com", tool_name="t", arguments={}
        )
        assert "-32602" in result

    @patch("agentic.mcp.client.requests.post")
    def test_non_mapping_headers_message_names_the_headers(self, mock_post):
        with pytest.raises(McpError) as exc:
            discover_mcp_tools(
                server_url="https://mcp.example.com", server_headers="abc"
            )
        assert "headers" in str(exc.value).lower()
        mock_post.assert_not_called()


class TestCharsetAndTimeoutDefaults:
    @patch("agentic.mcp.client.requests.post")
    def test_explicit_utf8_charset_also_decodes(self, mock_post):
        body = (
            "event: message\n"
            'data: {"jsonrpc": "2.0", "id": 1, "result": {"tools": '
            '[{"name": "café", "inputSchema": {"type": "object"}}]}}\n'
            "\n"
        )
        mock_post.return_value = _sse_response(body, charset="utf-8")
        tools = discover_mcp_tools(server_url="https://mcp.example.com")
        assert tools[0].name == "café"

    @patch("agentic.mcp.client.requests.post")
    def test_default_timeout_reaches_requests(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(
                return_value={"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}
            ),
        )
        discover_mcp_tools(server_url="https://mcp.example.com")
        assert mock_post.call_args.kwargs["timeout"] == 30


class TestNonMappingServerHeaders:
    """`server_headers` that isn't a mapping — reachable from an unvalidated API field."""

    @pytest.mark.parametrize("bad_headers", [["a", "b"], "abc", 7])
    @patch("agentic.mcp.client.requests.post")
    def test_discover_raises_on_non_mapping_headers(self, mock_post, bad_headers):
        with pytest.raises(McpError):
            discover_mcp_tools(
                server_url="https://mcp.example.com", server_headers=bad_headers
            )
        mock_post.assert_not_called()
