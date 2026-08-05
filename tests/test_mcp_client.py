from unittest.mock import MagicMock, patch

import pytest

from agentic.mcp.client import McpError, call_mcp_tool, discover_mcp_tools


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
                        ]
                    },
                }
            ),
        )

        tools = discover_mcp_tools(
            server_url="https://mcp.example.com",
            server_headers={"Authorization": "Bearer token"},
        )

        assert len(tools) == 2
        assert tools[0].name == "create_issue"
        assert tools[0].description == "Create a GitHub issue"
        assert tools[0].input_schema["properties"]["title"]["type"] == "string"
        assert tools[0].read_only_hint is False
        assert tools[1].name == "list_repos"
        assert tools[1].read_only_hint is True

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
        assert "error" in result.lower() or "Invalid" in result

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
        call_mcp_tool(
            server_url="https://mcp.example.com", tool_name="t", arguments={}
        )
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
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            text=SSE_TOOLS_BODY,
        )
        tools = discover_mcp_tools(server_url="https://mcp.example.com")
        assert len(tools) == 1
        assert tools[0].name == "search"
        assert tools[0].input_schema == {"type": "object"}

    @patch("agentic.mcp.client.requests.post")
    def test_call_parses_sse_body(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            text=SSE_CALL_BODY,
        )
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
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            text="event: message\n\n",
        )
        with pytest.raises(McpError) as exc:
            discover_mcp_tools(server_url="https://mcp.example.com")
        assert "no data frame" in str(exc.value)

    @patch("agentic.mcp.client.requests.post")
    def test_discover_raises_on_unparseable_sse_data_frame(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            text="event: message\ndata: not-json\n\n",
        )
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
            json=MagicMock(
                return_value={"jsonrpc": "2.0", "id": 1, "error": "boom"}
            ),
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
