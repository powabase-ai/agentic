from unittest.mock import MagicMock, patch

from agentic.mcp.client import call_mcp_tool, discover_mcp_tools


class TestDiscoverMcpTools:
    @patch("agentic.mcp.client.requests.post")
    def test_discover_returns_tools(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
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
    def test_discover_handles_error(self, mock_post):
        mock_post.side_effect = Exception("Connection refused")
        tools = discover_mcp_tools(server_url="https://bad.example.com")
        assert tools == []


class TestCallMcpTool:
    @patch("agentic.mcp.client.requests.post")
    def test_call_returns_text_result(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
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
