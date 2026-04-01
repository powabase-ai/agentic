"""Tests for McpTool class — TDD for R5 Task 2."""

from unittest.mock import MagicMock, patch

from agentic.agent.tools import McpTool


class TestMcpTool:
    def test_create(self):
        tool = McpTool(
            name="mcp__github__create_issue",
            description="Create a GitHub issue",
            server_name="github",
            server_url="https://mcp.example.com",
            mcp_tool_name="create_issue",
        )
        assert tool.name == "mcp__github__create_issue"
        assert tool.server_name == "github"
        assert tool.mcp_tool_name == "create_issue"

    def test_default_flags(self):
        tool = McpTool(
            name="t",
            description="",
            server_name="s",
            server_url="http://x",
            mcp_tool_name="t",
        )
        assert tool.is_concurrency_safe is False
        assert tool.is_read_only is False
        assert tool.is_destructive is False
        assert tool.max_result_chars == 50000

    def test_read_only_hint_maps_to_flags(self):
        tool = McpTool(
            name="t",
            description="",
            server_name="s",
            server_url="http://x",
            mcp_tool_name="t",
            is_concurrency_safe=True,
            is_read_only=True,
        )
        assert tool.is_concurrency_safe is True
        assert tool.is_read_only is True

    def test_to_function_schema(self):
        tool = McpTool(
            name="mcp__github__list_repos",
            description="List repos",
            input_schema={"type": "object", "properties": {"org": {"type": "string"}}},
            server_name="github",
            server_url="http://x",
            mcp_tool_name="list_repos",
        )
        schema = tool.to_function_schema()
        assert schema["function"]["name"] == "mcp__github__list_repos"
        assert "org" in schema["function"]["parameters"]["properties"]

    @patch("agentic.mcp.client.requests.post")
    def test_execute_calls_mcp_server(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"content": [{"type": "text", "text": "Issue created"}]},
                }
            ),
        )
        tool = McpTool(
            name="mcp__github__create_issue",
            description="Create issue",
            server_name="github",
            server_url="https://mcp.example.com",
            server_headers={"Authorization": "Bearer token"},
            mcp_tool_name="create_issue",
        )
        result = tool.execute({"title": "Bug"}, None)
        assert result == "Issue created"
        mock_post.assert_called_once()
