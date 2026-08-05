"""Live MCP server tests.

Opt-in: skipped unless MCP_INTEGRATION_URL names a reachable MCP server.

    MCP_INTEGRATION_URL=https://your-mcp-server uv run pytest \\
        tests/test_mcp_client_integration.py -v

Assertions are behavioural — a tool list is non-empty and well-formed — never
a tool count or a specific tool name, both of which change as a server evolves.
"""

import os

import pytest

from agentic.mcp import call_mcp_tool, discover_mcp_tools

MCP_URL = os.environ.get("MCP_INTEGRATION_URL")

pytestmark = pytest.mark.skipif(
    not MCP_URL,
    reason="set MCP_INTEGRATION_URL to run live MCP server tests",
)


def test_discover_returns_well_formed_tools():
    tools = discover_mcp_tools(MCP_URL)
    assert len(tools) >= 1
    for tool in tools:
        assert tool.name
        assert isinstance(tool.input_schema, dict)


def test_call_a_tool_that_needs_no_arguments():
    tools = discover_mcp_tools(MCP_URL)
    callable_without_args = [
        t for t in tools if not (t.input_schema or {}).get("required")
    ]
    if not callable_without_args:
        pytest.skip("server exposes no tool callable without arguments")

    result = call_mcp_tool(
        MCP_URL, tool_name=callable_without_args[0].name, arguments={}
    )
    assert isinstance(result, str)
    assert result
    assert not result.startswith("Error")
