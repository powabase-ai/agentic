import pytest

from agentic.agent.tools import BuiltinTool, CustomTool, ToolDefinition


class TestToolDefinition:
    def test_tool_definition_is_abstract(self):
        tool = ToolDefinition(
            name="test",
            description="A test tool",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        with pytest.raises(NotImplementedError):
            tool.execute({"q": "hello"}, None)

    def test_tool_to_function_schema(self):
        tool = ToolDefinition(
            name="search",
            description="Search for things",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        schema = tool.to_function_schema()
        assert schema == {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search for things",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }


class TestBuiltinTool:
    def test_execute_calls_handler(self):
        def handler(args, ctx):
            return f"result: {args['x']}"

        tool = BuiltinTool(
            name="my_tool",
            description="does stuff",
            input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
            handler=handler,
        )
        result = tool.execute({"x": "hello"}, None)
        assert result == "result: hello"

    def test_execute_handler_error_propagates(self):
        def bad_handler(args, ctx):
            raise ValueError("boom")

        tool = BuiltinTool(
            name="bad",
            description="fails",
            input_schema={"type": "object"},
            handler=bad_handler,
        )
        with pytest.raises(ValueError, match="boom"):
            tool.execute({}, None)


class TestCustomTool:
    def test_execute_calls_endpoint(self, mocker):
        mock_response = mocker.MagicMock()
        mock_response.text = '{"answer": 42}'
        mock_response.raise_for_status = mocker.MagicMock()
        mock_request = mocker.patch(
            "agentic.agent.tools.requests.request", return_value=mock_response
        )

        tool = CustomTool(
            name="api_tool",
            description="calls an API",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
            endpoint="https://example.com/api",
            method="POST",
            headers={"Authorization": "Bearer token"},
            timeout=10,
        )
        result = tool.execute({"q": "test"}, None)

        assert result == '{"answer": 42}'
        mock_request.assert_called_once_with(
            "POST",
            "https://example.com/api",
            json={"q": "test"},
            headers={"Authorization": "Bearer token"},
            timeout=10,
        )

    def test_execute_truncates_large_response(self, mocker):
        mock_response = mocker.MagicMock()
        mock_response.text = "x" * 15000
        mock_response.raise_for_status = mocker.MagicMock()
        mocker.patch("agentic.agent.tools.requests.request", return_value=mock_response)

        tool = CustomTool(
            name="big",
            description="big response",
            input_schema={"type": "object"},
            endpoint="https://example.com",
            method="GET",
            headers={},
            timeout=30,
        )
        result = tool.execute({}, None)
        assert len(result) == 10000
