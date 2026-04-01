import pytest

from agentic.agent.tools import (
    BuiltinTool,
    CustomTool,
    DelegateTool,
    KnowledgeSearchTool,
    ToolDefinition,
)


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


class TestKnowledgeSearchTool:
    def test_create(self):
        from agentic.agent.tools import KnowledgeSearchTool

        tool = KnowledgeSearchTool(
            name="search_policies",
            description="Search the policies knowledge base",
            knowledge_base_configs=[{"id": "kb-123", "top_k": 5}],
            max_context_tokens=8000,
        )
        assert tool.name == "search_policies"
        assert tool.knowledge_base_configs == [{"id": "kb-123", "top_k": 5}]
        assert tool.max_context_tokens == 8000
        assert tool.input_schema["type"] == "object"
        assert "query" in tool.input_schema["properties"]

    def test_execute_requires_handler(self):
        from agentic.agent.tools import KnowledgeSearchTool

        tool = KnowledgeSearchTool(
            name="search_test",
            description="test",
            knowledge_base_configs=[{"id": "kb-1"}],
        )
        with pytest.raises(NotImplementedError):
            tool.execute({"query": "test"}, None)

    def test_execute_with_handler(self):
        from agentic.agent.tools import KnowledgeSearchTool

        def mock_handler(query, kb_configs, max_tokens, session_history):
            return f"Results for: {query}"

        tool = KnowledgeSearchTool(
            name="search_test",
            description="test",
            knowledge_base_configs=[{"id": "kb-1"}],
            search_handler=mock_handler,
        )
        result = tool.execute({"query": "hello"}, None)
        assert result == "Results for: hello"

    def test_to_function_schema(self):
        from agentic.agent.tools import KnowledgeSearchTool

        tool = KnowledgeSearchTool(
            name="search_docs",
            description="Search documentation",
            knowledge_base_configs=[{"id": "kb-1"}],
        )
        schema = tool.to_function_schema()
        assert schema["function"]["name"] == "search_docs"
        assert "query" in schema["function"]["parameters"]["properties"]

    def test_combined_tool_has_kb_names_param(self):
        from agentic.agent.tools import KnowledgeSearchTool

        tool = KnowledgeSearchTool(
            name="knowledge_search",
            description="Search across KBs",
            knowledge_base_configs=[
                {"id": "kb-1", "name": "policies"},
                {"id": "kb-2", "name": "claims"},
            ],
            include_kb_filter=True,
        )
        schema = tool.to_function_schema()
        params = schema["function"]["parameters"]
        assert "knowledge_base_names" in params["properties"]
        assert params["properties"]["knowledge_base_names"]["type"] == "array"


class TestToolMetadataFlags:
    def test_tool_definition_defaults_fail_closed(self):
        tool = ToolDefinition(name="t", description="", input_schema={})
        assert tool.is_concurrency_safe is False
        assert tool.is_read_only is False
        assert tool.is_destructive is False
        assert tool.max_result_chars == 50000

    def test_builtin_inherits_defaults(self):
        tool = BuiltinTool(
            name="t", description="", input_schema={}, handler=lambda a, c: ""
        )
        assert tool.is_concurrency_safe is False

    def test_builtin_override_flags(self):
        tool = BuiltinTool(
            name="db",
            description="",
            input_schema={},
            handler=lambda a, c: "",
            is_concurrency_safe=True,
            is_read_only=True,
        )
        assert tool.is_concurrency_safe is True
        assert tool.is_read_only is True

    def test_custom_tool_override_flags(self):
        tool = CustomTool(
            name="api",
            description="",
            input_schema={},
            endpoint="",
            is_concurrency_safe=True,
            is_destructive=True,
        )
        assert tool.is_concurrency_safe is True
        assert tool.is_destructive is True

    def test_knowledge_search_defaults(self):
        tool = KnowledgeSearchTool(name="search", description="")
        assert tool.is_concurrency_safe is True
        assert tool.is_read_only is True
        assert tool.max_result_chars is None

    def test_delegate_defaults(self):
        from unittest.mock import MagicMock

        tool = DelegateTool(name="d", description="", agent=MagicMock())
        assert tool.is_concurrency_safe is False
        assert tool.max_result_chars is None

    def test_max_result_chars_on_builtin(self):
        tool = BuiltinTool(
            name="t",
            description="",
            input_schema={},
            handler=lambda a, c: "",
            max_result_chars=1000,
        )
        assert tool.max_result_chars == 1000
