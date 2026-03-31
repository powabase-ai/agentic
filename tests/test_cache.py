from agentic.agent.cache import sort_tools_for_cache
from agentic.agent.tools import (
    BuiltinTool,
    CustomTool,
    DelegateTool,
    KnowledgeSearchTool,
)


class TestSortToolsForCache:
    def test_builtins_before_custom(self):
        custom = CustomTool(
            name="z_custom", description="", input_schema={}, endpoint=""
        )
        builtin = BuiltinTool(
            name="a_builtin", description="", input_schema={}, handler=lambda a, c: ""
        )
        result = sort_tools_for_cache({"z_custom": custom, "a_builtin": builtin})
        assert result[0].name == "a_builtin"
        assert result[1].name == "z_custom"

    def test_alphabetical_within_type(self):
        b1 = BuiltinTool(
            name="zebra", description="", input_schema={}, handler=lambda a, c: ""
        )
        b2 = BuiltinTool(
            name="alpha", description="", input_schema={}, handler=lambda a, c: ""
        )
        result = sort_tools_for_cache({"zebra": b1, "alpha": b2})
        assert result[0].name == "alpha"
        assert result[1].name == "zebra"

    def test_full_priority_order(self):
        from unittest.mock import MagicMock

        builtin = BuiltinTool(
            name="db", description="", input_schema={}, handler=lambda a, c: ""
        )
        kb = KnowledgeSearchTool(name="search_docs", description="")
        custom = CustomTool(name="api", description="", input_schema={}, endpoint="")
        delegate = DelegateTool(name="delegate_to_x", description="", agent=MagicMock())
        tools = {
            "delegate_to_x": delegate,
            "api": custom,
            "search_docs": kb,
            "db": builtin,
        }
        result = sort_tools_for_cache(tools)
        names = [t.name for t in result]
        assert names == ["db", "search_docs", "api", "delegate_to_x"]

    def test_deterministic_across_calls(self):
        b1 = BuiltinTool(
            name="a", description="", input_schema={}, handler=lambda a, c: ""
        )
        b2 = BuiltinTool(
            name="b", description="", input_schema={}, handler=lambda a, c: ""
        )
        tools = {"b": b2, "a": b1}
        r1 = [t.name for t in sort_tools_for_cache(tools)]
        r2 = [t.name for t in sort_tools_for_cache(tools)]
        assert r1 == r2

    def test_empty_tools(self):
        assert sort_tools_for_cache({}) == []
