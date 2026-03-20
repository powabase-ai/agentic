"""Tests for the FunctionBlock import registry and dynamic loader."""

import pytest

from agentic.workflow.blocks.function import (
    IMPORT_REGISTRY,
    LEGACY_DEFAULT_IMPORTS,
    _load_imports,
)


class TestLoadImports:
    def test_stdlib_import(self):
        result = _load_imports(["json"])
        assert "json" in result
        import json

        assert result["json"] is json

    def test_aliased_import(self):
        result = _load_imports(["numpy"])
        assert "numpy" in result
        assert "np" in result
        assert result["numpy"] is result["np"]

    def test_multiple_imports(self):
        result = _load_imports(["json", "numpy"])
        assert "json" in result
        assert "numpy" in result
        assert "np" in result

    def test_unknown_key_ignored(self):
        result = _load_imports(["unknown_package"])
        assert result == {}

    def test_empty_list(self):
        result = _load_imports([])
        assert result == {}

    def test_no_alias_import(self):
        result = _load_imports(["math"])
        assert "math" in result
        # math has no alias, so only the module name should be present
        import math

        assert result["math"] is math

    def test_legacy_defaults_all_in_registry(self):
        for key in LEGACY_DEFAULT_IMPORTS:
            assert key in IMPORT_REGISTRY


class TestFunctionBlockImports:
    @pytest.mark.asyncio
    async def test_execute_with_selected_imports(self):
        from agentic.workflow.block import BlockInput
        from agentic.workflow.blocks.function import FunctionBlock

        block = FunctionBlock(
            config={
                "code": 'output = json.dumps({"a": 1})',
                "imports": ["json"],
            },
        )
        result = await block.execute(BlockInput(block_outputs={}))
        assert result.data["output"] == '{"a": 1}'

    @pytest.mark.asyncio
    async def test_execute_with_no_imports_key_uses_legacy(self):
        from agentic.workflow.block import BlockInput
        from agentic.workflow.blocks.function import FunctionBlock

        block = FunctionBlock(
            config={
                "code": "output = type(np).__name__",
            },
        )
        result = await block.execute(BlockInput(block_outputs={}))
        assert result.data["output"] == "module"

    @pytest.mark.asyncio
    async def test_execute_with_empty_imports(self):
        from agentic.workflow.block import BlockInput
        from agentic.workflow.blocks.function import FunctionBlock

        block = FunctionBlock(
            config={
                "code": "output = 42",
                "imports": [],
            },
        )
        result = await block.execute(BlockInput(block_outputs={}))
        assert result.data["output"] == 42
