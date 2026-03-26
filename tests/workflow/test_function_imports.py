"""Tests for the FunctionBlock import registry and dynamic loader."""

import pytest

from unittest.mock import patch

from agentic.workflow.blocks.function import (
    IMPORT_REGISTRY,
    LEGACY_DEFAULT_IMPORTS,
    _PIP_TO_IMPORT,
    _load_custom_packages,
    _load_imports,
    _parse_package_specs,
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


class TestCustomPackages:
    def test_parse_package_specs_simple(self):
        specs = _parse_package_specs("json, math, requests")
        assert specs == [("json", "json"), ("math", "math"), ("requests", "requests")]

    def test_parse_package_specs_with_version(self):
        specs = _parse_package_specs("openai>=1.0, tiktoken==0.5.1")
        assert specs == [
            ("openai", "openai>=1.0"),
            ("tiktoken", "tiktoken==0.5.1"),
        ]

    def test_parse_package_specs_empty(self):
        assert _parse_package_specs("") == []
        assert _parse_package_specs("   ") == []
        assert _parse_package_specs(",,,") == []

    def test_load_custom_packages_preinstalled(self):
        result = _load_custom_packages("json")
        import json

        assert "json" in result
        assert result["json"] is json

    def test_load_custom_packages_unknown(self):
        with patch(
            "agentic.workflow.blocks.function._pip_install", return_value=False
        ):
            result = _load_custom_packages("totally_nonexistent_pkg_xyz")
        assert "totally_nonexistent_pkg_xyz" not in result

    def test_pip_to_import_mapping(self):
        assert _PIP_TO_IMPORT["Pillow"] == "PIL"
        assert _PIP_TO_IMPORT["scikit-learn"] == "sklearn"
        assert _PIP_TO_IMPORT["pyyaml"] == "yaml"
        assert _PIP_TO_IMPORT["beautifulsoup4"] == "bs4"
        assert _PIP_TO_IMPORT["opencv-python"] == "cv2"
        assert _PIP_TO_IMPORT["python-dateutil"] == "dateutil"

    def test_parse_package_specs_hyphenated(self):
        specs = _parse_package_specs("my-custom-pkg")
        assert specs == [("my_custom_pkg", "my-custom-pkg")]

    def test_pip_install_called_on_missing(self):
        with patch(
            "agentic.workflow.blocks.function._pip_install", return_value=False
        ) as mock_pip:
            _load_custom_packages("some_missing_pkg")
        mock_pip.assert_called_once_with("some_missing_pkg")

    @pytest.mark.asyncio
    async def test_execute_with_custom_packages(self):
        from agentic.workflow.block import BlockInput
        from agentic.workflow.blocks.function import FunctionBlock

        block = FunctionBlock(
            config={
                "code": 'output = json.dumps({"key": "value"})',
                "imports": [],
                "custom_packages": "json",
            },
        )
        result = await block.execute(BlockInput(block_outputs={}))
        assert result.data["output"] == '{"key": "value"}'

    @pytest.mark.asyncio
    async def test_execute_custom_packages_wrong_type(self):
        from agentic.workflow.block import BlockInput
        from agentic.workflow.blocks.function import FunctionBlock

        block = FunctionBlock(
            config={
                "code": "output = 42",
                "imports": [],
                "custom_packages": ["json"],  # list instead of string
            },
        )
        result = await block.execute(BlockInput(block_outputs={}))
        # Should not crash — falls back gracefully
        assert result.data["output"] == 42

    @pytest.mark.asyncio
    async def test_execute_with_both_imports_and_custom(self):
        from agentic.workflow.block import BlockInput
        from agentic.workflow.blocks.function import FunctionBlock

        block = FunctionBlock(
            config={
                "code": "output = json.dumps({'pi': round(math.pi, 2)})",
                "imports": ["math"],
                "custom_packages": "json",
            },
        )
        result = await block.execute(BlockInput(block_outputs={}))
        assert result.data["output"] == '{"pi": 3.14}'
