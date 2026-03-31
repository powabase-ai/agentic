"""Tests for the FunctionBlock import registry and dynamic loader."""

import pytest

from unittest.mock import patch

from agentic.workflow.blocks.function import (
    IMPORT_REGISTRY,
    LEGACY_DEFAULT_IMPORTS,
    _PIP_TO_IMPORT,
    _load_custom_packages,
    _load_imports,
    _parse_imports_from_code,
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
    async def test_execute_with_no_imports_key_parses_code(self):
        """When no 'imports' key is present, imports are parsed from the code."""
        from agentic.workflow.block import BlockInput
        from agentic.workflow.blocks.function import FunctionBlock

        block = FunctionBlock(
            config={
                "code": "import numpy as np\noutput = type(np).__name__",
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


class TestParseImportsFromCode:
    def test_simple_import(self):
        registry, custom = _parse_imports_from_code("import json\nimport math")
        assert "json" in registry
        assert "math" in registry
        assert custom == []

    def test_from_import(self):
        registry, custom = _parse_imports_from_code("from datetime import timedelta")
        assert "datetime" in registry
        assert custom == []

    def test_aliased_import(self):
        registry, custom = _parse_imports_from_code("import numpy as np")
        assert "numpy" in registry
        assert custom == []

    def test_multi_import(self):
        registry, custom = _parse_imports_from_code("import json, math, re")
        assert "json" in registry
        assert "math" in registry
        assert "re" in registry

    def test_custom_module(self):
        registry, custom = _parse_imports_from_code("import openai")
        assert registry == []
        assert "openai" in custom

    def test_mixed_registry_and_custom(self):
        code = "import json\nimport openai\nfrom datetime import date"
        registry, custom = _parse_imports_from_code(code)
        assert "json" in registry
        assert "datetime" in registry
        assert "openai" in custom

    def test_dotted_import_uses_root(self):
        registry, custom = _parse_imports_from_code("from urllib.parse import urlparse")
        assert "urllib" in registry
        assert custom == []

    def test_empty_code(self):
        registry, custom = _parse_imports_from_code("")
        assert registry == []
        assert custom == []

    def test_no_imports(self):
        registry, custom = _parse_imports_from_code("output = 42")
        assert registry == []
        assert custom == []

    def test_import_in_comment_ignored(self):
        # Regex only matches lines starting with import/from (possibly indented)
        # A comment like "# import os" won't match because of the #
        registry, custom = _parse_imports_from_code("# import os\noutput = 1")
        assert registry == []
        assert custom == []


class TestFunctionBlockInputDataIsolation:
    @pytest.mark.asyncio
    async def test_input_data_is_deep_copy_of_block_outputs(self):
        """input_data in the sandbox must be a deep copy, not a reference
        to block_outputs, to prevent circular references when user code
        stores input_data in its output."""
        from agentic.workflow.block import BlockInput
        from agentic.workflow.blocks.function import FunctionBlock

        original_outputs = {"b1": {"output": "hello"}}
        block = FunctionBlock(
            config={
                "code": (
                    "# Store input_data in output — should NOT create circular ref\n"
                    "output = {'received': input_data}\n"
                ),
            },
        )
        block_input = BlockInput(block_outputs=original_outputs)
        result = await block.execute(block_input)

        received = result.data["output"]["received"]
        # Must contain the same data
        assert received == original_outputs
        # Must NOT be the same object (deep copy)
        assert received is not original_outputs
        assert received["b1"] is not original_outputs["b1"]


class TestCodeParsedImportExecution:
    @pytest.mark.asyncio
    async def test_import_json_in_code(self):
        from agentic.workflow.block import BlockInput
        from agentic.workflow.blocks.function import FunctionBlock

        block = FunctionBlock(
            config={
                "code": 'import json\noutput = json.dumps({"a": 1})',
            },
        )
        result = await block.execute(BlockInput(block_outputs={}))
        assert result.data["output"] == '{"a": 1}'

    @pytest.mark.asyncio
    async def test_from_datetime_import(self):
        from agentic.workflow.block import BlockInput
        from agentic.workflow.blocks.function import FunctionBlock

        block = FunctionBlock(
            config={
                "code": "from datetime import date\noutput = str(date(2024, 1, 1))",
            },
        )
        result = await block.execute(BlockInput(block_outputs={}))
        assert result.data["output"] == "2024-01-01"

    @pytest.mark.asyncio
    async def test_import_numpy_as_np(self):
        from agentic.workflow.block import BlockInput
        from agentic.workflow.blocks.function import FunctionBlock

        block = FunctionBlock(
            config={
                "code": "import numpy as np\noutput = float(np.mean([1, 2, 3]))",
            },
        )
        result = await block.execute(BlockInput(block_outputs={}))
        assert result.data["output"] == 2.0

    @pytest.mark.asyncio
    async def test_import_requests_preinstalled(self):
        from agentic.workflow.block import BlockInput
        from agentic.workflow.blocks.function import FunctionBlock

        block = FunctionBlock(
            config={
                "code": "import requests\noutput = type(requests).__name__",
            },
        )
        result = await block.execute(BlockInput(block_outputs={}))
        assert result.data["output"] == "module"

    @pytest.mark.asyncio
    async def test_disallowed_import_blocked(self):
        """Importing a module not in the code's import statements is blocked."""
        from agentic.workflow.block import BlockInput
        from agentic.workflow.blocks.function import FunctionBlock

        block = FunctionBlock(
            config={
                # Code uses subprocess but doesn't import it at top level
                "code": "import json\nx = __import__('subprocess')\noutput = 1",
            },
        )
        result = await block.execute(BlockInput(block_outputs={}))
        assert result.status == "error"
        assert "not allowed" in result.error

    @pytest.mark.asyncio
    async def test_backward_compat_with_imports_key(self):
        """Old workflows with explicit 'imports' key still work."""
        from agentic.workflow.block import BlockInput
        from agentic.workflow.blocks.function import FunctionBlock

        block = FunctionBlock(
            config={
                "code": 'output = json.dumps({"x": 1})',
                "imports": ["json"],
            },
        )
        result = await block.execute(BlockInput(block_outputs={}))
        assert result.data["output"] == '{"x": 1}'

    @pytest.mark.asyncio
    async def test_custom_module_pip_installed(self):
        """Custom modules not in registry trigger pip install."""
        from agentic.workflow.block import BlockInput
        from agentic.workflow.blocks.function import FunctionBlock

        block = FunctionBlock(
            config={
                "code": "import some_fake_pkg\noutput = 1",
            },
        )
        with patch(
            "agentic.workflow.blocks.function._pip_install", return_value=False
        ) as mock_pip:
            result = await block.execute(BlockInput(block_outputs={}))
        mock_pip.assert_called_once_with("some_fake_pkg")

    @pytest.mark.asyncio
    async def test_multiple_imports_in_code(self):
        from agentic.workflow.block import BlockInput
        from agentic.workflow.blocks.function import FunctionBlock

        block = FunctionBlock(
            config={
                "code": (
                    "import json\n"
                    "import math\n"
                    "output = json.dumps({'pi': round(math.pi, 2)})"
                ),
            },
        )
        result = await block.execute(BlockInput(block_outputs={}))
        assert result.data["output"] == '{"pi": 3.14}'
