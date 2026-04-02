"""Tests for the FunctionBlock import registry and dynamic loader."""

import pytest

from agentic.workflow.blocks.function import (
    _PIP_TO_IMPORT,
    IMPORT_REGISTRY,
    LEGACY_DEFAULT_IMPORTS,
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
                "code": "import numpy as np\noutput = str(np.array([1,2,3]))",
            },
        )
        result = await block.execute(BlockInput(block_outputs={}))
        assert result.data["output"] == "[1 2 3]"

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

    def test_pip_install_not_called_on_missing(self):
        """pip install is disabled — missing packages are just skipped."""
        result = _load_custom_packages("some_missing_pkg")
        assert "some_missing_pkg" not in result

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
                "code": "import requests\noutput = str(requests.codes.ok)",
            },
        )
        result = await block.execute(BlockInput(block_outputs={}))
        assert result.data["output"] == "200"

    @pytest.mark.asyncio
    async def test_undeclared_import_blocked_at_runtime(self):
        """With explicit imports config, only declared modules can be imported
        at runtime via the controlled import mechanism."""
        from agentic.workflow.block import BlockInput
        from agentic.workflow.blocks.function import FunctionBlock

        # Old-path: explicit imports=["json"]. Code tries to also import os.
        block = FunctionBlock(
            config={
                "code": (
                    "import os\n"
                    "output = 'escaped'"
                ),
                "imports": ["json"],
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
    async def test_custom_module_no_pip_install(self):
        """Custom modules not in registry are NOT pip-installed (disabled)."""
        from agentic.workflow.block import BlockInput
        from agentic.workflow.blocks.function import FunctionBlock

        block = FunctionBlock(
            config={
                "code": "import some_fake_pkg\noutput = 1",
            },
        )
        # Should not attempt pip install — just fail gracefully
        result = await block.execute(BlockInput(block_outputs={}))
        assert result.status == "error"

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


class TestSandboxSecurity:
    """Verify sandbox escape vectors are blocked."""

    @pytest.mark.asyncio
    async def test_dunder_class_blocked(self):
        """().__class__.__bases__ escape is rejected."""
        from agentic.workflow.block import BlockInput
        from agentic.workflow.blocks.function import FunctionBlock

        block = FunctionBlock(config={"code": "x = ().__class__.__bases__"})
        result = await block.execute(BlockInput(block_outputs={}))
        assert result.status == "error"
        assert "dunder" in result.error.lower() or "not allowed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_dunder_subclasses_blocked(self):
        from agentic.workflow.block import BlockInput
        from agentic.workflow.blocks.function import FunctionBlock

        block = FunctionBlock(config={"code": "x = object.__subclasses__()"})
        result = await block.execute(BlockInput(block_outputs={}))
        assert result.status == "error"

    @pytest.mark.asyncio
    async def test_dunder_globals_blocked(self):
        from agentic.workflow.block import BlockInput
        from agentic.workflow.blocks.function import FunctionBlock

        block = FunctionBlock(config={"code": "x = print.__globals__"})
        result = await block.execute(BlockInput(block_outputs={}))
        assert result.status == "error"

    @pytest.mark.asyncio
    async def test_dunder_import_in_code_blocked(self):
        """Direct __import__ call in user code is blocked."""
        from agentic.workflow.block import BlockInput
        from agentic.workflow.blocks.function import FunctionBlock

        block = FunctionBlock(config={"code": "x = __import__('os')"})
        result = await block.execute(BlockInput(block_outputs={}))
        assert result.status == "error"

    @pytest.mark.asyncio
    async def test_dunder_in_string_literal_allowed(self):
        """Dunder in a string value should NOT be blocked (not code execution)."""
        from agentic.workflow.block import BlockInput
        from agentic.workflow.blocks.function import FunctionBlock

        block = FunctionBlock(config={"code": "output = '__class__ is a string'"})
        result = await block.execute(BlockInput(block_outputs={}))
        assert result.data["output"] == "__class__ is a string"

    @pytest.mark.asyncio
    async def test_type_removed_from_builtins(self):
        """type() is removed to prevent dynamic class creation."""
        from agentic.workflow.block import BlockInput
        from agentic.workflow.blocks.function import FunctionBlock

        block = FunctionBlock(config={"code": "output = type(42)"})
        result = await block.execute(BlockInput(block_outputs={}))
        assert result.status == "error"


class TestPipInstallDisabled:
    """Verify pip install is fully disabled."""

    @pytest.mark.asyncio
    async def test_custom_packages_no_pip_install(self):
        """custom_packages config never triggers pip install (subprocess removed)."""
        from agentic.workflow.blocks import function as fn_module

        # subprocess should not even be importable from the module
        assert not hasattr(fn_module, "subprocess")

    @pytest.mark.asyncio
    async def test_unknown_import_no_pip_install(self):
        """Unknown imports parsed from code never trigger pip install."""
        from agentic.workflow.block import BlockInput
        from agentic.workflow.blocks.function import FunctionBlock

        block = FunctionBlock(
            config={
                "code": "import evil_package\noutput = 1",
            },
        )
        # Should just fail gracefully — no subprocess call possible
        result = await block.execute(BlockInput(block_outputs={}))
        assert result.status == "error"

    @pytest.mark.asyncio
    async def test_preinstalled_custom_package_still_works(self):
        """Already-installed packages via custom_packages still import fine."""
        from agentic.workflow.block import BlockInput
        from agentic.workflow.blocks.function import FunctionBlock

        block = FunctionBlock(
            config={
                "code": "output = json.dumps({'a': 1})",
                "imports": [],
                "custom_packages": "json",
            },
        )
        result = await block.execute(BlockInput(block_outputs={}))
        assert result.data["output"] == '{"a": 1}'
