"""Function block — executes sandboxed Python code."""

from __future__ import annotations

import copy
import importlib
import logging
import re
from typing import Any

from agentic.workflow.block import BaseBlock, BlockInput, BlockOutput

logger = logging.getLogger(__name__)

# ─── Import registry ─────────────────────────────────────────────────────────
# key -> (module_name, alias_or_None, is_stdlib)

IMPORT_REGISTRY: dict[str, tuple[str, str | None, bool]] = {
    # Stdlib
    "json": ("json", None, True),
    "re": ("re", None, True),
    "math": ("math", None, True),
    "datetime": ("datetime", None, True),
    "collections": ("collections", None, True),
    "itertools": ("itertools", None, True),
    "functools": ("functools", None, True),
    "hashlib": ("hashlib", None, True),
    "base64": ("base64", None, True),
    "urllib": ("urllib", None, True),
    "csv": ("csv", None, True),
    "io": ("io", None, True),
    "uuid": ("uuid", None, True),
    "copy": ("copy", None, True),
    "string": ("string", None, True),
    "textwrap": ("textwrap", None, True),
    "operator": ("operator", None, True),
    "typing": ("typing", None, True),
    # Third-party
    "numpy": ("numpy", "np", False),
    "pandas": ("pandas", "pd", False),
    "scipy": ("scipy", None, False),
    "seaborn": ("seaborn", "sns", False),
    "sklearn": ("sklearn", None, False),
    "matplotlib": ("matplotlib", None, False),
    "requests": ("requests", None, False),
    "httpx": ("httpx", None, False),
    "pydantic": ("pydantic", None, False),
    "yaml": ("yaml", None, False),
    "bs4": ("bs4", None, False),
    "PIL": ("PIL", None, False),
    "dateutil": ("dateutil", None, False),
    "tiktoken": ("tiktoken", None, False),
}

LEGACY_DEFAULT_IMPORTS = ["numpy", "pandas", "scipy", "seaborn", "sklearn"]


def _load_imports(selected_keys: list[str]) -> dict[str, Any]:
    """Dynamically import only the selected libraries."""
    result: dict[str, Any] = {}
    for key in selected_keys:
        entry = IMPORT_REGISTRY.get(key)
        if not entry:
            continue  # ignore unknown keys
        module_name, alias, _is_stdlib = entry
        try:
            mod = importlib.import_module(module_name)
            result[module_name] = mod
            if alias:
                result[alias] = mod
        except ImportError:
            logger.warning("Library %s not available", module_name)
    return result


# ─── Custom packages ─────────────────────────────────────────────────────────

_PIP_TO_IMPORT: dict[str, str] = {
    "Pillow": "PIL",
    "pillow": "PIL",
    "scikit-learn": "sklearn",
    "beautifulsoup4": "bs4",
    "pyyaml": "yaml",
    "opencv-python": "cv2",
    "python-dateutil": "dateutil",
}

# Reverse mapping: import name -> pip package name (for name resolution)
_IMPORT_TO_PIP: dict[str, str] = {v: k for k, v in _PIP_TO_IMPORT.items()}


def _parse_imports_from_code(code: str) -> tuple[list[str], list[str]]:
    """Extract module names from import statements in code.

    Returns (registry_keys, custom_modules) where:
    - registry_keys: modules found in IMPORT_REGISTRY
    - custom_modules: everything else (will be imported if pre-installed)
    """
    import_pattern = re.compile(
        r"^\s*(?:import\s+([\w.]+(?:\s*,\s*[\w.]+)*)"
        r"|from\s+([\w.]+)\s+import\s+)",
        re.MULTILINE,
    )
    modules: set[str] = set()
    for match in import_pattern.finditer(code):
        if match.group(1):  # import X, Y
            for name in match.group(1).split(","):
                modules.add(name.strip().split(".")[0])
        elif match.group(2):  # from X import ...
            modules.add(match.group(2).strip().split(".")[0])

    registry_keys = []
    custom_modules = []
    for mod in modules:
        if mod in IMPORT_REGISTRY:
            registry_keys.append(mod)
        else:
            custom_modules.append(mod)
    return registry_keys, custom_modules


def _parse_package_specs(raw: str) -> list[tuple[str, str]]:
    """Split comma-separated pip specifiers into (module_name, pip_spec) tuples."""
    if not raw or not raw.strip():
        return []
    results: list[tuple[str, str]] = []
    for part in raw.split(","):
        spec = part.strip()
        if not spec:
            continue
        # Extract the bare package name (before any version specifier)
        bare_name = re.split(r"[><=!~;@\[]", spec)[0].strip()
        # Map pip name to import name if known, otherwise use bare name
        module_name = _PIP_TO_IMPORT.get(bare_name, bare_name.replace("-", "_"))
        results.append((module_name, spec))
    return results


def _pip_install(pip_spec: str) -> bool:
    """Pip install is disabled for security. Always returns False."""
    logger.warning(
        "pip install disabled for security — %s must be pre-installed", pip_spec
    )
    return False


def _load_custom_packages(raw: str) -> dict[str, Any]:
    """Parse custom_packages string and import pre-installed modules."""
    result: dict[str, Any] = {}
    for module_name, _pip_spec in _parse_package_specs(raw):
        try:
            mod = importlib.import_module(module_name)
            result[module_name] = mod
        except ImportError:
            logger.warning(
                "Package %s is not pre-installed and cannot be used", module_name
            )
    return result


# Restricted builtins for sandboxed execution
_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "None": None,
    "True": True,
    "False": False,
}


def _check_dunder_access(code: str) -> str | None:
    """Reject code containing dunder attribute access.

    Returns None if safe, error message if blocked.
    Allows dunders inside string literals.
    """
    for i, line in enumerate(code.splitlines(), 1):
        # Strip string literals from the line before checking
        stripped = re.sub(r"'[^']*'|\"[^\"]*\"", "", line)
        if re.search(r"__\w+__", stripped):
            return f"Dunder attribute access is not allowed (line {i})"
    return None


class FunctionBlock(BaseBlock):
    block_type = "code"

    async def execute(self, block_input: BlockInput) -> BlockOutput:
        code = self.config.get("code", "")
        if not code.strip():
            return BlockOutput(data={"output": None})

        resolved_code = code

        # Check for dunder attribute access (sandbox escape vector)
        dunder_error = _check_dunder_access(resolved_code)
        if dunder_error:
            return BlockOutput(
                data={"output": None, "error": dunder_error},
                status="error",
                error=dunder_error,
            )

        # Build sandbox environment
        sandbox: dict[str, Any] = {"__builtins__": dict(_SAFE_BUILTINS)}
        # Expose block outputs as variables
        for block_id, output in block_input.block_outputs.items():
            safe_name = block_id.replace("-", "_")
            sandbox[safe_name] = output
        # Expose input data
        sandbox["input_data"] = copy.deepcopy(block_input.block_outputs)

        # Backward compat: old workflows with explicit imports config
        selected = self.config.get("imports")
        if selected is not None:
            sandbox.update(_load_imports(selected))
            custom_raw = self.config.get("custom_packages", "")
            if not isinstance(custom_raw, str):
                logger.warning(
                    "custom_packages must be a string, got %s",
                    type(custom_raw).__name__,
                )
                custom_raw = ""
            if custom_raw:
                sandbox.update(_load_custom_packages(custom_raw))
        else:
            # New behavior: parse imports from code
            registry_keys, custom_modules = _parse_imports_from_code(code)
            sandbox.update(_load_imports(registry_keys))
            if custom_modules:
                # Map import names to pip names where needed
                pip_specs = []
                for mod in custom_modules:
                    pip_specs.append(_IMPORT_TO_PIP.get(mod, mod))
                sandbox.update(_load_custom_packages(",".join(pip_specs)))

        # Build a controlled __import__ that only allows modules the user
        # explicitly imported or that are in the registry / stdlib safe-list.
        allowed_modules = set(IMPORT_REGISTRY.keys())
        if selected is None:
            # Add all parsed imports so user's import statements work in exec
            allowed_modules.update(registry_keys)
            allowed_modules.update(custom_modules)
        else:
            # Old path: allow whatever was selected + custom packages
            allowed_modules.update(selected)
            custom_raw = self.config.get("custom_packages", "")
            if isinstance(custom_raw, str) and custom_raw.strip():
                for mod_name, _ in _parse_package_specs(custom_raw):
                    allowed_modules.add(mod_name.split(".")[0])

        def _controlled_import(name, globals=None, locals=None, fromlist=(), level=0):
            root = name.split(".")[0]
            if root not in allowed_modules:
                raise ImportError(
                    f"Import of '{name}' is not allowed. "
                    f"Add it to your import statements at the top of the code."
                )
            return importlib.__import__(name, globals, locals, fromlist, level)

        sandbox["__builtins__"]["__import__"] = _controlled_import

        try:
            exec(resolved_code, sandbox)  # noqa: S102
            result = sandbox.get("output", sandbox.get("result"))
            return BlockOutput(data={"output": result})
        except Exception as e:
            logger.error("Function block execution failed: %s", e)
            return BlockOutput(
                data={"output": None, "error": str(e)},
                status="error",
                error=str(e),
            )
