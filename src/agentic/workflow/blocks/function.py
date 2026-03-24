"""Function block — executes sandboxed Python code."""

from __future__ import annotations

import importlib
import logging
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
    # Third-party
    "numpy": ("numpy", "np", False),
    "pandas": ("pandas", "pd", False),
    "scipy": ("scipy", None, False),
    "seaborn": ("seaborn", "sns", False),
    "sklearn": ("sklearn", None, False),
    "matplotlib": ("matplotlib", None, False),
    "requests": ("requests", None, False),
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
    "type": type,
    "zip": zip,
    "None": None,
    "True": True,
    "False": False,
}

class FunctionBlock(BaseBlock):
    block_type = "code"

    async def execute(self, block_input: BlockInput) -> BlockOutput:
        code = self.config.get("code", "")
        if not code.strip():
            return BlockOutput(data={"output": None})

        resolved_code = code

        # Build sandbox environment
        sandbox: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
        # Expose block outputs as variables
        for block_id, output in block_input.block_outputs.items():
            safe_name = block_id.replace("-", "_")
            sandbox[safe_name] = output
        # Expose input data
        sandbox["input_data"] = block_input.block_outputs
        # Load selected imports (fall back to legacy defaults for old workflows)
        selected = self.config.get("imports")
        if selected is None:
            selected = LEGACY_DEFAULT_IMPORTS
        sandbox.update(_load_imports(selected))

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
