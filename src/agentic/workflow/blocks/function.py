"""Function block — executes sandboxed Python code."""

from __future__ import annotations

import logging
from typing import Any

from agentic.workflow.block import BaseBlock, BlockInput, BlockOutput

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import scipy
except ImportError:
    scipy = None

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    import sklearn
except ImportError:
    sklearn = None

logger = logging.getLogger(__name__)

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

_DATA_SCIENCE_LIBS: dict[str, Any] = {}
if np is not None:
    _DATA_SCIENCE_LIBS["np"] = np
    _DATA_SCIENCE_LIBS["numpy"] = np
if pd is not None:
    _DATA_SCIENCE_LIBS["pd"] = pd
    _DATA_SCIENCE_LIBS["pandas"] = pd
if scipy is not None:
    _DATA_SCIENCE_LIBS["scipy"] = scipy
if sns is not None:
    _DATA_SCIENCE_LIBS["sns"] = sns
    _DATA_SCIENCE_LIBS["seaborn"] = sns
if sklearn is not None:
    _DATA_SCIENCE_LIBS["sklearn"] = sklearn


class FunctionBlock(BaseBlock):
    block_type = "function"

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
        # Expose pre-imported data science libraries
        sandbox.update(_DATA_SCIENCE_LIBS)

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
