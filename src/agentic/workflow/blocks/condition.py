"""Condition block — if/else-if/else branching with independent expressions."""

from __future__ import annotations

import logging
import re

from agentic.workflow.block import BaseBlock, BlockInput, BlockOutput
from agentic.workflow.expression import _coerce_value, evaluate_expression
from agentic.workflow.variable_resolver import resolve_variable

logger = logging.getLogger(__name__)

_VAR_PATTERN = re.compile(r"<([\w -]+\.[\w.]+)>")


class ConditionBlock(BaseBlock):
    block_type = "condition"

    def _migrate_legacy_config(self) -> list[dict]:
        """Convert old boolean/switch config to branches array."""
        mode = self.config.get("mode", "boolean")
        expression = self.config.get("expression", "True")

        if mode == "switch":
            cases = self.config.get("cases", [])
            branches = []
            for row in cases:
                if not isinstance(row, dict):
                    continue
                label = row.get("cells", {}).get("Case Label", "").strip()
                if label:
                    branches.append(
                        {"expression": f'{expression} == "{label}"'}
                    )
            return branches if branches else [{"expression": expression}]

        # Boolean mode (default)
        return [{"expression": expression}]

    async def execute(self, block_input: BlockInput) -> BlockOutput:
        branches = self.config.get("branches", [])
        if not branches and ("mode" in self.config or "expression" in self.config):
            branches = self._migrate_legacy_config()

        variables = dict(block_input.block_outputs)
        variables.update(block_input.data)

        branch_diagnostics = []

        for idx, branch in enumerate(branches):
            raw_expr = branch.get("expression", "True")
            handle_id = "if" if idx == 0 else f"elif_{idx}"

            ref_counter = 0
            branch_vars = {}

            def _replace_ref(match: re.Match) -> str:
                nonlocal ref_counter
                ref = match.group(1)
                value = resolve_variable(ref, block_input.block_outputs)
                if isinstance(value, str) and value == f"<{ref}>":
                    return value
                placeholder = f"_ref_{ref_counter}"
                ref_counter += 1
                coerced = _coerce_value(value)
                variables[placeholder] = coerced
                branch_vars[placeholder] = coerced
                return placeholder

            resolved = _VAR_PATTERN.sub(_replace_ref, str(raw_expr))

            logger.info(
                "Condition branch %s eval: raw=%r resolved=%r",
                handle_id,
                raw_expr,
                resolved,
            )

            try:
                result = bool(evaluate_expression(resolved, variables))
            except Exception as e:
                logger.error("Branch %s eval failed: %s", handle_id, e)
                branch_diagnostics.append({
                    "branch": handle_id,
                    "raw": raw_expr,
                    "resolved": resolved,
                    "variables": branch_vars,
                    "error": str(e),
                })
                continue

            branch_diagnostics.append({
                "branch": handle_id,
                "raw": raw_expr,
                "resolved": resolved,
                "variables": branch_vars,
                "result": result,
            })

            if result:
                return BlockOutput(
                    data={
                        "output": True,
                        "result": handle_id,
                        "route": handle_id,
                        "_debug": branch_diagnostics,
                    }
                )

        return BlockOutput(
            data={
                "output": False,
                "result": "else",
                "route": "else",
                "_debug": branch_diagnostics,
            }
        )
