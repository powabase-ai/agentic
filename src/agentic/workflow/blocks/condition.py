"""Condition block — evaluates a boolean expression for branching."""

from __future__ import annotations

import logging
import re

from agentic.workflow.block import BaseBlock, BlockInput, BlockOutput
from agentic.workflow.expression import evaluate_expression
from agentic.workflow.variable_resolver import resolve_variable

logger = logging.getLogger(__name__)

_VAR_PATTERN = re.compile(r"<(\w+(?:\.\w+)+)>")


def _coerce_numeric(value: object) -> object:
    """Try to convert a string to int or float for numeric comparisons."""
    if not isinstance(value, str):
        return value
    value = value.strip()
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


class ConditionBlock(BaseBlock):
    block_type = "condition"

    async def execute(self, block_input: BlockInput) -> BlockOutput:
        raw_expression = self.config.get("expression", "True")

        # Build variables for the expression evaluator
        variables = dict(block_input.block_outputs)
        variables.update(block_input.data)

        # Resolve <ref> patterns into placeholder variables so that
        # resolved values keep their Python type (str, int, etc.)
        # instead of being text-substituted into the expression.
        ref_counter = 0
        resolved_expression = raw_expression

        def _replace_ref(match: re.Match) -> str:
            nonlocal ref_counter
            ref = match.group(1)
            value = resolve_variable(ref, block_input.block_outputs)
            # If unresolved, leave as-is (will likely fail evaluation)
            if isinstance(value, str) and value == f"<{ref}>":
                return value
            placeholder = f"_ref_{ref_counter}"
            ref_counter += 1
            variables[placeholder] = _coerce_numeric(value)
            return placeholder

        resolved_expression = _VAR_PATTERN.sub(_replace_ref, str(raw_expression))

        logger.info(
            "Condition eval: raw=%r resolved=%r variables=%r",
            raw_expression,
            resolved_expression,
            {k: v for k, v in variables.items() if k.startswith("_ref_")},
        )

        try:
            result = bool(evaluate_expression(resolved_expression, variables))
        except Exception as e:
            logger.error("Condition evaluation failed: %s", e)
            result = False

        return BlockOutput(
            data={
                "output": result,
                "result": result,
                "route": "true" if result else "false",
            }
        )
