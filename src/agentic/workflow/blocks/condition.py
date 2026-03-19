"""Condition block — evaluates a boolean expression for branching."""

from __future__ import annotations

import logging

from agentic.workflow.block import BaseBlock, BlockInput, BlockOutput
from agentic.workflow.expression import evaluate_expression
from agentic.workflow.variable_resolver import resolve_value

logger = logging.getLogger(__name__)


class ConditionBlock(BaseBlock):
    block_type = "condition"

    async def execute(self, block_input: BlockInput) -> BlockOutput:
        expression = self.config.get("expression", "True")
        expression = str(resolve_value(expression, block_input.block_outputs))

        # Build variables for the expression evaluator
        variables = dict(block_input.block_outputs)
        variables.update(block_input.data)

        try:
            result = bool(evaluate_expression(expression, variables))
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
