"""Orchestration block — runs a multi-agent orchestration."""

from __future__ import annotations

import logging

from agentic.workflow.block import BaseBlock, BlockInput, BlockOutput
from agentic.workflow.variable_resolver import resolve_value

logger = logging.getLogger(__name__)


class OrchestrationBlock(BaseBlock):
    block_type = "orchestration"

    async def execute(self, block_input: BlockInput) -> BlockOutput:
        input_msg = resolve_value(
            self.config.get("input", ""), block_input.block_outputs
        )

        run_orchestration = block_input.services.get("run_orchestration")
        if not run_orchestration:
            raise RuntimeError("run_orchestration service not available")

        orchestration_id = self.config.get("orchestration_id")
        if not orchestration_id:
            raise RuntimeError("orchestration_id is required")

        result = await run_orchestration(
            orchestration_id=orchestration_id,
            message=str(input_msg),
        )

        return BlockOutput(
            data={
                "output": result.get("content", ""),
                "status": result.get("status", "completed"),
            }
        )
