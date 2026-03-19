"""Starter block — entry point that passes input variables through."""

import json

from agentic.workflow.block import BaseBlock, BlockInput, BlockOutput


class StarterBlock(BaseBlock):
    block_type = "starter"

    async def execute(self, block_input: BlockInput) -> BlockOutput:
        # Config "input" defines defaults; runtime variables override
        config_input = self.config.get("input", {})
        if isinstance(config_input, str):
            try:
                config_input = json.loads(config_input)
            except (json.JSONDecodeError, TypeError):
                config_input = {}
        if not isinstance(config_input, dict):
            config_input = {}
        merged = {**config_input, **block_input.data}
        return BlockOutput(data={"output": merged})
