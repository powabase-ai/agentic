"""Split block — forwards input identically to multiple parallel branches."""

import logging

from agentic.workflow.block import BaseBlock, BlockInput, BlockOutput

logger = logging.getLogger(__name__)


class SplitBlock(BaseBlock):
    block_type = "split"

    async def execute(self, block_input: BlockInput) -> BlockOutput:
        input_data = self.config.get("input")

        # Fallback: if _inputMappings didn't populate config,
        # try to get data from the most recent upstream block
        if input_data is None and block_input.block_outputs:
            for _bid, bout in reversed(list(block_input.block_outputs.items())):
                if isinstance(bout, dict) and "output" in bout:
                    input_data = bout["output"]
                    break
                elif bout is not None:
                    input_data = bout
                    break

        if input_data is None:
            input_data = ""

        logger.debug("SplitBlock config: %s, resolved input: %s", self.config, input_data)
        return BlockOutput(data={"output": input_data})
