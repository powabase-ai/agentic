"""Split block — forwards input identically to multiple parallel branches."""

from agentic.workflow.block import BaseBlock, BlockInput, BlockOutput


class SplitBlock(BaseBlock):
    block_type = "split"

    async def execute(self, block_input: BlockInput) -> BlockOutput:
        input_data = self.config.get("input", "")
        return BlockOutput(data={"output": input_data})
