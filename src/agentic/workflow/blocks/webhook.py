"""Webhook block — entry point triggered by external HTTP POST."""

from agentic.workflow.block import BaseBlock, BlockInput, BlockOutput


class WebhookBlock(BaseBlock):
    block_type = "webhook"

    async def execute(self, block_input: BlockInput) -> BlockOutput:
        # The webhook payload is passed in as block_input.data by the trigger endpoint
        payload = block_input.data or {}
        return BlockOutput(data={"output": payload})
