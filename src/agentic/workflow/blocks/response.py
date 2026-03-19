"""Response block — terminal output node that formats the final result."""

from agentic.workflow.block import BaseBlock, BlockInput, BlockOutput
from agentic.workflow.variable_resolver import resolve_value


class ResponseBlock(BaseBlock):
    block_type = "response"

    async def execute(self, block_input: BlockInput) -> BlockOutput:
        output_template = self.config.get("output", "")
        result = resolve_value(output_template, block_input.block_outputs)

        status = self.config.get("status", "200")
        headers = self.config.get("headers", [])

        # Convert table format to dict if needed
        if isinstance(headers, list):
            headers = {
                row["cells"]["Key"]: row["cells"]["Value"]
                for row in headers
                if isinstance(row, dict) and row.get("cells", {}).get("Key")
            }

        try:
            status_code = int(status) if status else 200
        except (ValueError, TypeError):
            status_code = 200

        return BlockOutput(
            data={
                "output": result,
                "status": status_code,
                "headers": headers,
            }
        )
