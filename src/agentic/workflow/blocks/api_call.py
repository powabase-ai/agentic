"""API Call block — makes HTTP requests."""

from __future__ import annotations

import json
import logging

import httpx

from agentic.workflow.block import BaseBlock, BlockInput, BlockOutput
from agentic.workflow.variable_resolver import resolve_value

logger = logging.getLogger(__name__)


def _table_to_dict(value):
    """Convert table format [{cells: {Key: "foo", Value: "bar"}}] to {"foo": "bar"}."""
    if isinstance(value, list):
        return {
            row["cells"]["Key"]: row["cells"]["Value"]
            for row in value
            if isinstance(row, dict) and row.get("cells", {}).get("Key")
        }
    return value


class APICallBlock(BaseBlock):
    block_type = "api_call"

    async def execute(self, block_input: BlockInput) -> BlockOutput:
        url = str(resolve_value(self.config.get("url", ""), block_input.block_outputs))
        method = self.config.get("method", "GET").upper()
        headers = resolve_value(self.config.get("headers", {}), block_input.block_outputs)
        body = resolve_value(self.config.get("body", ""), block_input.block_outputs)
        params = resolve_value(self.config.get("params", {}), block_input.block_outputs)
        timeout_val = self.config.get("timeout")

        # Field is labeled "Timeout (ms)" — always convert to seconds
        try:
            timeout_num = float(timeout_val) if timeout_val else 30000
            timeout_num = timeout_num / 1000
        except (ValueError, TypeError):
            timeout_num = 30

        # Handle table format for headers and params
        headers = _table_to_dict(headers)
        params = _table_to_dict(params)

        if isinstance(headers, str):
            try:
                headers = json.loads(headers)
            except json.JSONDecodeError:
                headers = {}

        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {}

        if isinstance(body, str) and body.strip():
            try:
                body = json.loads(body)
            except json.JSONDecodeError:
                pass  # keep as string

        try:
            async with httpx.AsyncClient(timeout=timeout_num) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers if isinstance(headers, dict) else None,
                    params=params if params else None,
                    json=body if isinstance(body, (dict, list)) else None,
                    content=body if isinstance(body, str) and body else None,
                )

            try:
                response_data = response.json()
            except (json.JSONDecodeError, ValueError):
                response_data = response.text

            return BlockOutput(
                data={
                    "output": response_data,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                }
            )
        except Exception as e:
            logger.error("API call failed: %s", e)
            return BlockOutput(
                data={"output": None, "error": str(e)},
                status="error",
                error=str(e),
            )
