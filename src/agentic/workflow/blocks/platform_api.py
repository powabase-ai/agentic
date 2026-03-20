"""Platform API block — calls internal platform service endpoints."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from agentic.workflow.block import BaseBlock, BlockInput, BlockOutput
from agentic.workflow.variable_resolver import resolve_value

logger = logging.getLogger(__name__)

# Maps (resource, operation) -> (HTTP method, path template)
# Path templates use {id} as placeholder for resource_id.
ROUTE_MAP: dict[tuple[str, str], tuple[str, str]] = {
    # Agents
    ("agents", "list"): ("GET", "/api/agents"),
    ("agents", "get"): ("GET", "/api/agents/{id}"),
    ("agents", "run"): ("POST", "/api/agents/{id}/run"),
    ("agents", "create"): ("POST", "/api/agents"),
    ("agents", "update"): ("PATCH", "/api/agents/{id}"),
    ("agents", "delete"): ("DELETE", "/api/agents/{id}"),
    # Knowledge Bases
    ("knowledge_bases", "list"): ("GET", "/api/knowledge-bases"),
    ("knowledge_bases", "get"): ("GET", "/api/knowledge-bases/{id}"),
    ("knowledge_bases", "search"): ("POST", "/api/knowledge-bases/{id}/search"),
    ("knowledge_bases", "create"): ("POST", "/api/knowledge-bases"),
    ("knowledge_bases", "update"): ("PATCH", "/api/knowledge-bases/{id}"),
    ("knowledge_bases", "delete"): ("DELETE", "/api/knowledge-bases/{id}"),
    # Sources
    ("sources", "list"): ("GET", "/api/sources"),
    ("sources", "get"): ("GET", "/api/sources/{id}"),
    ("sources", "delete"): ("DELETE", "/api/sources/{id}"),
    # Sessions
    ("sessions", "get"): ("GET", "/api/sessions/{id}"),
    ("sessions", "list_messages"): ("GET", "/api/sessions/{id}/messages"),
    ("sessions", "delete"): ("DELETE", "/api/sessions/{id}"),
    # Context Handlers
    ("context_handlers", "list"): ("GET", "/api/context-handlers"),
    ("context_handlers", "get"): ("GET", "/api/context-handlers/{id}"),
    ("context_handlers", "create"): ("POST", "/api/context-handlers"),
    # Database
    ("database", "query"): ("POST", "/api/database/query"),
}

# Maps resource name -> config key for the operation dropdown
_OPERATION_KEY_MAP: dict[str, str] = {
    "agents": "agents_operation",
    "knowledge_bases": "kb_operation",
    "sources": "sources_operation",
    "sessions": "sessions_operation",
    "context_handlers": "ch_operation",
    "database": "db_operation",
}


class PlatformAPIBlock(BaseBlock):
    block_type = "platform_api"

    def _get_operation(
        self, resource: str, block_outputs: dict[str, Any]
    ) -> str | None:
        """Read the correct operation config key based on the resource."""
        key = _OPERATION_KEY_MAP.get(resource)
        if not key:
            return None
        return resolve_value(self.config.get(key, ""), block_outputs)

    def _resolve_route(
        self, resource: str, operation: str, resource_id: str | None
    ) -> tuple[str, str]:
        """Look up (method, path) and substitute resource_id."""
        entry = ROUTE_MAP.get((resource, operation))
        if not entry:
            raise ValueError(
                f"Unknown route: resource={resource!r}, operation={operation!r}"
            )
        method, path_template = entry
        path = path_template.replace("{id}", resource_id or "")
        return method, path

    async def execute(self, block_input: BlockInput) -> BlockOutput:
        base_url = block_input.services.get("platform_api_base_url")
        token = block_input.services.get("platform_api_token")

        if not base_url:
            return BlockOutput(
                data={"output": None, "error": "platform_api_base_url not in services"},
                status="error",
                error="platform_api_base_url not in services",
            )

        resource = resolve_value(
            self.config.get("resource", ""), block_input.block_outputs
        )
        operation = self._get_operation(resource, block_input.block_outputs)
        resource_id = resolve_value(
            self.config.get("resource_id", ""), block_input.block_outputs
        )
        body = resolve_value(
            self.config.get("body", ""), block_input.block_outputs
        )
        params = resolve_value(
            self.config.get("params", ""), block_input.block_outputs
        )

        if not resource:
            return BlockOutput(
                data={"output": None, "error": "Resource is required"},
                status="error",
                error="Resource is required",
            )

        if resource not in _OPERATION_KEY_MAP:
            msg = f"Unknown resource: {resource!r}"
            return BlockOutput(
                data={"output": None, "error": msg},
                status="error",
                error=msg,
            )

        if not operation:
            return BlockOutput(
                data={"output": None, "error": "Operation is required"},
                status="error",
                error="Operation is required",
            )

        # Resolve route
        try:
            method, path = self._resolve_route(
                resource, operation, resource_id or None
            )
        except ValueError as e:
            return BlockOutput(
                data={"output": None, "error": str(e)},
                status="error",
                error=str(e),
            )

        # Parse body if string
        if isinstance(body, str) and body.strip():
            try:
                body = json.loads(body)
            except json.JSONDecodeError:
                pass

        # Parse params if string
        if isinstance(params, str) and params.strip():
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {}

        url = f"{base_url.rstrip('/')}{path}"
        headers: dict[str, str] = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=body if isinstance(body, (dict, list)) else None,
                    content=body if isinstance(body, str) and body else None,
                    params=params if isinstance(params, dict) and params and method == "GET" else None,
                )

            try:
                response_data = response.json()
            except (json.JSONDecodeError, ValueError):
                response_data = response.text

            return BlockOutput(
                data={
                    "output": response_data,
                    "status_code": response.status_code,
                }
            )
        except Exception as e:
            logger.error("Platform API call failed: %s", e)
            return BlockOutput(
                data={"output": None, "error": str(e)},
                status="error",
                error=str(e),
            )
