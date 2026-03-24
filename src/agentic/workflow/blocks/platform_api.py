"""Platform API block — calls internal platform service endpoints."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from agentic.workflow.block import BaseBlock, BlockInput, BlockOutput
from agentic.workflow.variable_resolver import resolve_value

logger = logging.getLogger(__name__)

_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

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
    ("database", "list_tables"): ("GET", "/api/database/tables"),
    ("database", "list"): ("GET", "/api/database/tables/{table}"),
    ("database", "get"): ("GET", "/api/database/tables/{table}/{row_id}"),
    ("database", "create"): ("POST", "/api/database/tables/{table}"),
    ("database", "update"): ("PATCH", "/api/database/tables/{table}/{row_id}"),
    ("database", "delete"): ("DELETE", "/api/database/tables/{table}/{row_id}"),
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


_BODY_FIELD_MAP: dict[tuple[str, str], list[tuple[str, str]]] = {
    ("agents", "create"): [
        ("agents_cu_name", "name"),
        ("agents_cu_model", "model"),
        ("agents_cu_system_prompt", "system_prompt"),
        ("agents_cu_settings", "settings"),
    ],
    ("agents", "update"): [
        ("agents_cu_name", "name"),
        ("agents_cu_model", "model"),
        ("agents_cu_system_prompt", "system_prompt"),
        ("agents_cu_settings", "settings"),
    ],
    ("agents", "run"): [
        ("agents_run_message", "message"),
        ("agents_run_session_id", "session_id"),
        ("agents_run_kb", "knowledge_bases"),
        ("agents_run_context_handler_id", "context_handler_id"),
        ("agents_run_max_context_tokens", "max_context_tokens"),
    ],
    ("knowledge_bases", "create"): [
        ("kb_cu_name", "name"),
        ("kb_cu_description", "description"),
        ("kb_cu_indexing_config", "indexing_config"),
        ("kb_cu_retrieval_config", "retrieval_config"),
    ],
    ("knowledge_bases", "update"): [
        ("kb_cu_name", "name"),
        ("kb_cu_description", "description"),
        ("kb_cu_indexing_config", "indexing_config"),
        ("kb_cu_retrieval_config", "retrieval_config"),
    ],
    ("knowledge_bases", "search"): [
        ("kb_search_query", "query"),
        ("kb_search_top_k", "top_k"),
        ("kb_search_retrieval_method", "retrieval_method"),
        ("kb_search_similarity_threshold", "similarity_threshold"),
        ("kb_search_filter_metadata", "filter_metadata"),
    ],
    ("context_handlers", "create"): [
        ("ch_create_query", "query"),
        ("ch_create_kb", "knowledge_bases"),
        ("ch_create_max_context_tokens", "max_context_tokens"),
    ],
}


class PlatformAPIBlock(BaseBlock):
    block_type = "platform_api"

    def _assemble_body(
        self, resource: str, operation: str, block_outputs: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Build a request body from form-field config keys.

        Mirrors the frontend's BODY_FIELDS logic so that input-mapped
        values (which arrive only at runtime) are included in the body.
        """
        field_list = _BODY_FIELD_MAP.get((resource, operation))
        if not field_list:
            return None

        body: dict[str, Any] = {}
        for config_key, body_key in field_list:
            raw = self.config.get(config_key)
            if raw is None:
                continue
            value = resolve_value(raw, block_outputs)
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            # Attempt JSON parse for fields that may hold structured data
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    pass
            body[body_key] = value
        return body if body else None

    def _get_operation(
        self, resource: str, block_outputs: dict[str, Any]
    ) -> str | None:
        """Read the correct operation config key based on the resource."""
        key = _OPERATION_KEY_MAP.get(resource)
        if not key:
            return None
        return resolve_value(self.config.get(key, ""), block_outputs)

    def _resolve_route(
        self,
        resource: str,
        operation: str,
        resource_id: str | None,
        block_outputs: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """Look up (method, path) and substitute placeholders."""
        entry = ROUTE_MAP.get((resource, operation))
        if not entry:
            raise ValueError(
                f"Unknown route: resource={resource!r}, operation={operation!r}"
            )
        method, path_template = entry
        path = path_template.replace("{id}", str(resource_id) if resource_id else "")

        # Database table/row_id placeholders
        if "{table}" in path:
            table = resolve_value(
                self.config.get("db_table", ""), block_outputs or {}
            )
            if not table or not _TABLE_NAME_RE.match(str(table)):
                raise ValueError(
                    f"Invalid table name: {table!r} (alphanumeric and underscores only)"
                )
            path = path.replace("{table}", str(table))
        if "{row_id}" in path:
            row_id = resolve_value(
                self.config.get("db_row_id", ""), block_outputs or {}
            )
            if not row_id:
                raise ValueError("db_row_id is required for this operation")
            path = path.replace("{row_id}", str(row_id))

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
                resource, operation, resource_id or None,
                block_outputs=block_input.block_outputs,
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

        # Assemble body from form fields and merge with any pre-existing body.
        # The frontend pre-assembles static fields into body at save time, but
        # input-mapped values only arrive at runtime. We merge so both are included.
        # Skip when user opted for raw JSON editing (matches frontend behaviour).
        use_raw_json = self.config.get("use_raw_json", False)
        assembled = (
            self._assemble_body(resource, operation, block_input.block_outputs)
            if not use_raw_json
            else None
        )
        if assembled:
            if isinstance(body, dict):
                # Start with assembled fields, then overlay non-empty body values.
                # This ensures runtime-mapped fields fill gaps left by empty
                # frontend placeholders, while explicit body values win.
                merged = dict(assembled)
                for k, v in body.items():
                    if v is not None and v != "":
                        merged[k] = v
                body = merged
            elif not body or (isinstance(body, str) and not body.strip()):
                body = assembled

        # Handle db_body directly for database create/update (no wrapping key)
        if resource == "database" and operation in ("create", "update") and not use_raw_json:
            db_body_raw = resolve_value(
                self.config.get("db_body", ""), block_input.block_outputs
            )
            if isinstance(db_body_raw, str) and db_body_raw.strip():
                try:
                    body = json.loads(db_body_raw)
                except json.JSONDecodeError:
                    pass
            elif isinstance(db_body_raw, dict):
                body = db_body_raw

        # Parse params if string
        if isinstance(params, str) and params.strip():
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {}

        # Inject limit/offset for database list operations
        if resource == "database" and operation == "list":
            if not isinstance(params, dict):
                params = {}
            limit = resolve_value(
                self.config.get("db_list_limit", ""), block_input.block_outputs
            )
            offset = resolve_value(
                self.config.get("db_list_offset", ""), block_input.block_outputs
            )
            if limit:
                params["limit"] = str(limit)
            if offset:
                params["offset"] = str(offset)

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
