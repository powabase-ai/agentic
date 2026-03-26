"""
DAG Engine — executes a directed acyclic graph of workflow blocks.

Uses Kahn's algorithm for topological ordering and supports both
batch and streaming execution modes.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from collections.abc import AsyncGenerator
from typing import Any

from agentic.workflow.block import BlockInput, BlockOutput, BlockRegistry
from agentic.workflow.variable_resolver import resolve_value

logger = logging.getLogger(__name__)


def _resolve_output_path(source_data: Any, output_field: str) -> Any | None:
    """Walk a dotted path through block output data."""
    parts = output_field.split(".")
    value = source_data
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None
    return value


@dataclass
class BlockDefinition:
    """A block within a workflow graph."""

    id: str
    type: str
    name: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class EdgeDefinition:
    """A connection between two blocks."""

    id: str
    source_block_id: str
    target_block_id: str
    source_handle: str = "output"
    target_handle: str = "input"
    condition: str | None = None


@dataclass
class ExecutionEvent:
    """Event emitted during workflow execution."""

    type: str  # block_start, block_chunk, block_complete, block_error
    block_id: str
    block_type: str = ""
    data: Any = None
    duration_ms: float | None = None


class DAGEngine:
    """Execute a DAG of blocks using topological ordering."""

    def __init__(
        self,
        blocks: list[BlockDefinition],
        edges: list[EdgeDefinition],
        variables: dict[str, Any] | None = None,
        services: dict[str, Any] | None = None,
    ) -> None:
        self._blocks = {b.id: b for b in blocks}
        self._edges = edges
        self._variables = variables or {}
        self._services = services or {}

        # Build adjacency and reverse-adjacency maps
        self._adj: dict[str, list[EdgeDefinition]] = {b.id: [] for b in blocks}
        self._in_edges: dict[str, list[EdgeDefinition]] = {b.id: [] for b in blocks}
        for edge in edges:
            self._adj[edge.source_block_id].append(edge)
            self._in_edges[edge.target_block_id].append(edge)

    def _topological_order(self) -> list[str]:
        """Return block IDs in topological order using Kahn's algorithm."""
        in_degree: dict[str, int] = {bid: 0 for bid in self._blocks}
        for edge in self._edges:
            in_degree[edge.target_block_id] += 1

        queue: deque[str] = deque(
            bid for bid, deg in in_degree.items() if deg == 0
        )
        order: list[str] = []

        while queue:
            bid = queue.popleft()
            order.append(bid)
            for edge in self._adj[bid]:
                in_degree[edge.target_block_id] -= 1
                if in_degree[edge.target_block_id] == 0:
                    queue.append(edge.target_block_id)

        if len(order) != len(self._blocks):
            raise ValueError("Workflow graph contains a cycle")
        return order

    def _should_execute_block(
        self,
        block_id: str,
        block_outputs: dict[str, Any],
    ) -> bool:
        """Check whether a block should execute based on incoming edge conditions."""
        bdef = self._blocks[block_id]
        if not bdef.enabled:
            return False

        in_edges = self._in_edges[block_id]
        if not in_edges:
            return True  # starter block (no incoming edges)

        for edge in in_edges:
            source_output = block_outputs.get(edge.source_block_id)
            if source_output is None:
                continue

            # If edge has a condition (e.g. "true"/"false"), check route
            if edge.condition is not None:
                route = None
                if isinstance(source_output, dict):
                    route = source_output.get("route")
                if route is None:
                    continue  # source has no routing info — can't satisfy condition
                if str(route) != str(edge.condition):
                    continue
                return True  # route matches condition

            # No condition — just needs source to have produced output
            return True

        # If all edges had conditions and none matched, skip
        if all(e.condition is not None for e in in_edges):
            return False
        return False

    @staticmethod
    def _apply_input_mappings(
        resolved_config: dict[str, Any],
        block_outputs: dict[str, Any],
    ) -> None:
        """Process structured input mappings, supporting dotted output paths."""
        input_mappings = resolved_config.pop("_inputMappings", [])
        if not isinstance(input_mappings, list):
            return
        for mapping in input_mappings:
            source_id = mapping.get("sourceId")
            output_field = mapping.get("outputField", "output")
            target_field = mapping.get("targetField")
            if not source_id or not target_field:
                continue
            source_data = block_outputs.get(source_id)
            if source_data is None:
                continue
            value = _resolve_output_path(source_data, output_field)
            # Backward compat: if field's root isn't "output", try prepending it
            if value is None and output_field.split(".")[0] != "output":
                value = _resolve_output_path(source_data, f"output.{output_field}")
            if value is None:
                continue
            # Skip if the user manually specified a value for this field
            existing = resolved_config.get(target_field)
            if existing is not None and existing != "":
                continue
            resolved_config[target_field] = value

    async def execute(self) -> AsyncGenerator[ExecutionEvent]:
        """Execute all blocks, yielding events as they occur."""
        order = self._topological_order()
        block_outputs: dict[str, Any] = {}

        for block_id in order:
            bdef = self._blocks[block_id]

            if not self._should_execute_block(block_id, block_outputs):
                logger.debug("Skipping block %s (condition not met)", block_id)
                continue

            yield ExecutionEvent(
                type="block_start", block_id=block_id, block_type=bdef.type
            )

            start_time = time.monotonic()

            try:
                block = BlockRegistry.create(bdef.type, config=bdef.config)
                # Preserve raw expression for condition blocks so their
                # own <ref> resolver handles type-safe substitution.
                raw_expression = (
                    bdef.config.get("expression")
                    if bdef.type == "condition"
                    else None
                )
                raw_branches = (
                    bdef.config.get("branches")
                    if bdef.type == "condition"
                    else None
                )
                raw_code = (
                    bdef.config.get("code")
                    if bdef.type in ("code", "function")
                    else None
                )
                resolved_config = resolve_value(bdef.config, block_outputs)

                self._apply_input_mappings(resolved_config, block_outputs)

                # Restore raw expression/code/branches AFTER input mappings
                # so stale mappings targeting these fields can't overwrite them.
                if raw_expression is not None:
                    resolved_config["expression"] = raw_expression
                if raw_branches is not None:
                    resolved_config["branches"] = raw_branches
                if raw_code is not None:
                    resolved_config["code"] = raw_code

                block.config = resolved_config

                block_input = BlockInput(
                    data=self._variables,
                    block_outputs=block_outputs,
                    services=self._services,
                )

                output = await block.execute(block_input)
                block_outputs[block_id] = output.data
                # Also index by block name so <BlockName.output> refs resolve
                if bdef.name and bdef.name not in block_outputs:
                    block_outputs[bdef.name] = output.data
                elapsed_ms = (time.monotonic() - start_time) * 1000

                event = ExecutionEvent(
                    type="block_complete",
                    block_id=block_id,
                    block_type=bdef.type,
                    data=output.data,
                    duration_ms=round(elapsed_ms, 2),
                )
                yield event

            except Exception as e:
                elapsed_ms = (time.monotonic() - start_time) * 1000
                logger.error("Block %s failed: %s", block_id, e, exc_info=True)
                error_event = ExecutionEvent(
                    type="block_error",
                    block_id=block_id,
                    block_type=bdef.type,
                    data={"error": str(e)},
                    duration_ms=round(elapsed_ms, 2),
                )
                yield error_event
                # Don't propagate — downstream blocks simply won't execute

    async def execute_streaming(self) -> AsyncGenerator[ExecutionEvent]:
        """Execute with streaming support for agent blocks.

        Yields ExecutionEvent objects. Agent blocks emit ``block_chunk``
        events for each streamed token.
        """
        order = self._topological_order()
        block_outputs: dict[str, Any] = {}

        for block_id in order:
            bdef = self._blocks[block_id]

            if not self._should_execute_block(block_id, block_outputs):
                logger.debug("Skipping block %s (condition not met)", block_id)
                continue

            yield ExecutionEvent(
                type="block_start", block_id=block_id, block_type=bdef.type
            )

            start_time = time.monotonic()

            try:
                block = BlockRegistry.create(bdef.type, config=bdef.config)
                # Preserve raw expression/code/branches so their own resolvers
                # handle type-safe substitution.
                raw_expression = (
                    bdef.config.get("expression")
                    if bdef.type == "condition"
                    else None
                )
                raw_branches = (
                    bdef.config.get("branches")
                    if bdef.type == "condition"
                    else None
                )
                raw_code = (
                    bdef.config.get("code")
                    if bdef.type in ("code", "function")
                    else None
                )
                resolved_config = resolve_value(bdef.config, block_outputs)

                self._apply_input_mappings(resolved_config, block_outputs)

                # Restore raw expression/code/branches AFTER input mappings
                # so stale mappings targeting these fields can't overwrite them.
                if raw_expression is not None:
                    resolved_config["expression"] = raw_expression
                if raw_branches is not None:
                    resolved_config["branches"] = raw_branches
                if raw_code is not None:
                    resolved_config["code"] = raw_code

                block.config = resolved_config

                block_input = BlockInput(
                    data=self._variables,
                    block_outputs=block_outputs,
                    services=self._services,
                )

                final_output = None
                async for item in block.stream(block_input):
                    if isinstance(item, str):
                        yield ExecutionEvent(
                            type="block_chunk",
                            block_id=block_id,
                            block_type=bdef.type,
                            data=item,
                        )
                    elif isinstance(item, BlockOutput):
                        final_output = item

                elapsed_ms = (time.monotonic() - start_time) * 1000

                if final_output:
                    block_outputs[block_id] = final_output.data
                    if bdef.name and bdef.name not in block_outputs:
                        block_outputs[bdef.name] = final_output.data
                    yield ExecutionEvent(
                        type="block_complete",
                        block_id=block_id,
                        block_type=bdef.type,
                        data=final_output.data,
                        duration_ms=round(elapsed_ms, 2),
                    )
                else:
                    # Fallback: stream yielded no BlockOutput — run execute()
                    output = await block.execute(block_input)
                    block_outputs[block_id] = output.data
                    if bdef.name and bdef.name not in block_outputs:
                        block_outputs[bdef.name] = output.data
                    elapsed_ms = (time.monotonic() - start_time) * 1000
                    yield ExecutionEvent(
                        type="block_complete",
                        block_id=block_id,
                        block_type=bdef.type,
                        data=output.data,
                        duration_ms=round(elapsed_ms, 2),
                    )

            except Exception as e:
                elapsed_ms = (time.monotonic() - start_time) * 1000
                logger.error("Block %s failed: %s", block_id, e, exc_info=True)
                yield ExecutionEvent(
                    type="block_error",
                    block_id=block_id,
                    block_type=bdef.type,
                    data={"error": str(e)},
                    duration_ms=round(elapsed_ms, 2),
                )
