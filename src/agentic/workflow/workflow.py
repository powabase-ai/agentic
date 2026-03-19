"""
Workflow — DAG-based pipeline execution with multiple block types.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agentic.workflow.engine import (
    BlockDefinition,
    DAGEngine,
    EdgeDefinition,
    ExecutionEvent,
)

logger = logging.getLogger(__name__)

# Ensure block types are registered on import
import agentic.workflow.blocks  # noqa: F401


class Workflow:
    """
    DAG-based workflow that executes a graph of blocks.

    Example:
        >>> wf = Workflow(name="my-workflow")
        >>> wf.add_block("start", "starter", "Start")
        >>> wf.add_block("agent1", "agent", "Agent", config={"model": "gpt-4o-mini", "prompt": "<start.output>"})
        >>> wf.add_block("out", "response", "Output", config={"output": "<agent1.output>"})
        >>> wf.add_edge("start", "agent1")
        >>> wf.add_edge("agent1", "out")
        >>> result = wf.run(variables={"input": "Hello"})
    """

    def __init__(
        self,
        name: str = "workflow",
        description: str = "",
        variables: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.variables = variables or {}
        self._blocks: list[BlockDefinition] = []
        self._edges: list[EdgeDefinition] = []
        self._block_ids: set[str] = set()
        self._edge_counter = 0

    def add_block(
        self,
        block_id: str,
        block_type: str,
        name: str = "",
        config: dict[str, Any] | None = None,
        enabled: bool = True,
    ) -> None:
        if block_id in self._block_ids:
            raise ValueError(f"Duplicate block id: {block_id}")
        self._blocks.append(
            BlockDefinition(
                id=block_id,
                type=block_type,
                name=name or block_id,
                config=config or {},
                enabled=enabled,
            )
        )
        self._block_ids.add(block_id)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        source_handle: str = "output",
        target_handle: str = "input",
        condition: str | None = None,
    ) -> None:
        self._edge_counter += 1
        self._edges.append(
            EdgeDefinition(
                id=f"edge-{self._edge_counter}",
                source_block_id=source_id,
                target_block_id=target_id,
                source_handle=source_handle,
                target_handle=target_handle,
                condition=condition,
            )
        )

    @classmethod
    def from_graph(
        cls,
        name: str,
        blocks: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        variables: dict[str, Any] | None = None,
        description: str = "",
    ) -> Workflow:
        """Create a Workflow from serialized block/edge dicts (e.g. from DB)."""
        wf = cls(name=name, description=description, variables=variables)
        for b in blocks:
            wf.add_block(
                block_id=b["id"],
                block_type=b["type"],
                name=b.get("name", ""),
                config=b.get("config", {}),
                enabled=b.get("enabled", True),
            )
        for e in edges:
            wf.add_edge(
                source_id=e["source_block_id"],
                target_id=e["target_block_id"],
                source_handle=e.get("source_handle", "output"),
                target_handle=e.get("target_handle", "input"),
                condition=e.get("condition"),
            )
        return wf

    def _build_engine(
        self,
        variables: dict[str, Any] | None = None,
        services: dict[str, Any] | None = None,
    ) -> DAGEngine:
        merged_vars = {**self.variables, **(variables or {})}
        return DAGEngine(
            blocks=self._blocks,
            edges=self._edges,
            variables=merged_vars,
            services=services,
        )

    def run(
        self,
        variables: dict[str, Any] | None = None,
        services: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Synchronous execution. Returns dict of block_id -> output data."""
        return asyncio.run(self.arun(variables=variables, services=services))

    async def arun(
        self,
        variables: dict[str, Any] | None = None,
        services: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Async execution. Returns dict of block_id -> output data."""
        results, _ = await self.arun_detailed(
            variables=variables, services=services
        )
        return results

    async def arun_detailed(
        self,
        variables: dict[str, Any] | None = None,
        services: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[ExecutionEvent]]:
        """Async execution returning (results, events) with timing info."""
        engine = self._build_engine(variables=variables, services=services)
        results: dict[str, Any] = {}
        events: list[ExecutionEvent] = []
        async for event in engine.execute():
            events.append(event)
            if event.type in ("block_complete", "block_error"):
                results[event.block_id] = event.data
        return results, events

    async def astream(
        self,
        variables: dict[str, Any] | None = None,
        services: dict[str, Any] | None = None,
    ):
        """Async streaming execution. Yields ExecutionEvent objects."""
        engine = self._build_engine(variables=variables, services=services)
        async for event in engine.execute_streaming():
            yield event
