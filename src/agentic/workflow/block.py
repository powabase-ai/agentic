"""
Block primitives for workflow execution.

Defines BlockInput, BlockOutput, BaseBlock, and BlockRegistry used
by the DAG engine to execute workflow graphs.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BlockInput:
    """Input data passed to a block during execution."""

    data: dict[str, Any] = field(default_factory=dict)
    block_outputs: dict[str, Any] = field(default_factory=dict)
    services: dict[str, Any] = field(default_factory=dict)


@dataclass
class BlockOutput:
    """Result from a block execution."""

    data: dict[str, Any] = field(default_factory=dict)
    status: str = "success"
    error: str | None = None


class BaseBlock(ABC):
    """Abstract base class for all workflow blocks."""

    block_type: str = ""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    async def execute(self, block_input: BlockInput) -> BlockOutput:
        """Execute the block and return output."""
        ...

    async def stream(
        self, block_input: BlockInput
    ) -> AsyncGenerator[str | BlockOutput]:
        """
        Stream output from the block.

        Yields str chunks during streaming, then yields a final BlockOutput.
        Default implementation just calls execute() and yields the result.
        """
        result = await self.execute(block_input)
        yield result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config!r})"


class BlockRegistry:
    """Registry mapping block type strings to BaseBlock subclasses."""

    _registry: dict[str, type[BaseBlock]] = {}

    @classmethod
    def register(cls, block_type: str, block_class: type[BaseBlock]) -> None:
        cls._registry[block_type] = block_class

    @classmethod
    def get(cls, block_type: str) -> type[BaseBlock] | None:
        return cls._registry.get(block_type)

    @classmethod
    def create(
        cls, block_type: str, config: dict[str, Any] | None = None
    ) -> BaseBlock:
        block_class = cls._registry.get(block_type)
        if not block_class:
            raise ValueError(f"Unknown block type: {block_type}")
        return block_class(config=config)

    @classmethod
    def available_types(cls) -> list[str]:
        return list(cls._registry.keys())
