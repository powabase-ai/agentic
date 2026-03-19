"""Workflow block implementations."""

from agentic.workflow.blocks.agent import AgentBlock
from agentic.workflow.blocks.api_call import APICallBlock
from agentic.workflow.blocks.condition import ConditionBlock
from agentic.workflow.blocks.function import FunctionBlock
from agentic.workflow.blocks.response import ResponseBlock
from agentic.workflow.blocks.router import RouterBlock
from agentic.workflow.blocks.split import SplitBlock
from agentic.workflow.blocks.starter import StarterBlock
from agentic.workflow.block import BlockRegistry

# Auto-register all block types
BlockRegistry.register("starter", StarterBlock)
BlockRegistry.register("agent", AgentBlock)
BlockRegistry.register("function", FunctionBlock)
BlockRegistry.register("condition", ConditionBlock)
BlockRegistry.register("router", RouterBlock)
BlockRegistry.register("api_call", APICallBlock)
BlockRegistry.register("response", ResponseBlock)
BlockRegistry.register("split", SplitBlock)

__all__ = [
    "StarterBlock",
    "AgentBlock",
    "FunctionBlock",
    "ConditionBlock",
    "RouterBlock",
    "APICallBlock",
    "ResponseBlock",
    "SplitBlock",
]
