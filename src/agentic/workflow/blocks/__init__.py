"""Workflow block implementations."""

from agentic.workflow.blocks.agent import AgentBlock
from agentic.workflow.blocks.api_call import APICallBlock
from agentic.workflow.blocks.condition import ConditionBlock
from agentic.workflow.blocks.function import FunctionBlock
from agentic.workflow.blocks.response import ResponseBlock
from agentic.workflow.blocks.platform_api import PlatformAPIBlock
from agentic.workflow.blocks.split import SplitBlock
from agentic.workflow.blocks.starter import StarterBlock
from agentic.workflow.blocks.orchestration import OrchestrationBlock
from agentic.workflow.blocks.webhook import WebhookBlock
from agentic.workflow.block import BlockRegistry

# Auto-register all block types
BlockRegistry.register("starter", StarterBlock)
BlockRegistry.register("agent", AgentBlock)
BlockRegistry.register("code", FunctionBlock)
BlockRegistry.register("function", FunctionBlock)  # backward compat alias
BlockRegistry.register("condition", ConditionBlock)
BlockRegistry.register("general_api", APICallBlock)
BlockRegistry.register("api_call", APICallBlock)  # backward compat alias
BlockRegistry.register("platform_api", PlatformAPIBlock)
BlockRegistry.register("response", ResponseBlock)
BlockRegistry.register("split", SplitBlock)
BlockRegistry.register("webhook", WebhookBlock)
BlockRegistry.register("orchestration", OrchestrationBlock)

__all__ = [
    "StarterBlock",
    "WebhookBlock",
    "AgentBlock",
    "FunctionBlock",
    "ConditionBlock",
    "APICallBlock",
    "PlatformAPIBlock",
    "ResponseBlock",
    "SplitBlock",
    "OrchestrationBlock",
]
