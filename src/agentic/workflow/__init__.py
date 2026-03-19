"""
Workflow module — DAG-based pipeline execution with multiple block types.

Provides:
- Workflow: High-level API for building and running workflows
- DAGEngine: Core execution engine
- Block primitives: BaseBlock, BlockInput, BlockOutput, BlockRegistry
- 7 built-in block types: starter, agent, function, condition, router, api_call, response
- Variable resolver and expression evaluator
"""

from agentic.workflow.block import BaseBlock, BlockInput, BlockOutput, BlockRegistry
from agentic.workflow.engine import (
    BlockDefinition,
    DAGEngine,
    EdgeDefinition,
    ExecutionEvent,
)
from agentic.workflow.expression import evaluate_expression
from agentic.workflow.output import WorkflowOutput
from agentic.workflow.session import WorkflowSession
from agentic.workflow.variable_resolver import resolve_string, resolve_value, resolve_variable
from agentic.workflow.workflow import Workflow

# Import blocks to trigger auto-registration
import agentic.workflow.blocks  # noqa: F401

__all__ = [
    "BaseBlock",
    "BlockDefinition",
    "BlockInput",
    "BlockOutput",
    "BlockRegistry",
    "DAGEngine",
    "EdgeDefinition",
    "ExecutionEvent",
    "Workflow",
    "WorkflowOutput",
    "WorkflowSession",
    "evaluate_expression",
    "resolve_string",
    "resolve_value",
    "resolve_variable",
]
