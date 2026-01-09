"""
Workflow session - state container for workflow execution.

Status: Not yet implemented.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from agentic.workflow.output import WorkflowOutput


@dataclass
class WorkflowSession:
    """
    Container for workflow execution history and state.
    
    WorkflowSession maintains the history of all workflow executions,
    workflow-level state, and embedded orchestration/agent contexts.
    
    Status: Not yet implemented.
    
    Planned Attributes:
        session_id: Unique identifier for this session
        workflow_name: Name of the workflow
        outputs: List of all WorkflowOutputs from this session
        session_state: Workflow-level state that persists across runs
        orchestration_contexts: Per-orchestration state within this workflow
    
    See Also:
        - docs/CONCEPTS.md for detailed documentation
    """
    
    session_id: str = field(default_factory=lambda: str(uuid4()))
    workflow_name: str | None = None
    outputs: list[WorkflowOutput] = field(default_factory=list)
    session_state: dict[str, Any] = field(default_factory=dict)
    orchestration_contexts: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

