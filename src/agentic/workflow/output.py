"""
Workflow output - result of a workflow execution.

Status: Not yet implemented.
"""

from dataclasses import dataclass, field
from typing import Any

from agentic.execution.base import BaseOutput


@dataclass
class WorkflowOutput(BaseOutput):
    """
    Result of a single Workflow.run() call.
    
    WorkflowOutput captures the complete result of executing a workflow,
    including outputs from all nodes and workflow-level metadata.
    
    Status: Not yet implemented.
    
    Planned Attributes:
        content: The final workflow output content
        node_outputs: Outputs from each node in execution order
        workflow_metadata: Execution metadata (timing, paths taken, etc.)
    
    See Also:
        - docs/CONCEPTS.md for detailed documentation
    """
    
    content: Any | None = None
    node_outputs: list[Any] = field(default_factory=list)
    workflow_metadata: dict[str, Any] = field(default_factory=dict)

