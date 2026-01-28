"""
Workflow - pipeline execution with multiple nodes.

Status: Not yet implemented.
"""

from typing import Any


class Workflow:
    """
    Pipeline execution with multiple nodes.

    Workflow defines and executes complex pipelines that can include:
    - Agent nodes: Single agent execution
    - Orchestration nodes: Multi-agent coordination
    - Conditional logic: If/else branching
    - Loops: Iterative execution
    - Data transformations: Processing between nodes

    Status: Not yet implemented.

    Planned Example:
        >>> from agentic import Agent, Workflow
        >>>
        >>> workflow = Workflow(
        ...     name="document-processor",
        ...     triggers=["file_upload"],
        ... )
        >>> workflow.add_node("extract", agent=extractor_agent)
        >>> workflow.add_node("analyze", agent=analyzer_agent)
        >>> workflow.connect("extract", "analyze")
        >>>
        >>> output = workflow.run(document=doc)

    See Also:
        - docs/CONCEPTS.md for detailed documentation on workflow design
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize a Workflow.

        Raises:
            NotImplementedError: Workflow is not yet implemented.
        """
        raise NotImplementedError(
            "Workflow is not yet implemented. "
            "See docs/CONCEPTS.md for planned features and roadmap."
        )

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the workflow. Not yet implemented."""
        raise NotImplementedError("Workflow.run() is not yet implemented.")

    async def arun(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the workflow asynchronously. Not yet implemented."""
        raise NotImplementedError("Workflow.arun() is not yet implemented.")
