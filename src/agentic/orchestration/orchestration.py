"""
Orchestration - multi-agent coordination patterns.

Status: Not yet implemented.
"""

from typing import Any


class Orchestration:
    """
    Multi-agent coordination.
    
    Orchestration coordinates multiple agents using one of four modes:
    - Sequential: Agents run one after another, passing output forward
    - Supervisor: A coordinator agent delegates tasks to specialists
    - Router: Routes input to a single best agent based on criteria
    - Parallel: Multiple agents run simultaneously on the same input
    
    Status: Not yet implemented.
    
    Planned Example:
        >>> from agentic import Agent, Orchestration
        >>> 
        >>> researcher = Agent(name="researcher", ...)
        >>> writer = Agent(name="writer", ...)
        >>> 
        >>> orch = Orchestration(
        ...     mode="sequential",
        ...     agents=[researcher, writer],
        ... )
        >>> output = orch.run("Write an article about AI")
    
    See Also:
        - docs/CONCEPTS.md for detailed documentation on orchestration modes
    """
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize an Orchestration.
        
        Raises:
            NotImplementedError: Orchestration is not yet implemented.
        """
        raise NotImplementedError(
            "Orchestration is not yet implemented. "
            "See docs/CONCEPTS.md for planned features and roadmap."
        )
    
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the orchestration. Not yet implemented."""
        raise NotImplementedError("Orchestration.run() is not yet implemented.")
    
    async def arun(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the orchestration asynchronously. Not yet implemented."""
        raise NotImplementedError("Orchestration.arun() is not yet implemented.")

