"""
Agent session - container for conversation history across runs.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from agentic.agent.output import AgentOutput


@dataclass
class AgentSession:
    """
    Container for agent conversation history across multiple runs.

    AgentSession maintains the history of all executions for a given agent,
    allowing for multi-turn conversations and history retrieval.

    Attributes:
        session_id: Unique identifier for this session
        agent_name: Optional name of the agent this session belongs to
        outputs: List of all AgentOutputs from this session
        state: Session-level state that persists across runs
        created_at: When this session was created
        updated_at: When this session was last updated

    Example:
        >>> session = AgentSession(session_id="sess-123", agent_name="assistant")
        >>> output = agent.run("Hello", session=session)
        >>> session.add_output(output)
        >>> len(session.outputs)
        1
        >>> messages = session.get_messages(limit=10)
    """

    session_id: str = field(default_factory=lambda: str(uuid4()))
    agent_name: str | None = None
    outputs: list[AgentOutput] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_output(self, output: AgentOutput) -> None:
        """
        Add an execution output to this session.

        Args:
            output: The AgentOutput to add
        """
        self.outputs.append(output)
        self.updated_at = datetime.now()

    def get_messages(
        self,
        limit: int | None = None,
        include_system: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Get conversation messages from session history.

        Extracts messages from all outputs in chronological order,
        useful for building context for subsequent runs.

        Args:
            limit: Maximum number of messages to return (from most recent)
            include_system: Whether to include system messages

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        all_messages: list[dict[str, Any]] = []

        for output in self.outputs:
            for msg in output.messages:
                if not include_system and msg.get("role") == "system":
                    continue
                all_messages.append(msg)

        if limit is not None and len(all_messages) > limit:
            return all_messages[-limit:]

        return all_messages

    def get_last_output(self) -> AgentOutput | None:
        """
        Get the most recent output from this session.

        Returns:
            The last AgentOutput, or None if session is empty
        """
        if not self.outputs:
            return None
        return self.outputs[-1]

    def clear(self) -> None:
        """Clear all outputs from this session."""
        self.outputs = []
        self.updated_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """
        Convert session to dictionary for serialization.

        Returns:
            Dictionary representation of this session
        """
        return {
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "outputs": [o.to_dict() for o in self.outputs],
            "state": self.state,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
