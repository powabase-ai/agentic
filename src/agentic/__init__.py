"""
agentic - A minimal, well-documented agent framework.

agentic provides a clean interface for building LLM-powered agents,
multi-agent orchestrations, and complex workflows.

Quick Start:
    >>> from agentic import Agent
    >>>
    >>> agent = Agent(
    ...     model="gpt-4o-mini",
    ...     system_prompt="You are a helpful assistant.",
    ... )
    >>> output = agent.run("Hello!")
    >>> print(output.content)

Core Components:
    - Agent: Single LLM-powered agent
    - AgentOutput: Result of agent execution
    - AgentSession: Conversation history container

    - Orchestration: Multi-agent coordination (not yet implemented)
    - Workflow: Pipeline execution (not yet implemented)

    - ExecutionContext: Runtime context for executions
    - ExecutionStatus: Status enum (PENDING, RUNNING, COMPLETED, etc.)

    - Knowledge: Indexing and retrieval for RAG applications

See Also:
    - README.md for installation and quickstart
    - docs/CONCEPTS.md for detailed concept explanations
"""

# Agent module - fully implemented
from agentic.agent import Agent, AgentOutput, AgentSession
from agentic.agent.tools import BuiltinTool, CustomTool, ToolDefinition

# Execution infrastructure
from agentic.execution import ExecutionContext, ExecutionStatus

# Ingest module - content ingestion (connectors & extractors)
from agentic.ingest import (
    Connector,
    ContentItem,
    Derivative,
    ExtractionResult,
    Extractor,
    ExtractorRegistry,
    FileUploadConnector,
    RawContent,
)

# Knowledge module - base classes and interfaces
from agentic.knowledge import (
    ChunkingStrategy,
    Embedder,
    IndexingAlgorithm,
    IndexingConfig,
    IndexResult,
    KnowledgeStore,
    RetrievalAlgorithm,
    RetrievalConfig,
    RetrievedItem,
)

# Orchestration module - placeholder
from agentic.orchestration import (
    Orchestration,
    OrchestrationOutput,
    OrchestrationSession,
)

# Workflow module - placeholder
from agentic.workflow import Workflow, WorkflowOutput, WorkflowSession

__version__ = "0.1.0"

__all__ = [
    # Agent (implemented)
    "Agent",
    "AgentOutput",
    "AgentSession",
    # Tools (implemented)
    "ToolDefinition",
    "BuiltinTool",
    "CustomTool",
    # Orchestration (placeholder)
    "Orchestration",
    "OrchestrationOutput",
    "OrchestrationSession",
    # Workflow (placeholder)
    "Workflow",
    "WorkflowOutput",
    "WorkflowSession",
    # Execution infrastructure
    "ExecutionContext",
    "ExecutionStatus",
    # Knowledge (base classes)
    "KnowledgeStore",
    "IndexingAlgorithm",
    "RetrievalAlgorithm",
    "ChunkingStrategy",
    "Embedder",
    "IndexResult",
    "RetrievedItem",
    "IndexingConfig",
    "RetrievalConfig",
    # Ingest (connectors & extractors)
    "RawContent",
    "Derivative",
    "ExtractionResult",
    "ContentItem",
    "Connector",
    "FileUploadConnector",
    "Extractor",
    "ExtractorRegistry",
]
