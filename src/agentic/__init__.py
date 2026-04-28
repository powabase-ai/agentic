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

import importlib as _importlib

from . import llm  # noqa: F401  -- triggers llm/setup.py on cold import
from .llm import setup as _llm_setup

# Re-run setup on every package init (incl. importlib.reload) so the LiteLLM
# global flag is restored even if a caller mutated it after first import.
_importlib.reload(_llm_setup)

# Agent module - fully implemented
from agentic.agent import Agent, AgentOutput, AgentSession, ToolCallRecord  # noqa: E402
from agentic.agent.tools import (  # noqa: E402
    BuiltinTool,
    CustomTool,
    DelegateTool,
    KnowledgeSearchTool,
    McpTool,
    ToolDefinition,
)

# Execution infrastructure
from agentic.execution import (  # noqa: E402
    ExecutionContext,
    ExecutionStatus,
    MaxDepthExceeded,
    TokenBudget,
)

# Ingest module - content ingestion (connectors & extractors)
from agentic.ingest import (  # noqa: E402
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
from agentic.knowledge import (  # noqa: E402
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

# Orchestration module
from agentic.orchestration import (  # noqa: E402
    Orchestration,
    OrchestrationEntity,
    OrchestrationOutput,
    OrchestrationSession,
    StrategyEngine,
    SupervisorEngine,
    get_strategy_engine,
)

# Workflow module - placeholder
from agentic.workflow import Workflow, WorkflowOutput, WorkflowSession  # noqa: E402

__version__ = "0.1.0"

__all__ = [
    # Agent (implemented)
    "Agent",
    "AgentOutput",
    "AgentSession",
    "ToolCallRecord",
    # Tools (implemented)
    "ToolDefinition",
    "BuiltinTool",
    "CustomTool",
    "DelegateTool",
    "KnowledgeSearchTool",
    "McpTool",
    # Orchestration
    "Orchestration",
    "OrchestrationEntity",
    "OrchestrationOutput",
    "OrchestrationSession",
    "StrategyEngine",
    "SupervisorEngine",
    "get_strategy_engine",
    # Workflow (placeholder)
    "Workflow",
    "WorkflowOutput",
    "WorkflowSession",
    # Execution infrastructure
    "ExecutionContext",
    "ExecutionStatus",
    "TokenBudget",
    "MaxDepthExceeded",
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
