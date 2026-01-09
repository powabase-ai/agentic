# Agentic Concepts

This document provides detailed explanations of the core concepts in the agentic framework.

## Table of Contents

1. [Overview](#overview)
2. [Agent](#agent)
3. [Orchestration](#orchestration) (planned)
4. [Workflow](#workflow) (planned)
5. [Execution Model](#execution-model)
6. [Design Philosophy](#design-philosophy)

---

## Overview

agentic is built around three tiers of abstraction:

```
┌─────────────────────────────────────────────────────────────────┐
│                        DEFINITION                                │
│  Static configuration that defines behavior                      │
│  • Agent: model, system_prompt, name                            │
│  • Orchestration: mode, agents (planned)                        │
│  • Workflow: nodes, connections (planned)                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        EXECUTION                                 │
│  Single invocation result                                        │
│  • AgentOutput: content, messages, usage, status                │
│  • OrchestrationOutput (planned)                                │
│  • WorkflowOutput (planned)                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         SESSION                                  │
│  Stateful container persisting across executions                 │
│  • AgentSession: outputs[], state                               │
│  • OrchestrationSession (planned)                               │
│  • WorkflowSession (planned)                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Agent

An **Agent** is the fundamental building block of agentic. It wraps an LLM with a system prompt and provides a clean interface for execution.

### Definition

```python
from agentic import Agent

agent = Agent(
    model="gpt-4o-mini",        # LLM model identifier
    system_prompt="You are a helpful assistant.",  # Behavior instructions
    name="assistant",           # Optional name for logging
)
```

**Attributes:**
- `model`: LLM model identifier (supports any litellm-compatible model)
- `system_prompt`: Instructions sent as the system message
- `name`: Optional identifier for logging and debugging

### Execution

Call `run()` or `arun()` to execute the agent:

```python
# Synchronous
output = agent.run("Hello!")

# Asynchronous
output = await agent.arun("Hello!")
```

**Input Types:**
- String: `agent.run("Hello!")` 
- Message list: `agent.run([{"role": "user", "content": "Hello!"}])`

### AgentOutput

Every execution returns an `AgentOutput`:

```python
@dataclass
class AgentOutput(BaseOutput):
    # Inherited from BaseOutput
    execution_id: str          # Unique ID for this execution
    status: ExecutionStatus    # COMPLETED, FAILED, etc.
    started_at: datetime       # When execution started
    completed_at: datetime     # When execution finished
    error: str | None          # Error message if failed
    
    # Agent-specific
    content: str | None        # The LLM's response
    messages: list[dict]       # All messages exchanged
    usage: dict | None         # Token usage stats
```

### AgentSession

`AgentSession` maintains conversation history across multiple runs:

```python
from agentic import Agent, AgentSession

agent = Agent(model="gpt-4o-mini", system_prompt="You are helpful.")
session = AgentSession()

# First turn
output1 = agent.run("My name is Alice", session=session)
session.add_output(output1)

# Second turn - includes previous messages
output2 = agent.run("What's my name?", session=session)
# Agent responds: "Your name is Alice."
```

**Session Methods:**
- `add_output(output)`: Add an execution output to history
- `get_messages(limit=None)`: Get conversation messages
- `get_last_output()`: Get most recent output
- `clear()`: Reset session history

---

## Orchestration

> **Status: Not yet implemented**

**Orchestration** coordinates multiple agents using one of four modes:

### Planned Modes

1. **Sequential**: Agents run one after another, each building on previous output
   ```
   Agent A → Agent B → Agent C → Final Output
   ```

2. **Supervisor**: A coordinator agent delegates to specialists
   ```
   Supervisor ←→ [Agent A, Agent B, Agent C]
   ```

3. **Router**: Routes to a single best agent based on input
   ```
   Input → Router → Selected Agent → Output
   ```

4. **Parallel**: Multiple agents run simultaneously
   ```
   Input → [Agent A, Agent B, Agent C] → Aggregated Output
   ```

### Planned API

```python
from agentic import Agent, Orchestration

researcher = Agent(name="researcher", system_prompt="...")
writer = Agent(name="writer", system_prompt="...")
editor = Agent(name="editor", system_prompt="...")

# Sequential orchestration
orch = Orchestration(
    mode="sequential",
    agents=[researcher, writer, editor],
)
output = orch.run("Write an article about AI")
```

---

## Workflow

> **Status: Not yet implemented**

**Workflow** defines complex pipelines with multiple node types:

### Planned Features

- **Node Types**:
  - Agent nodes (single agent execution)
  - Orchestration nodes (multi-agent coordination)
  - Function nodes (data transformation)
  - Conditional nodes (if/else branching)
  - Loop nodes (iterative execution)

- **Triggers**: Events that start workflow execution
  - Manual trigger
  - Scheduled trigger
  - Webhook trigger
  - File upload trigger

### Planned API

```python
from agentic import Workflow, Agent

workflow = Workflow(name="document-processor")

# Add nodes
workflow.add_node("extract", agent=extractor_agent)
workflow.add_node("analyze", agent=analyzer_agent)
workflow.add_node("summarize", agent=summarizer_agent)

# Define connections
workflow.connect("extract", "analyze")
workflow.connect("analyze", "summarize")

# Execute
output = workflow.run(document=doc)
```

---

## Execution Model

### ExecutionStatus

All executions go through these states:

```python
class ExecutionStatus(str, Enum):
    PENDING = "pending"       # Created but not started
    RUNNING = "running"       # Currently executing
    COMPLETED = "completed"   # Finished successfully
    FAILED = "failed"         # Encountered an error
    CANCELLED = "cancelled"   # Cancelled before completion
```

### ExecutionContext

Runtime context passed through the execution:

```python
@dataclass
class ExecutionContext:
    execution_id: str          # Unique execution ID
    session_id: str | None     # Parent session if any
    user_id: str | None        # User who initiated
    metadata: dict             # Custom key-value data
```

### BaseOutput

All output types inherit from `BaseOutput`:

```python
@dataclass
class BaseOutput:
    execution_id: str
    status: ExecutionStatus
    started_at: datetime
    completed_at: datetime | None
    error: str | None
```

This provides consistent methods across all output types:
- `is_success()`: Check if completed successfully
- `is_failed()`: Check if failed
- `is_done()`: Check if in terminal state
- `duration_seconds()`: Get execution duration
- `to_dict()`: Serialize to dictionary

---

## Design Philosophy

### 1. Separation of Concerns

We separate **definition** (what it is) from **execution** (what happened) from **session** (state over time):

- **Definition models** (`Agent`, `Orchestration`, `Workflow`) are configuration
- **Output models** (`AgentOutput`, etc.) capture single execution results
- **Session models** (`AgentSession`, etc.) maintain state across executions

### 2. Minimal Core

The core framework includes only essential features:
- Agent + LLM integration
- Session management
- Execution tracking

Tools, memory providers, and other extensions are built on top.

### 3. Type Safety

All models use dataclasses with full type hints:
```python
def run(self, input: str | list[dict]) -> AgentOutput:
    ...
```

### 4. Framework, Not Platform

agentic is a framework for building agents, not a hosted platform. It provides:
- Clean abstractions
- Local execution
- No vendor lock-in

Platforms can be built on top of agentic for persistence, APIs, and multi-tenancy.

---

## Roadmap

1. **v0.1** (current): Agent core with session support
2. **v0.2**: Tools interface and execution
3. **v0.3**: Memory provider system
4. **v0.4**: Orchestration module
5. **v0.5**: Workflow module
6. **v1.0**: Stable API with streaming support

