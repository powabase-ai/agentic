# agentic

The agent/knowledge/orchestration/workflow engine that powers the Powabase stack — a well-documented Python library you can also import on its own.

## Features

- **Agent**: Single LLM-powered agent with a clean run() interface
- **Session Management**: Built-in conversation history tracking
- **Multi-Model Support**: Uses [litellm](https://github.com/BerriAI/litellm) for 100+ LLM providers
- **Type-Safe**: Full type hints and dataclass-based outputs
- **Orchestration**: Multi-agent coordination (sequential, supervisor, router, parallel)
- **Workflow**: Multi-step pipelines with conditional logic, loops, and a sandboxed code-execution block

## Installation

```bash
# Using pip
pip install agentic

# Using uv
uv add agentic

# From source
git clone https://github.com/your-org/agentic.git
cd agentic
pip install -e .
```

## Quick Start

```python
from agentic import Agent

# Create an agent
agent = Agent(
    model="gpt-4o-mini",
    system_prompt="You are a helpful assistant.",
)

# Run the agent
output = agent.run("What is 2+2?")
print(output.content)  # "2 + 2 equals 4."

# Check execution details
print(output.status)  # ExecutionStatus.COMPLETED
print(output.usage)   # {"prompt_tokens": 15, "completion_tokens": 8, ...}
```

## Multi-Turn Conversations

Use `AgentSession` to maintain conversation history:

```python
from agentic import Agent, AgentSession

agent = Agent(
    model="gpt-4o-mini",
    system_prompt="You are a helpful assistant.",
)

# Create a session
session = AgentSession()

# First turn
output1 = agent.run("My name is Alice", session=session)
session.add_output(output1)

# Second turn - agent remembers the name
output2 = agent.run("What's my name?", session=session)
session.add_output(output2)
print(output2.content)  # "Your name is Alice."

# Get conversation history
messages = session.get_messages()
```

## Async Support

```python
import asyncio
from agentic import Agent

async def main():
    agent = Agent(model="gpt-4o-mini", system_prompt="You are helpful.")
    output = await agent.arun("Hello!")
    print(output.content)

asyncio.run(main())
```

## Model Selection

agentic uses [litellm](https://github.com/BerriAI/litellm) under the hood, supporting 100+ LLM providers:

```python
# OpenAI
agent = Agent(model="gpt-4o")

# Anthropic
agent = Agent(model="claude-3-opus-20240229")

# Azure OpenAI
agent = Agent(model="azure/my-deployment")

# Local models (Ollama)
agent = Agent(model="ollama/llama2")
```

Set up API keys via environment variables:
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Core Concepts

### Agent

An `Agent` wraps an LLM with a system prompt. It provides a simple `run()` method that handles message formatting, LLM calls, and response parsing.

### AgentOutput

The result of `agent.run()`. Contains:
- `content`: The LLM's response text
- `status`: Execution status (COMPLETED, FAILED, etc.)
- `messages`: All messages exchanged
- `usage`: Token usage statistics
- Timing information (`started_at`, `completed_at`)

### AgentSession

Container for conversation history. Pass a session to `run()` to include previous messages in the context.

### ExecutionContext

Runtime context carrying `execution_id`, `session_id`, and custom metadata. Created automatically or can be provided explicitly.

See [docs/CONCEPTS.md](docs/CONCEPTS.md) for detailed documentation.

## Project Structure

```
agentic/src/agentic/
├── agent/           # Agent implementation
│   ├── agent.py     # Agent class
│   ├── output.py    # AgentOutput dataclass
│   └── session.py   # AgentSession for history
├── orchestration/   # Multi-agent coordination (sequential, supervisor, router, parallel)
├── workflow/        # Pipeline execution (blocks, conditions, sandboxed functions)
└── execution/       # Shared infrastructure
    ├── context.py   # ExecutionContext
    ├── status.py    # ExecutionStatus enum
    └── base.py      # BaseOutput class
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/agentic
```

## License

Apache-2.0 — see [LICENSE](LICENSE).

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit a pull request.

