"""
Agent - single LLM-powered agent definition and execution.
"""

import logging
from collections.abc import AsyncGenerator, Generator
from datetime import datetime
from typing import Any

import litellm

logger = logging.getLogger(__name__)

from agentic.agent.output import AgentOutput
from agentic.agent.session import AgentSession
from agentic.execution.context import ExecutionContext
from agentic.execution.status import ExecutionStatus
from agentic.knowledge.model_config import AGENT_DEFAULT_MODEL


class Agent:
    """
    A single LLM-powered agent.

    An Agent wraps an LLM with a system prompt and provides a simple interface
    for running conversations. It can run standalone or be composed into an
    Orchestration or Workflow.

    Attributes:
        model: The LLM model identifier (e.g., "gpt-4o", "claude-3-opus")
        system_prompt: Instructions that define the agent's behavior
        name: Optional name for this agent (useful for logging/debugging)

    Example:
        >>> agent = Agent(
        ...     model="gpt-4o-mini",
        ...     system_prompt="You are a helpful assistant.",
        ...     name="assistant",
        ... )
        >>> output = agent.run("What is 2+2?")
        >>> print(output.content)
        "2 + 2 equals 4."

    See Also:
        - AgentOutput: The result type returned by run()
        - AgentSession: For maintaining conversation history across runs
    """

    def __init__(
        self,
        model: str = AGENT_DEFAULT_MODEL,
        system_prompt: str = "",
        name: str | None = None,
    ) -> None:
        """
        Initialize an Agent.

        Args:
            model: LLM model identifier. Supports any model supported by litellm
                   (OpenAI, Anthropic, etc.). Default: "gpt-4o-mini"
            system_prompt: Instructions that define the agent's behavior.
                          This is sent as the system message to the LLM.
            name: Optional name for this agent. Useful for logging and
                  debugging, especially in multi-agent scenarios.
        """
        self.model = model
        self.system_prompt = system_prompt
        self.name = name

    def run(
        self,
        input: str | list[dict[str, Any]],
        session: AgentSession | None = None,
        context: ExecutionContext | None = None,
    ) -> AgentOutput:
        """
        Execute the agent with the given input.

        Args:
            input: The user input. Can be a string or a list of message dicts
                   in OpenAI format [{"role": "user", "content": "..."}]
            session: Optional session for multi-turn conversations. If provided,
                     previous messages from the session will be included.
            context: Optional execution context. If not provided, a new one
                     will be created with a generated execution_id.

        Returns:
            AgentOutput containing the response and execution metadata.

        Example:
            >>> output = agent.run("Hello!")
            >>> print(output.content)
            "Hello! How can I help you today?"

            >>> # Multi-turn conversation
            >>> session = AgentSession()
            >>> output1 = agent.run("My name is Alice", session=session)
            >>> session.add_output(output1)
            >>> output2 = agent.run("What's my name?", session=session)
            >>> print(output2.content)
            "Your name is Alice."
        """
        # Create execution context if not provided
        if context is None:
            context = ExecutionContext(
                session_id=session.session_id if session else None,
            )

        started_at = datetime.now()

        try:
            # Build messages list
            messages = self._build_messages(input, session)

            # Call LLM via litellm
            response = litellm.completion(
                model=self.model,
                messages=messages,
                num_retries=3,
            )

            # Extract response content
            content = response.choices[0].message.content

            # Extract usage (including extended token details)
            usage = self._extract_usage(response)

            # Add assistant message to messages list
            messages.append({"role": "assistant", "content": content})

            return AgentOutput(
                execution_id=context.execution_id,
                status=ExecutionStatus.COMPLETED,
                started_at=started_at,
                completed_at=datetime.now(),
                content=content,
                messages=messages,
                usage=usage,
            )

        except Exception as e:
            logger.error("Agent execution failed: %s", e, exc_info=True)
            return AgentOutput(
                execution_id=context.execution_id,
                status=ExecutionStatus.FAILED,
                started_at=started_at,
                completed_at=datetime.now(),
                error=str(e),
                messages=self._build_messages(input, session),
            )

    async def arun(
        self,
        input: str | list[dict[str, Any]],
        session: AgentSession | None = None,
        context: ExecutionContext | None = None,
    ) -> AgentOutput:
        """
        Async version of run().

        Execute the agent asynchronously with the given input.

        Args:
            input: The user input. Can be a string or a list of message dicts.
            session: Optional session for multi-turn conversations.
            context: Optional execution context.

        Returns:
            AgentOutput containing the response and execution metadata.

        Example:
            >>> output = await agent.arun("Hello!")
            >>> print(output.content)
        """
        # Create execution context if not provided
        if context is None:
            context = ExecutionContext(
                session_id=session.session_id if session else None,
            )

        started_at = datetime.now()

        try:
            # Build messages list
            messages = self._build_messages(input, session)

            # Call LLM via litellm (async)
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                num_retries=3,
            )

            # Extract response content
            content = response.choices[0].message.content

            # Extract usage (including extended token details)
            usage = self._extract_usage(response)

            # Add assistant message to messages list
            messages.append({"role": "assistant", "content": content})

            return AgentOutput(
                execution_id=context.execution_id,
                status=ExecutionStatus.COMPLETED,
                started_at=started_at,
                completed_at=datetime.now(),
                content=content,
                messages=messages,
                usage=usage,
            )

        except Exception as e:
            logger.error("Agent async execution failed: %s", e, exc_info=True)
            return AgentOutput(
                execution_id=context.execution_id,
                status=ExecutionStatus.FAILED,
                started_at=started_at,
                completed_at=datetime.now(),
                error=str(e),
                messages=self._build_messages(input, session),
            )

    def stream(
        self,
        input: str | list[dict[str, Any]],
        session: AgentSession | None = None,
        context: ExecutionContext | None = None,
    ) -> Generator[str, None, AgentOutput]:
        """
        Stream the agent's response, yielding chunks as they arrive.

        This method returns a generator that yields content chunks (strings)
        as they stream from the LLM. After iteration completes, you can get
        the final AgentOutput by calling `.send(None)` or using the generator's
        return value.

        Args:
            input: The user input. Can be a string or a list of message dicts.
            session: Optional session for multi-turn conversations.
            context: Optional execution context.

        Yields:
            str: Content chunks as they stream from the LLM.

        Returns:
            AgentOutput: The final output after streaming completes.

        Example:
            >>> agent = Agent(system_prompt="You are helpful")
            >>> gen = agent.stream("Tell me a story")
            >>> for chunk in gen:
            ...     print(chunk, end="", flush=True)
            >>> # After iteration, get the final output
            >>> try:
            ...     gen.send(None)
            ... except StopIteration as e:
            ...     output = e.value
            ...     print(f"\\nTotal tokens: {output.total_tokens()}")
        """
        # Create execution context if not provided
        if context is None:
            context = ExecutionContext(
                session_id=session.session_id if session else None,
            )

        started_at = datetime.now()
        messages = self._build_messages(input, session)
        content_chunks: list[str] = []

        try:
            # Call LLM with streaming + usage reporting
            response = litellm.completion(
                model=self.model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                num_retries=3,
            )

            # Yield chunks as they arrive, capturing the final usage chunk
            usage_data = None
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    content_chunks.append(content)
                    yield content
                # The final chunk has usage but an empty choices list
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    usage_data = self._extract_usage(chunk)

            # Build final content
            final_content = "".join(content_chunks)

            # Add assistant message to messages list
            messages.append({"role": "assistant", "content": final_content})

            return AgentOutput(
                execution_id=context.execution_id,
                status=ExecutionStatus.COMPLETED,
                started_at=started_at,
                completed_at=datetime.now(),
                content=final_content,
                messages=messages,
                usage=usage_data,
            )

        except Exception as e:
            logger.error("Agent streaming failed: %s", e, exc_info=True)
            return AgentOutput(
                execution_id=context.execution_id,
                status=ExecutionStatus.FAILED,
                started_at=started_at,
                completed_at=datetime.now(),
                error=str(e),
                messages=messages,
            )

    async def astream(
        self,
        input: str | list[dict[str, Any]],
        session: AgentSession | None = None,
        context: ExecutionContext | None = None,
    ) -> AsyncGenerator[str]:
        """
        Async stream the agent's response, yielding chunks as they arrive.

        This is the async version of stream(). Unlike the sync version,
        this is a pure async generator that yields content chunks.

        Args:
            input: The user input. Can be a string or a list of message dicts.
            session: Optional session for multi-turn conversations.
            context: Optional execution context.

        Yields:
            str: Content chunks as they stream from the LLM.

        Example:
            >>> agent = Agent(system_prompt="You are helpful")
            >>> async for chunk in agent.astream("Tell me a story"):
            ...     print(chunk, end="", flush=True)

        Note:
            To get the final AgentOutput with this method, use astream_with_output()
            or collect chunks manually and call arun() for metadata.
        """
        # Create execution context if not provided
        if context is None:
            context = ExecutionContext(
                session_id=session.session_id if session else None,
            )

        messages = self._build_messages(input, session)

        # Call LLM with streaming (async)
        # Note: acompletion with stream=True returns async generator directly
        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            stream=True,
            num_retries=3,
        )

        # Yield chunks as they arrive
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @staticmethod
    def _extract_usage(response) -> dict[str, int] | None:
        """Extract usage info from a litellm response, including extended token details."""
        if not response.usage:
            return None
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        # Reasoning model details (o1, o3, etc.)
        details = getattr(response.usage, "completion_tokens_details", None)
        if details:
            reasoning = getattr(details, "reasoning_tokens", None)
            if reasoning is not None:
                usage["reasoning_tokens"] = reasoning
        # Prompt cache details
        prompt_details = getattr(response.usage, "prompt_tokens_details", None)
        if prompt_details:
            cached = getattr(prompt_details, "cached_tokens", None)
            if cached is not None:
                usage["cached_tokens"] = cached
        return usage

    def _build_messages(
        self,
        input: str | list[dict[str, Any]],
        session: AgentSession | None,
    ) -> list[dict[str, Any]]:
        """
        Build the messages list for the LLM call.

        Args:
            input: User input (string or message list)
            session: Optional session for history

        Returns:
            List of message dicts in OpenAI format
        """
        messages: list[dict[str, Any]] = []

        # Add system message if provided
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Add session history if provided
        if session:
            history = session.get_messages(include_system=False)
            messages.extend(history)

        # Add current input
        if isinstance(input, str):
            messages.append({"role": "user", "content": input})
        elif isinstance(input, list):
            messages.extend(input)

        return messages

    def __repr__(self) -> str:
        """Return a string representation of the agent."""
        name_part = f", name={self.name!r}" if self.name else ""
        return f"Agent(model={self.model!r}{name_part})"
