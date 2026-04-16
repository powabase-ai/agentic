"""
Agent - single LLM-powered agent definition and execution.
"""

import json
import logging
import threading
import time
from collections.abc import AsyncGenerator, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import litellm

from agentic.agent.cache import sort_tools_for_cache
from agentic.agent.compaction import (
    compact_messages,
    estimate_token_count,
    get_context_threshold,
    prune_messages,
)
from agentic.agent.errors import classify_error, classify_finish_reason
from agentic.agent.loop_state import LoopState
from agentic.agent.normalization import normalize_messages
from agentic.agent.output import AgentOutput, ToolCallRecord
from agentic.agent.session import AgentSession
from agentic.agent.tools import ToolDefinition
from agentic.execution.context import ExecutionContext
from agentic.execution.status import ExecutionStatus
from agentic.knowledge.model_config import AGENT_DEFAULT_MODEL

logger = logging.getLogger(__name__)


def _build_tool_message(tc_id: str, result) -> tuple[dict, dict | None]:
    """Build a tool result message, handling multimodal content.

    LLM APIs require role: "tool" content to be a string. When tool results
    contain multimodal content (list[dict] with image_url blocks), we split
    them: text goes in the tool message, images are returned as a separate
    user message to be appended *after* all tool messages (APIs require all
    tool responses to be consecutive).

    Returns (tool_message, optional_user_message).
    """
    if isinstance(result, list):
        text_parts = [
            item.get("text", "")
            for item in result
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        tool_msg = {
            "role": "tool",
            "tool_call_id": tc_id,
            "content": "\n".join(text_parts)
            if text_parts
            else "[Retrieved multimodal content]",
        }
        image_blocks = [
            item
            for item in result
            if isinstance(item, dict) and item.get("type") == "image_url"
        ]
        if image_blocks:
            user_msg = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here are the retrieved images from the knowledge base search:",
                    },
                    *image_blocks,
                ],
            }
            return tool_msg, user_msg
        return tool_msg, None
    else:
        return {
            "role": "tool",
            "tool_call_id": tc_id,
            "content": result,
        }, None


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
        temperature: float | None = None,
        max_tokens: int | None = None,
        api_key: str | None = None,
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
            temperature: Sampling temperature (0-2). None uses the model default.
            max_tokens: Maximum tokens to generate. None uses the model default.
            api_key: Optional API key for the model's provider. Passed directly
                     to litellm, overriding env vars. Enables thread-safe
                     concurrent requests without os.environ mutation.
        """
        self.model = model
        self.system_prompt = system_prompt
        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key

    def run(
        self,
        input: str | list[dict[str, Any]],
        session: AgentSession | None = None,
        context: ExecutionContext | None = None,
        tools: dict[str, ToolDefinition] | None = None,
        max_steps: int = 25,
        fallback_model: str | None = None,
        tool_rules: dict[str, list[dict]] | None = None,
        hooks: list | None = None,
        response_format: dict | None = None,
        timeout_seconds: int | float | None = None,
    ) -> AgentOutput:
        """
        Execute the agent with the given input using a ReAct loop.

        When tools are provided, the agent enters a ReAct (Reason + Act) loop:
        the LLM can request tool calls, receive results, and iterate until it
        produces a final text response or hits the max step limit.

        Without tools, behaves identically to a single LLM call (backward compat).

        The loop is structured as a 5-phase pipeline per iteration:
          Phase 1 (Setup) — abort check, normalize messages, emit step_started
          Phase 2 (LLM Call) — call litellm, accumulate usage, enforce budget
          Phase 3 (Decision) — stop if no tool calls or last step
          Phase 4 (Tool Execution) — execute tools, record results
          Phase 5 (Continuation) — compaction, doom loop check, advance state

        Args:
            input: The user input. Can be a string or a list of message dicts
                   in OpenAI format [{"role": "user", "content": "..."}]
            session: Optional session for multi-turn conversations. If provided,
                     previous messages from the session will be included.
            context: Optional execution context. If not provided, a new one
                     will be created with a generated execution_id.
            tools: Optional dict of tool name -> ToolDefinition for the agent
                   to use during execution.
            max_steps: Maximum number of LLM calls before forcing a text-only
                       response. Default: 25.

        Returns:
            AgentOutput containing the response and execution metadata.

        Example:
            >>> output = agent.run("Hello!")
            >>> print(output.content)
            "Hello! How can I help you today?"

            >>> # With tools
            >>> from agentic.agent.tools import BuiltinTool
            >>> tool = BuiltinTool(name="get_time", ...)
            >>> output = agent.run("What time?", tools={"get_time": tool})
        """
        # Create execution context if not provided
        if context is None:
            context = ExecutionContext(
                session_id=session.session_id if session else None,
            )

        started_at = datetime.now()

        # Hard timeout — fire abort_signal after timeout_seconds
        timer: threading.Timer | None = None
        if timeout_seconds:
            if not context.abort_signal:
                context.abort_signal = threading.Event()
            timer = threading.Timer(timeout_seconds, context.abort_signal.set)
            timer.daemon = True
            timer.start()

        try:
            # OnRunStart hook — fire before anything else
            if hooks:
                from agentic.agent.hooks import run_hooks as _run_hooks

                input_str = input if isinstance(input, str) else str(input)
                on_start = _run_hooks(
                    "OnRunStart",
                    "",
                    {"message": input_str},
                    None,
                    hooks,
                    context=context,
                )
                if on_start.blocked:
                    return AgentOutput(
                        execution_id=context.execution_id,
                        status=ExecutionStatus.FAILED,
                        started_at=started_at,
                        completed_at=datetime.now(),
                        error=on_start.message or "Blocked by OnRunStart hook",
                    )

            # Build messages list
            messages = self._build_messages(input, session)

            # Build tool schemas using cache-aware ordering
            tool_schemas = None
            available_tool_names = None
            if tools:
                sorted_tools = sort_tools_for_cache(tools)
                tool_schemas = [t.to_function_schema() for t in sorted_tools]
                available_tool_names = set(tools.keys())

            # Initialize structured loop state
            state = LoopState.initial(messages=messages, model=self.model)

            # Accumulator state (not part of LoopState since these only grow)
            all_tool_calls: list[ToolCallRecord] = []
            collected_events: list[dict] = []
            total_usage: dict[str, int] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            # For doom loop detection: list of (tool_name, arguments_str) tuples
            recent_calls: list[tuple[str, str]] = []

            # Wrap on_event to also collect into local list
            original_on_event = context.on_event

            def collecting_on_event(event: dict) -> None:
                collected_events.append(event)
                if original_on_event:
                    original_on_event(event)

            context.on_event = collecting_on_event

            while True:
                # The "step" as seen by events/output is turn_count + 1
                # (turn_count starts at 0 and increments in Phase 5)
                step = state.turn_count + 1

                # ===== PHASE 1: SETUP =====
                if context.is_aborted:
                    return AgentOutput(
                        execution_id=context.execution_id,
                        status=ExecutionStatus.CANCELLED,
                        started_at=started_at,
                        completed_at=datetime.now(),
                        error="Execution aborted",
                        messages=list(state.messages),
                        usage=total_usage,
                        steps=step,
                        tool_calls=all_tool_calls,
                        events=collected_events,
                    )

                is_last_step = step >= max_steps or state.budget_exhausted

                # Normalize messages before LLM call
                normalized = normalize_messages(
                    list(state.messages), available_tool_names
                )

                # Proactive context management — prune/compact before LLM call if near threshold
                token_estimate = estimate_token_count(normalized)
                threshold = get_context_threshold(state.current_model)
                if token_estimate > threshold:
                    pruned = prune_messages(normalized)
                    if estimate_token_count(pruned) > threshold:
                        if state.compact_failure_count < 3:
                            try:
                                pruned = compact_messages(pruned)
                            except Exception:
                                logger.warning("Proactive compaction failed")
                    normalized = pruned
                    context.emit_event(
                        {
                            "type": "proactive_compact",
                            "step": step,
                            "tokens_before": token_estimate,
                        }
                    )

                # Emit step_started event
                context.emit_event(
                    {"type": "step_started", "step": step, "is_last_step": is_last_step}
                )

                # ===== PHASE 2: LLM CALL =====
                # On last step, don't pass tools to force a text-only response
                step_tools = None if is_last_step else tool_schemas

                # Build call kwargs
                call_kwargs: dict[str, Any] = {
                    "model": state.current_model,
                    "messages": normalized,
                    "num_retries": 3,
                    "stream": False,
                }
                if self.temperature is not None:
                    call_kwargs["temperature"] = self.temperature
                if self.max_tokens is not None:
                    call_kwargs["max_tokens"] = self.max_tokens
                if step_tools:
                    call_kwargs["tools"] = step_tools
                if response_format is not None:
                    call_kwargs["response_format"] = response_format
                if self.api_key is not None:
                    call_kwargs["api_key"] = self.api_key

                # Call LLM
                try:
                    response = litellm.completion(**call_kwargs)
                except Exception as llm_error:
                    error_type = classify_error(llm_error)

                    # Recovery: model fallback on rate limit
                    if (
                        error_type == "rate_limit"
                        and fallback_model
                        and state.current_model != fallback_model
                    ):
                        context.emit_event(
                            {
                                "type": "model_fallback",
                                "step": step,
                                "from": state.current_model,
                                "to": fallback_model,
                            }
                        )
                        state = state.with_fallback_model(fallback_model)
                        continue

                    # Recovery: reactive compact on prompt too long
                    if (
                        error_type == "prompt_too_long"
                        and not state.has_attempted_compact
                        and state.compact_failure_count < 3
                    ):
                        context.emit_event({"type": "reactive_compact", "step": step})
                        try:
                            pruned = prune_messages(list(state.messages))
                            compacted = compact_messages(pruned)
                            state = state.recover(
                                messages=compacted,
                                reason="reactive_compact",
                                has_attempted_compact=True,
                                compact_failure_count=0,
                            )
                            continue
                        except Exception:
                            state = state.recover(
                                messages=state.messages,
                                reason="compact_failed",
                                compact_failure_count=state.compact_failure_count + 1,
                            )

                    # Recovery: model fallback on server error
                    if (
                        error_type == "model_error"
                        and fallback_model
                        and state.current_model != fallback_model
                    ):
                        context.emit_event(
                            {
                                "type": "model_fallback",
                                "step": step,
                                "from": state.current_model,
                                "to": fallback_model,
                            }
                        )
                        state = state.with_fallback_model(fallback_model)
                        continue

                    # Unrecoverable — re-raise to outer exception handler
                    raise

                # Accumulate usage
                step_usage = self._extract_usage(response)
                if step_usage:
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        total_usage[k] += step_usage.get(k, 0)

                # Budget enforcement: consume tokens and check limits
                if context.budget and step_usage:
                    context.budget.consume(step_usage.get("total_tokens", 0))
                    if context.budget.exceeded:
                        state = state.with_budget_exhausted()
                    elif context.budget.remaining < (context.budget.max_tokens * 0.15):
                        budget_msg = {
                            "role": "system",
                            "content": (
                                f"BUDGET WARNING: You have approximately "
                                f"{context.budget.remaining} tokens remaining. "
                                "Wrap up your work efficiently. Avoid unnecessary "
                                "tool calls."
                            ),
                            "_injected": True,
                        }
                        # Add to working messages (will be picked up via next_turn)
                        working_budget = list(state.messages) + [budget_msg]
                        state = state.recover(
                            messages=working_budget, reason="budget_warning"
                        )

                # Check abort after (potentially slow) LLM call
                if context.is_aborted:
                    return AgentOutput(
                        execution_id=context.execution_id,
                        status=ExecutionStatus.CANCELLED,
                        started_at=started_at,
                        completed_at=datetime.now(),
                        error="Execution aborted",
                        messages=list(state.messages),
                        usage=total_usage,
                        steps=step,
                        tool_calls=all_tool_calls,
                        events=collected_events,
                    )

                assistant_msg = response.choices[0].message
                finish_reason = response.choices[0].finish_reason

                # Determine if the LLM requested tool calls
                has_tool_calls = (
                    tools
                    and assistant_msg.tool_calls
                    and isinstance(assistant_msg.tool_calls, list)
                    and len(assistant_msg.tool_calls) > 0
                )

                # Append assistant message to conversation
                msg_dict: dict[str, Any] = {
                    "role": "assistant",
                    "content": assistant_msg.content,
                }
                if has_tool_calls:
                    msg_dict["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in assistant_msg.tool_calls
                    ]

                working_messages = list(state.messages) + [msg_dict]

                # Emit reasoning content when the LLM explains its
                # thinking alongside tool calls (intermediary reasoning).
                if has_tool_calls and assistant_msg.content:
                    context.emit_event(
                        {
                            "type": "reasoning",
                            "step": step,
                            "content": assistant_msg.content,
                        }
                    )

                # ===== PHASE 3: DECISION =====
                if finish_reason == "stop" or not has_tool_calls or is_last_step:
                    # Check for truncated output (finish_reason = "length" or "max_tokens")
                    output_issue = classify_finish_reason(finish_reason)
                    if output_issue == "max_output_tokens" and not is_last_step:
                        if state.output_recovery_count < 3:
                            recovery_msg = {
                                "role": "user",
                                "content": (
                                    "Your output was truncated. Continue exactly "
                                    "where you left off — no recap, no apology."
                                ),
                                "_injected": True,
                            }
                            context.emit_event(
                                {
                                    "type": "output_recovery",
                                    "step": step,
                                    "attempt": state.output_recovery_count + 1,
                                }
                            )
                            state = state.with_output_recovery(
                                messages=working_messages + [recovery_msg]
                            )
                            continue
                        # 3 attempts exhausted — return partial content (fall through)

                    # Emit step_completed before returning
                    context.emit_event(
                        {
                            "type": "step_completed",
                            "step": step,
                            "finish_reason": finish_reason,
                            "has_tool_calls": has_tool_calls,
                        }
                    )
                    break

                # ===== PHASE 4: TOOL EXECUTION =====
                # Partition into concurrent-safe and exclusive
                concurrent_calls = []
                exclusive_calls = []
                for tc in assistant_msg.tool_calls:
                    tool_name = tc.function.name
                    tool_def = tools.get(tool_name) if tools else None
                    if tool_def and tool_def.is_concurrency_safe:
                        concurrent_calls.append(tc)
                    else:
                        exclusive_calls.append(tc)

                # Collect deferred user messages (multimodal images) across
                # both concurrent and exclusive calls. All tool messages must
                # appear before any user messages — LLM APIs require all tool
                # responses for a given assistant message to be consecutive.
                deferred_image_msgs = []

                # Execute concurrent-safe tools in parallel
                if concurrent_calls:
                    with ThreadPoolExecutor(max_workers=len(concurrent_calls)) as pool:
                        futures = {}
                        for tc in concurrent_calls:
                            futures[
                                pool.submit(
                                    self._execute_single_tool,
                                    tc,
                                    tools,
                                    context,
                                    step,
                                    tool_rules,
                                    hooks,
                                )
                            ] = tc
                        for future in as_completed(futures):
                            tc = futures[future]
                            result, record = future.result()
                            all_tool_calls.append(record)
                            tool_msg, user_msg = _build_tool_message(tc.id, result)
                            working_messages.append(tool_msg)
                            if user_msg:
                                deferred_image_msgs.append(user_msg)
                            recent_calls.append(
                                (tc.function.name, tc.function.arguments)
                            )

                # Execute exclusive tools sequentially
                for tc in exclusive_calls:
                    if context.is_aborted:
                        break
                    result, record = self._execute_single_tool(
                        tc, tools, context, step, tool_rules, hooks
                    )
                    all_tool_calls.append(record)
                    tool_msg, user_msg = _build_tool_message(tc.id, result)
                    working_messages.append(tool_msg)
                    if user_msg:
                        deferred_image_msgs.append(user_msg)
                    recent_calls.append((tc.function.name, tc.function.arguments))

                # Append multimodal image messages after ALL tool messages
                working_messages.extend(deferred_image_msgs)

                # Emit step_completed after all tool calls are done
                context.emit_event(
                    {
                        "type": "step_completed",
                        "step": step,
                        "finish_reason": finish_reason,
                        "has_tool_calls": has_tool_calls,
                    }
                )

                # Check abort after tool execution
                if context.is_aborted:
                    return AgentOutput(
                        execution_id=context.execution_id,
                        status=ExecutionStatus.CANCELLED,
                        started_at=started_at,
                        completed_at=datetime.now(),
                        error="Execution aborted during tool execution",
                        messages=working_messages,
                        usage=total_usage,
                        steps=step,
                        tool_calls=all_tool_calls,
                        events=collected_events,
                    )

                # Compaction check: summarize history if context is growing large
                if estimate_token_count(working_messages) > get_context_threshold(
                    state.current_model
                ):
                    context.emit_event(
                        {
                            "type": "compaction",
                            "step": step,
                            "message_count": len(working_messages),
                        }
                    )
                    working_messages = compact_messages(working_messages)

                # Check for doom loop: last 3 calls identical
                if len(recent_calls) >= 3:
                    last_three = recent_calls[-3:]
                    if last_three[0] == last_three[1] == last_three[2]:
                        return AgentOutput(
                            execution_id=context.execution_id,
                            status=ExecutionStatus.FAILED,
                            started_at=started_at,
                            completed_at=datetime.now(),
                            error="Doom loop detected: 3 identical tool calls in a row",
                            messages=working_messages,
                            usage=total_usage,
                            steps=step,
                            tool_calls=all_tool_calls,
                            events=collected_events,
                        )

                # ===== PHASE 5: CONTINUATION =====
                state = state.next_turn(messages=working_messages)

            # Build final output
            content = assistant_msg.content

            # PreResponse hook: allow hooks to inspect/block/modify final output
            if hooks and content:
                from agentic.agent.hooks import run_hooks

                pre_response = run_hooks(
                    "PreResponse", "", {}, content, hooks, context=context
                )
                if pre_response.blocked:
                    content = (
                        f"[Response blocked: "
                        f"{pre_response.message or 'PreResponse hook denied'}]"
                    )
                elif pre_response.modified_output:
                    content = pre_response.modified_output

            # OnRunComplete hook — fire-and-forget after successful run
            if hooks:
                from agentic.agent.hooks import run_hooks as _run_hooks_complete

                context.emit_event(
                    {"type": "on_run_complete", "status": "completed", "steps": step}
                )
                _run_hooks_complete(
                    "OnRunComplete",
                    "",
                    {"status": "completed", "content": content or "", "steps": step},
                    None,
                    hooks,
                    context=context,
                )

            return AgentOutput(
                execution_id=context.execution_id,
                status=ExecutionStatus.COMPLETED,
                started_at=started_at,
                completed_at=datetime.now(),
                content=content,
                messages=working_messages,
                usage=total_usage,
                steps=step,
                tool_calls=all_tool_calls,
                events=collected_events,
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
        finally:
            if timer is not None:
                timer.cancel()

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
                **(
                    {"temperature": self.temperature}
                    if self.temperature is not None
                    else {}
                ),
                **(
                    {"max_tokens": self.max_tokens}
                    if self.max_tokens is not None
                    else {}
                ),
                **({"api_key": self.api_key} if self.api_key is not None else {}),
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
                **(
                    {"temperature": self.temperature}
                    if self.temperature is not None
                    else {}
                ),
                **(
                    {"max_tokens": self.max_tokens}
                    if self.max_tokens is not None
                    else {}
                ),
                **({"api_key": self.api_key} if self.api_key is not None else {}),
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
            **(
                {"temperature": self.temperature}
                if self.temperature is not None
                else {}
            ),
            **({"max_tokens": self.max_tokens} if self.max_tokens is not None else {}),
            **({"api_key": self.api_key} if self.api_key is not None else {}),
        )

        # Yield chunks as they arrive
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _execute_single_tool(
        self, tc, tools, context, step, tool_rules=None, hooks=None
    ):
        """Execute a single tool call. Returns (result_str, ToolCallRecord).

        Thread-safe: only reads from `tools` dict and `tc` (both immutable
        during execution). `context.emit_event()` is GIL-protected.
        """
        tool_name = tc.function.name
        arguments_str = tc.function.arguments
        try:
            arguments = json.loads(arguments_str) if arguments_str else {}
        except json.JSONDecodeError:
            arguments = {}

        context.emit_event(
            {
                "type": "tool_call",
                "step": step,
                "tool_name": tool_name,
                "arguments": arguments,
                "call_id": tc.id,
            }
        )

        tool_start = time.monotonic()
        tool_def = tools.get(tool_name) if tools else None

        # === Pre-execution pipeline ===
        if tool_def:
            # Step 1: Schema-level input validation
            valid, validation_msg = tool_def.validate_input(arguments)
            if not valid:
                result = f"Error: Input validation failed — {validation_msg}"
                tool_duration_ms = int((time.monotonic() - tool_start) * 1000)
                return self._finish_tool(
                    result,
                    tool_def,
                    tool_name,
                    arguments,
                    step,
                    context,
                    tc,
                    tool_duration_ms,
                )

            # Step 2: Evaluate runtime rules
            if tool_rules:
                rules_for_tool = tool_rules.get(tool_name, [])
                if rules_for_tool:
                    from agentic.agent.rules import evaluate_rules

                    allowed, reason = evaluate_rules(arguments, rules_for_tool)
                    if not allowed:
                        result = f"Error: Blocked by rule — {reason}"
                        tool_duration_ms = int((time.monotonic() - tool_start) * 1000)
                        return self._finish_tool(
                            result,
                            tool_def,
                            tool_name,
                            arguments,
                            step,
                            context,
                            tc,
                            tool_duration_ms,
                        )

            # Step 3: Run PreToolUse hooks
            if hooks:
                from agentic.agent.hooks import run_hooks

                pre_result = run_hooks(
                    "PreToolUse", tool_name, arguments, None, hooks, context=context
                )
                if pre_result.blocked:
                    result = (
                        f"Error: Blocked by hook — "
                        f"{pre_result.message or 'PreToolUse hook denied'}"
                    )
                    tool_duration_ms = int((time.monotonic() - tool_start) * 1000)
                    return self._finish_tool(
                        result,
                        tool_def,
                        tool_name,
                        arguments,
                        step,
                        context,
                        tc,
                        tool_duration_ms,
                    )
                if pre_result.modified_input:
                    arguments = pre_result.modified_input

        # === OnDelegation hook for DelegateTool ===
        if hooks and tool_def:
            from agentic.agent.tools import DelegateTool as _DelegateTool

            if isinstance(tool_def, _DelegateTool):
                from agentic.agent.hooks import run_hooks as _run_hooks_delegation

                delegation_result = _run_hooks_delegation(
                    "OnDelegation", tool_name, arguments, None, hooks, context=context
                )
                if delegation_result.blocked:
                    result = f"Delegation blocked: {delegation_result.message}"
                    tool_duration_ms = int((time.monotonic() - tool_start) * 1000)
                    return self._finish_tool(
                        result,
                        tool_def,
                        tool_name,
                        arguments,
                        step,
                        context,
                        tc,
                        tool_duration_ms,
                    )

        # === Execute the tool ===
        if tool_def:
            try:
                result = tool_def.execute(arguments, context)
            except Exception as tool_err:
                result = f"Error: {tool_err}"
        else:
            result = f"Error: Unknown tool '{tool_name}'"
        tool_duration_ms = int((time.monotonic() - tool_start) * 1000)

        # === Post-execution: Run PostToolUse hooks ===
        if hooks and tool_def:
            from agentic.agent.hooks import run_hooks

            post_result = run_hooks(
                "PostToolUse", tool_name, arguments, result, hooks, context=context
            )
            if post_result.modified_output:
                result = post_result.modified_output

        # Safety cap on result size (skip for multimodal list results)
        if (
            tool_def
            and tool_def.max_result_chars is not None
            and isinstance(result, str)
            and len(result) > tool_def.max_result_chars
        ):
            original_len = len(result)
            result = (
                result[: tool_def.max_result_chars]
                + f"\n\n[Truncated: {original_len} chars total]"
            )

        record = ToolCallRecord(
            step=step,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            duration_ms=tool_duration_ms,
        )

        # Build result diagnostics for debugging
        if isinstance(result, list):
            block_types = [b.get("type", "unknown") for b in result if isinstance(b, dict)]
            result_meta = {
                "result_type": "multimodal",
                "block_count": len(result),
                "block_types": block_types,
            }
        else:
            result_meta = {
                "result_type": "text",
                "char_count": len(result) if isinstance(result, str) else 0,
            }

        context.emit_event(
            {
                "type": "tool_result",
                "step": step,
                "tool_name": tool_name,
                "call_id": tc.id,
                "result_preview": (result[:200] if isinstance(result, str) else "[multimodal content]") if result else "",
                "result_meta": result_meta,
                "duration_ms": tool_duration_ms,
            }
        )

        return result, record

    def _finish_tool(
        self,
        result,
        tool_def,
        tool_name,
        arguments,
        step,
        context,
        tc,
        tool_duration_ms,
    ):
        """Build ToolCallRecord and emit tool_result event for early returns."""
        record = ToolCallRecord(
            step=step,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            duration_ms=tool_duration_ms,
        )
        context.emit_event(
            {
                "type": "tool_result",
                "step": step,
                "tool_name": tool_name,
                "call_id": tc.id,
                "result_preview": (result[:200] if isinstance(result, str) else "[multimodal content]") if result else "",
                "duration_ms": tool_duration_ms,
            }
        )
        return result, record

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
