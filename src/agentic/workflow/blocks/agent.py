"""Agent block — executes an LLM call using the agentic Agent class."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from agentic.agent.agent import Agent
from agentic.workflow.block import BaseBlock, BlockInput, BlockOutput
from agentic.workflow.variable_resolver import resolve_value

logger = logging.getLogger(__name__)


class AgentBlock(BaseBlock):
    block_type = "agent"

    def _resolve_prompt(self, block_input: BlockInput) -> str:
        # Primary: "input" field (from structured mappings or frontend input sub-block)
        input_raw = self.config.get("input", "")
        if input_raw:
            return str(resolve_value(input_raw, block_input.block_outputs))
        # Legacy fallback: "messages" or "prompt"
        messages_raw = self.config.get("messages", "")
        prompt_raw = self.config.get("prompt", "")
        raw = messages_raw if messages_raw else prompt_raw
        return str(resolve_value(raw, block_input.block_outputs))

    async def _build_system_prompt(self, block_input: BlockInput) -> str:
        import asyncio

        system_prompt = self.config.get("system_prompt", "")
        system_prompt = str(resolve_value(system_prompt, block_input.block_outputs))

        # Retrieve KB context if knowledge bases are configured
        kb_configs = self.config.get("knowledge_bases", [])
        retrieve_fn = block_input.services.get("retrieve_kb_context")
        if kb_configs and retrieve_fn:
            prompt_text = self._resolve_prompt(block_input)
            try:
                # Support both sync and async retrieve functions
                if asyncio.iscoroutinefunction(retrieve_fn):
                    rag_context = await retrieve_fn(
                        query=prompt_text, knowledge_base_configs=kb_configs
                    )
                else:
                    rag_context = retrieve_fn(
                        query=prompt_text, knowledge_base_configs=kb_configs
                    )
                if rag_context:
                    system_prompt = f"{system_prompt}\n\nContext:\n{rag_context}"
            except Exception:
                logger.warning("KB retrieval failed", exc_info=True)

        return system_prompt

    def _build_agent(self, model: str, system_prompt: str) -> Agent:
        temperature = self.config.get("temperature")
        max_tokens = self.config.get("max_tokens")

        parsed_temp = None
        if temperature is not None and temperature != "":
            try:
                parsed_temp = float(temperature)
            except (ValueError, TypeError):
                pass

        parsed_max = None
        if max_tokens is not None and max_tokens != "":
            try:
                parsed_max = int(float(max_tokens))
            except (ValueError, TypeError):
                pass

        return Agent(
            model=model,
            system_prompt=system_prompt,
            temperature=parsed_temp,
            max_tokens=parsed_max,
        )

    async def execute(self, block_input: BlockInput) -> BlockOutput:
        model = self.config.get("model", "gpt-4o-mini")
        system_prompt = await self._build_system_prompt(block_input)
        prompt = self._resolve_prompt(block_input)

        agent = self._build_agent(model, system_prompt)
        output = await agent.arun(prompt)

        hook = block_input.services.get("on_agent_run_complete")
        if hook is not None:
            try:
                hook(
                    {
                        "block_id": self.config.get("block_id"),
                        "model": model,
                        "system_prompt": system_prompt,
                        "prompt": prompt,
                        "content": output.content,
                        "usage": output.usage,
                    }
                )
            except Exception:
                logger.exception(
                    "on_agent_run_complete hook failed in AgentBlock; continuing"
                )

        return BlockOutput(
            data={
                "output": output.content,
                "model": model,
                "usage": output.usage,
            }
        )

    async def stream(
        self, block_input: BlockInput
    ) -> AsyncGenerator[str | BlockOutput]:
        model = self.config.get("model", "gpt-4o-mini")
        system_prompt = await self._build_system_prompt(block_input)
        prompt = self._resolve_prompt(block_input)

        agent = self._build_agent(model, system_prompt)
        chunks: list[str] = []

        async for chunk in agent.astream(prompt):
            chunks.append(chunk)
            yield chunk

        content = "".join(chunks)
        yield BlockOutput(
            data={
                "output": content,
                "model": model,
            }
        )
