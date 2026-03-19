"""Router block — uses an LLM to choose a route from a set of options."""

from __future__ import annotations

import json
import logging

from agentic.agent.agent import Agent
from agentic.workflow.block import BaseBlock, BlockInput, BlockOutput
from agentic.workflow.variable_resolver import resolve_value

logger = logging.getLogger(__name__)


class RouterBlock(BaseBlock):
    block_type = "router"

    async def execute(self, block_input: BlockInput) -> BlockOutput:
        model = self.config.get("model", "gpt-4o-mini")
        prompt = self.config.get("prompt", "")
        prompt = str(resolve_value(prompt, block_input.block_outputs))

        routes = self.config.get("routes", [])
        route_names = [r if isinstance(r, str) else r.get("name", str(i))
                       for i, r in enumerate(routes)]

        system_prompt = self.config.get("system_prompt", "")
        system_prompt = str(resolve_value(system_prompt, block_input.block_outputs))

        # Ask LLM to pick a route
        routing_prompt = (
            f"{system_prompt}\n\n"
            f"Given the following input, choose exactly one route from the options below.\n"
            f"Options: {json.dumps(route_names)}\n\n"
            f"Input: {prompt}\n\n"
            f"Respond with ONLY the route name, nothing else."
        )

        agent = Agent(model=model)
        output = await agent.arun(routing_prompt)
        chosen = (output.content or "").strip().strip('"').strip("'")

        # Normalize: pick closest match
        route = chosen if chosen in route_names else (route_names[0] if route_names else "default")

        return BlockOutput(
            data={
                "output": route,
                "route": route,
                "all_routes": route_names,
            }
        )
