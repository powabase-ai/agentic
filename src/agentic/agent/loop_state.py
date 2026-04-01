"""Structured state for the ReAct loop — immutable, replaced atomically at each continue site."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class LoopState:
    """Immutable state carried between ReAct loop iterations."""

    messages: tuple[dict[str, Any], ...]
    turn_count: int
    output_recovery_count: int
    has_attempted_compact: bool
    compact_failure_count: int
    budget_exhausted: bool
    current_model: str
    transition: dict[str, Any] | None

    @classmethod
    def initial(cls, messages: list[dict] | tuple[dict, ...], model: str) -> LoopState:
        msgs = tuple(messages) if isinstance(messages, list) else messages
        return cls(
            messages=msgs,
            turn_count=0,
            output_recovery_count=0,
            has_attempted_compact=False,
            compact_failure_count=0,
            budget_exhausted=False,
            current_model=model,
            transition=None,
        )

    def next_turn(self, messages: tuple[dict, ...] | list[dict]) -> LoopState:
        msgs = tuple(messages) if isinstance(messages, list) else messages
        return replace(
            self,
            messages=msgs,
            turn_count=self.turn_count + 1,
            transition={"reason": "next_turn"},
        )

    def recover(
        self,
        messages: tuple[dict, ...] | list[dict],
        reason: str,
        has_attempted_compact: bool | None = None,
        compact_failure_count: int | None = None,
    ) -> LoopState:
        msgs = tuple(messages) if isinstance(messages, list) else messages
        return replace(
            self,
            messages=msgs,
            has_attempted_compact=has_attempted_compact
            if has_attempted_compact is not None
            else self.has_attempted_compact,
            compact_failure_count=compact_failure_count
            if compact_failure_count is not None
            else self.compact_failure_count,
            transition={"reason": reason},
        )

    def with_fallback_model(self, fallback_model: str) -> LoopState:
        return replace(
            self, current_model=fallback_model, transition={"reason": "model_fallback"}
        )

    def with_output_recovery(
        self, messages: tuple[dict, ...] | list[dict]
    ) -> LoopState:
        msgs = tuple(messages) if isinstance(messages, list) else messages
        attempt = self.output_recovery_count + 1
        return replace(
            self,
            messages=msgs,
            output_recovery_count=attempt,
            transition={"reason": "output_recovery", "attempt": attempt},
        )

    def with_budget_exhausted(self) -> LoopState:
        return replace(self, budget_exhausted=True)
