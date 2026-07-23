from unittest.mock import patch

import pytest

from agentic.agent.agent import Agent
from agentic.agent.hooks import HookConfig
from agentic.agent.output import AgentOutput
from agentic.execution.status import ExecutionStatus
from agentic.orchestration.orchestration import Orchestration


def _fake_agent_output():
    return AgentOutput(
        execution_id="e", status=ExecutionStatus.COMPLETED, content="done"
    )


class TestSupervisorHookThreading:
    def test_supervisor_forwards_hooks_to_orchestrator(self):
        specialist = Agent(name="specialist", model="gpt-5.4", system_prompt="s")
        orch = Orchestration(name="t", description="d", strategy="supervisor")
        orch.add_entity(
            entity_type="agent", agent=specialist, role_description="does things"
        )
        hooks = [
            HookConfig(
                event="PreResponse",
                type="http",
                config={"url": "https://x"},
                id="h1",
                position=0,
            )
        ]

        with patch("agentic.orchestration.strategies.Agent") as MockAgent:
            MockAgent.return_value.run.return_value = _fake_agent_output()
            orch.run("hi", hooks=hooks)

        MockAgent.return_value.run.assert_called_once()
        assert MockAgent.return_value.run.call_args.kwargs.get("hooks") == hooks

    def test_run_without_hooks_is_backward_compatible(self):
        specialist = Agent(name="specialist", model="gpt-5.4", system_prompt="s")
        orch = Orchestration(name="t", description="d", strategy="supervisor")
        orch.add_entity(
            entity_type="agent", agent=specialist, role_description="does things"
        )

        with patch("agentic.orchestration.strategies.Agent") as MockAgent:
            MockAgent.return_value.run.return_value = _fake_agent_output()
            orch.run("hi")

        assert MockAgent.return_value.run.call_args.kwargs.get("hooks") is None


class TestNonSupervisorEnginesIgnoreHooks:
    """R5-I1 (re-argued): the guarantee that a hook can never fire on a
    sequential/parallel orchestration is STRUCTURAL, not a data claim.

    Round 5 defended this with "no hook rows exist yet", which is unverifiable
    and expires the moment someone creates one. The actual protection is that
    these engines accept `hooks` and never reference it — so a hook row that
    reaches them stays inert rather than firing unenforced. Nothing pinned that
    silence, leaving it one refactor away from breaking.

    NOTE: these engines invoke `entity.agent.run` on the real Agent instance,
    NOT the `strategies.Agent` class the supervisor tests patch — so the
    specialist's own `run` is what must be observed here.
    """

    def _orch_and_agent(self, strategy):
        specialist = Agent(name="specialist", model="gpt-5.4", system_prompt="s")
        orch = Orchestration(name="t", description="d", strategy=strategy)
        orch.add_entity(
            entity_type="agent", agent=specialist, role_description="does things"
        )
        return orch, specialist

    def _hooks(self):
        return [
            HookConfig(
                event="PreResponse",
                type="http",
                config={"url": "https://x"},
                id="h1",
                position=0,
            )
        ]

    @pytest.mark.parametrize("strategy", ["sequential", "parallel"])
    def test_engine_does_not_forward_hooks_to_agents(self, strategy):
        orch, specialist = self._orch_and_agent(strategy)
        with patch.object(
            specialist, "run", return_value=_fake_agent_output()
        ) as mock_run:
            orch.run("hi", hooks=self._hooks())

        assert mock_run.call_count >= 1, "the specialist never ran; test is inert"
        for call in mock_run.call_args_list:
            assert not call.kwargs.get("hooks"), (
                f"A {strategy} orchestration must not fire hooks — there is no "
                "supervisor to attach them to, and firing them here would "
                "enforce a policy the CRUD layer refused to configure."
            )

    def test_sequential_still_accepts_the_hooks_kwarg(self):
        """It must accept-and-ignore, not raise — the route passes hooks
        uniformly regardless of strategy."""
        orch, specialist = self._orch_and_agent("sequential")
        with patch.object(specialist, "run", return_value=_fake_agent_output()):
            out = orch.run("hi", hooks=self._hooks())
        assert out is not None
