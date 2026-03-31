import pytest

from agentic.agent.loop_state import LoopState


class TestLoopStateCreation:
    def test_create_initial_state(self):
        state = LoopState.initial(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o-mini",
        )
        assert state.turn_count == 0
        assert state.output_recovery_count == 0
        assert state.has_attempted_compact is False
        assert state.compact_failure_count == 0
        assert state.budget_exhausted is False
        assert state.current_model == "gpt-4o-mini"
        assert state.transition is None
        assert len(state.messages) == 1

    def test_state_is_frozen(self):
        state = LoopState.initial(messages=[], model="gpt-4o-mini")
        with pytest.raises(AttributeError):
            state.turn_count = 5


class TestLoopStateTransitions:
    def test_next_turn(self):
        state = LoopState.initial(
            messages=[{"role": "user", "content": "hi"}], model="gpt-4o-mini"
        )
        tool_results = [{"role": "tool", "tool_call_id": "1", "content": "ok"}]
        next_state = state.next_turn(messages=(*state.messages, *tool_results))
        assert next_state.turn_count == 1
        assert next_state.transition == {"reason": "next_turn"}
        assert len(next_state.messages) == 2
        assert next_state.output_recovery_count == state.output_recovery_count

    def test_recovery_does_not_increment_turn(self):
        state = LoopState.initial(messages=[], model="gpt-4o-mini")
        state = state.next_turn(messages=state.messages)
        recovered = state.recover(
            messages=state.messages,
            reason="reactive_compact",
            has_attempted_compact=True,
        )
        assert recovered.turn_count == 1
        assert recovered.transition == {"reason": "reactive_compact"}
        assert recovered.has_attempted_compact is True

    def test_model_fallback(self):
        state = LoopState.initial(messages=[], model="claude-sonnet-4-6")
        fallback = state.with_fallback_model("gpt-4.1-mini")
        assert fallback.current_model == "gpt-4.1-mini"
        assert fallback.turn_count == state.turn_count
        assert fallback.transition == {"reason": "model_fallback"}

    def test_output_recovery(self):
        state = LoopState.initial(messages=[], model="gpt-4o-mini")
        recovery_msg = {"role": "user", "content": "Continue where you left off."}
        recovered = state.with_output_recovery(messages=(*state.messages, recovery_msg))
        assert recovered.output_recovery_count == 1
        assert recovered.transition == {"reason": "output_recovery", "attempt": 1}
        assert recovered.turn_count == state.turn_count

    def test_budget_exhaustion(self):
        state = LoopState.initial(messages=[], model="gpt-4o-mini")
        exhausted = state.with_budget_exhausted()
        assert exhausted.budget_exhausted is True
