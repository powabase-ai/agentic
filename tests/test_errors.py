from agentic.agent.errors import classify_error, classify_finish_reason


class TestClassifyError:
    def test_rate_limit(self):
        from litellm.exceptions import RateLimitError

        error = RateLimitError(
            message="rate limited", llm_provider="anthropic", model="claude-sonnet-4-6"
        )
        assert classify_error(error) == "rate_limit"

    def test_context_overflow_from_message(self):
        error = Exception("prompt is too long: 200000 tokens > 100000 maximum")
        assert classify_error(error) == "prompt_too_long"

    def test_context_overflow_413(self):
        error = Exception("Error code: 413")
        assert classify_error(error) == "prompt_too_long"

    def test_max_output_tokens(self):
        error = Exception("max_tokens")
        assert classify_error(error) == "max_output_tokens"

    def test_model_error(self):
        error = Exception("Internal server error")
        assert classify_error(error) == "model_error"

    def test_unrecoverable(self):
        error = Exception("Invalid API key")
        assert classify_error(error) == "unrecoverable"

    def test_none_returns_none(self):
        assert classify_error(None) is None


class TestClassifyFinishReason:
    def test_max_tokens_finish_reason(self):
        assert classify_finish_reason("length") == "max_output_tokens"

    def test_stop_finish_reason(self):
        assert classify_finish_reason("stop") is None

    def test_tool_calls_finish_reason(self):
        assert classify_finish_reason("tool_calls") is None
