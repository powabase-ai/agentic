import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

import agentic.agent.model_registry as mr
from agentic.agent import compaction


def _reset_cache():
    mr._openrouter_cache = {}
    mr._openrouter_cache_ts = 0.0
    mr._openrouter_failure_ts = 0.0
    mr._warned.clear()


@pytest.fixture(autouse=True)
def reset_registry_cache():
    """Reset the module-global registry cache around EVERY test in this file.

    These globals are process-wide; without teardown a test that populates or
    poisons them leaks into unrelated tests in the same session.
    """
    _reset_cache()
    yield
    _reset_cache()


class TestFetchOpenRouterRegistry:
    def test_prefers_top_provider_then_root_context_length(self):
        payload = {
            "data": [
                {
                    "id": "a/b",
                    "context_length": 100,
                    "top_provider": {"context_length": 200},
                },
                {"id": "c/d", "context_length": 300, "top_provider": {}},
                {"id": "no-window"},
            ]
        }
        resp = MagicMock()
        resp.json.return_value = payload
        resp.raise_for_status.return_value = None
        with patch.object(mr.httpx, "get", return_value=resp):
            reg = mr._fetch_openrouter_registry()
        assert reg["a/b"] == 200  # top_provider wins
        assert reg["c/d"] == 300  # falls back to root context_length
        assert "no-window" not in reg  # skipped, no window

    def test_failure_returns_empty(self):
        with patch.object(mr.httpx, "get", side_effect=Exception("network")):
            assert mr._fetch_openrouter_registry() == {}


class TestResolveContextWindow:
    def test_openrouter_deepseek_via_registry(self):
        fake = {"deepseek/deepseek-v4-pro": 1048576}
        with patch.object(mr, "_fetch_openrouter_registry", return_value=fake):
            assert (
                mr.resolve_context_window("openrouter/deepseek/deepseek-v4-pro")
                == 1048576
            )

    def test_openrouter_qwen_via_registry(self):
        fake = {"qwen/qwen3.7-max": 1000000}
        with patch.object(mr, "_fetch_openrouter_registry", return_value=fake):
            assert mr.resolve_context_window("openrouter/qwen/qwen3.7-max") == 1000000

    def test_mainstream_via_litellm(self):
        # Anthropic's dashed litellm name has no OpenRouter match, so the bare
        # lookup misses and litellm answers — and Claude is shared-budget, so
        # max_input IS the total. Registry mocked empty to stay off the network.
        with (
            patch.object(mr, "_fetch_openrouter_registry", return_value={}),
            patch.object(
                mr.litellm,
                "get_model_info",
                return_value={"max_input_tokens": 1000000},
            ),
        ):
            assert mr.resolve_context_window("claude-opus-4-8") == 1000000

    def test_bare_input_only_model_prefers_openrouter_total(self):
        # GPT-5 is input-only: litellm reports max_input=272_000, but its real
        # window is 400_000 total (output billed separately). OpenRouter's
        # context_length carries the total; preferring it recovers the ~128k of
        # context litellm's input ceiling hides.
        with (
            patch.object(
                mr, "_fetch_openrouter_registry", return_value={"openai/gpt-5": 400_000}
            ),
            patch.object(
                mr.litellm,
                "get_llm_provider",
                return_value=("gpt-5", "openai", None, None),
            ),
            patch.object(
                mr.litellm, "get_model_info", return_value={"max_input_tokens": 272_000}
            ),
        ):
            assert mr.resolve_context_window("gpt-5") == 400_000

    def test_bare_model_missing_from_openrouter_falls_back_to_litellm(self):
        # Provider resolves but the derived id isn't in the registry (name-format
        # mismatch). Must fall back to litellm rather than the default — no
        # regression from today's litellm-only behavior.
        with (
            patch.object(
                mr,
                "_fetch_openrouter_registry",
                return_value={"openai/gpt-4o": 128_000},
            ),
            patch.object(
                mr.litellm,
                "get_llm_provider",
                return_value=("claude-opus-4-8", "anthropic", None, None),
            ),
            patch.object(
                mr.litellm,
                "get_model_info",
                return_value={"max_input_tokens": 1_000_000},
            ),
        ):
            assert mr.resolve_context_window("claude-opus-4-8") == 1_000_000

    def test_bare_model_provider_unresolved_falls_back_to_litellm(self):
        # A bare open-weight name litellm can't attribute to a provider must not
        # crash the derivation — it falls through to litellm.
        with (
            patch.object(
                mr, "_fetch_openrouter_registry", return_value={"openai/gpt-5": 400_000}
            ),
            patch.object(
                mr.litellm,
                "get_llm_provider",
                side_effect=Exception("unknown provider"),
            ),
            patch.object(
                mr.litellm, "get_model_info", return_value={"max_input_tokens": 131_072}
            ),
        ):
            assert mr.resolve_context_window("some-local-model") == 131_072

    def test_bare_model_negative_openrouter_total_falls_back(self):
        # The positivity guard applies to the bare-name path too: a nonsense
        # negative total must not be preferred over litellm's real window.
        with (
            patch.object(
                mr, "_fetch_openrouter_registry", return_value={"openai/gpt-5": -1}
            ),
            patch.object(
                mr.litellm,
                "get_llm_provider",
                return_value=("gpt-5", "openai", None, None),
            ),
            patch.object(
                mr.litellm, "get_model_info", return_value={"max_input_tokens": 272_000}
            ),
        ):
            assert mr.resolve_context_window("gpt-5") == 272_000

    def test_openrouter_miss_falls_back_to_litellm_strip(self):
        def fake_info(m):
            if m == "deepseek/deepseek-v4-pro":
                return {"max_input_tokens": 1048576}
            raise Exception("unmapped")

        with (
            patch.object(mr, "_fetch_openrouter_registry", return_value={}),
            patch.object(mr.litellm, "get_model_info", side_effect=fake_info),
        ):
            assert (
                mr.resolve_context_window("openrouter/deepseek/deepseek-v4-pro")
                == 1048576
            )

    def test_unknown_returns_default_and_warns(self, caplog):
        with (
            patch.object(mr, "_fetch_openrouter_registry", return_value={}),
            patch.object(
                mr.litellm, "get_model_info", side_effect=Exception("unmapped")
            ),
        ):
            with caplog.at_level("WARNING"):
                got = mr.resolve_context_window("openrouter/qwen/qwen3.7-max")
        assert got == mr.DEFAULT_CONTEXT_WINDOW
        assert "context_window_unresolved" in caplog.text

    def test_registry_cached_across_calls(self):
        with patch.object(
            mr,
            "_fetch_openrouter_registry",
            return_value={"deepseek/deepseek-v4-pro": 1048576},
        ) as fetch:
            mr.resolve_context_window("openrouter/deepseek/deepseek-v4-pro")
            mr.resolve_context_window("openrouter/deepseek/deepseek-v4-pro")
        assert fetch.call_count == 1  # second call served from cache

    def test_stale_cache_survives_failed_refetch(self):
        with patch.object(
            mr,
            "_fetch_openrouter_registry",
            return_value={"deepseek/deepseek-v4-pro": 1048576},
        ):
            mr.resolve_context_window("openrouter/deepseek/deepseek-v4-pro")
        # Force TTL expiry (TTL is 6h) without mocking time.time().
        mr._openrouter_cache_ts = time.time() - (7 * 60 * 60)
        with patch.object(mr, "_fetch_openrouter_registry", return_value={}):
            assert (
                mr.resolve_context_window("openrouter/deepseek/deepseek-v4-pro")
                == 1048576
            )

    def test_negative_openrouter_cached_window_falls_through(self):
        # A negative `context_length` cached from the registry must not be
        # treated as a valid window — `if window:` alone lets `-5` through
        # since a negative int is truthy, which yields a negative threshold
        # downstream (the exact defect B1 was supposed to make impossible).
        with (
            patch.object(
                mr,
                "_fetch_openrouter_registry",
                return_value={"bad/model": -5},
            ),
            patch.object(
                mr.litellm, "get_model_info", side_effect=Exception("unmapped")
            ),
        ):
            assert (
                mr.resolve_context_window("openrouter/bad/model")
                == mr.DEFAULT_CONTEXT_WINDOW
            )

    def test_negative_litellm_window_falls_through_to_default(self):
        with (
            patch.object(mr, "_fetch_openrouter_registry", return_value={}),
            patch.object(
                mr.litellm, "get_model_info", return_value={"max_input_tokens": -100}
            ),
        ):
            assert (
                mr.resolve_context_window("claude-opus-4-8")
                == mr.DEFAULT_CONTEXT_WINDOW
            )

    def test_litellm_window_string_is_coerced_to_int(self):
        # `_fetch_openrouter_registry` int-coerces its window (:85); the
        # litellm path must do the same so `resolve_context_window` honors
        # its `-> int` annotation instead of handing the caller a `str`.
        with (
            patch.object(mr, "_fetch_openrouter_registry", return_value={}),
            patch.object(
                mr.litellm,
                "get_model_info",
                return_value={"max_input_tokens": "1000000"},
            ),
        ):
            got = mr.resolve_context_window("claude-opus-4-8")
        assert got == 1000000
        assert isinstance(got, int)

    def test_litellm_window_non_numeric_string_falls_back_to_default(self):
        with (
            patch.object(mr, "_fetch_openrouter_registry", return_value={}),
            patch.object(
                mr.litellm,
                "get_model_info",
                return_value={"max_input_tokens": "unknown"},
            ),
        ):
            assert (
                mr.resolve_context_window("claude-opus-4-8")
                == mr.DEFAULT_CONTEXT_WINDOW
            )


class TestFailureBackoff:
    def test_failed_fetch_not_retried_on_every_call(self):
        # Empty cache + failing fetch must not re-issue an HTTP GET per call:
        # get_context_threshold runs twice per agent step.
        with (
            patch.object(mr, "_fetch_openrouter_registry", return_value={}) as fetch,
            patch.object(
                mr.litellm, "get_model_info", side_effect=Exception("unmapped")
            ),
        ):
            for _ in range(5):
                mr.resolve_context_window("openrouter/qwen/qwen3.7-max")
        assert fetch.call_count == 1

    def test_fetch_retried_after_backoff_expires(self):
        with (
            patch.object(mr, "_fetch_openrouter_registry", return_value={}) as fetch,
            patch.object(
                mr.litellm, "get_model_info", side_effect=Exception("unmapped")
            ),
        ):
            mr.resolve_context_window("openrouter/qwen/qwen3.7-max")
            mr._openrouter_failure_ts = time.time() - (
                mr._OPENROUTER_FAILURE_TTL_SECONDS + 1
            )
            mr.resolve_context_window("openrouter/qwen/qwen3.7-max")
        assert fetch.call_count == 2

    def test_good_fetch_clears_backoff(self):
        with patch.object(mr, "_fetch_openrouter_registry", return_value={}):
            mr.resolve_context_window("openrouter/qwen/qwen3.7-max")
        assert mr._openrouter_failure_ts > 0.0
        mr._openrouter_failure_ts = time.time() - (
            mr._OPENROUTER_FAILURE_TTL_SECONDS + 1
        )
        with patch.object(
            mr, "_fetch_openrouter_registry", return_value={"qwen/qwen3.7-max": 999}
        ):
            assert mr.resolve_context_window("openrouter/qwen/qwen3.7-max") == 999
        assert mr._openrouter_failure_ts == 0.0


# Shape of a real OpenRouter /api/v1/models entry, trimmed to the fields the
# registry reads. Used to exercise the resolver end-to-end without stubbing it.
_OPENROUTER_PAYLOAD = {
    "data": [
        {
            "id": "deepseek/deepseek-v4-pro",
            "name": "DeepSeek: V4 Pro",
            "context_length": 1048576,
            "architecture": {"modality": "text->text"},
            "pricing": {"prompt": "0.0000004", "completion": "0.0000016"},
            "top_provider": {
                "context_length": 1048576,
                # Real published value: the ceiling a caller MAY request, not
                # what we request. Large enough that a re-introduced
                # subtraction is unmistakable in the composed assertions below.
                "max_completion_tokens": 384000,
                "is_moderated": False,
            },
        },
        {
            # top_provider carries NO context_length, so resolution must fall
            # back to the root `context_length`. Keep it that way: with the key
            # present on both, the fallback branch is never exercised.
            "id": "qwen/qwen3.7-max",
            "context_length": 1000000,
            "top_provider": {"is_moderated": False},
        },
    ]
}


class TestTotalVsInputTokens:
    """The published total is stored raw — output room is reserved downstream."""

    def test_max_completion_tokens_not_subtracted_from_total(self):
        # max_completion_tokens is the ceiling a caller MAY request, not what
        # we request. get_context_threshold already reserves the real
        # max_tokens; subtracting here too would double-count it.
        resp = MagicMock()
        resp.json.return_value = _OPENROUTER_PAYLOAD
        resp.raise_for_status.return_value = None
        with patch.object(mr.httpx, "get", return_value=resp):
            reg = mr._fetch_openrouter_registry()
        assert reg["deepseek/deepseek-v4-pro"] == 1048576

    def test_falls_back_to_root_context_length(self):
        resp = MagicMock()
        resp.json.return_value = _OPENROUTER_PAYLOAD
        resp.raise_for_status.return_value = None
        with patch.object(mr.httpx, "get", return_value=resp):
            reg = mr._fetch_openrouter_registry()
        assert reg["qwen/qwen3.7-max"] == 1000000

    def test_inverted_max_completion_does_not_shrink_the_window(self):
        # max_completion_tokens > context_length is malformed provider data.
        # We store the published total regardless — the single output
        # reservation lives in get_context_threshold, keyed on the output we
        # actually request. Subtracting here would yield a 1-token window.
        payload = {
            "data": [
                {
                    "id": "weird/model",
                    "top_provider": {
                        "context_length": 1000,
                        "max_completion_tokens": 5000,
                    },
                }
            ]
        }
        resp = MagicMock()
        resp.json.return_value = payload
        resp.raise_for_status.return_value = None
        with patch.object(mr.httpx, "get", return_value=resp):
            reg = mr._fetch_openrouter_registry()
        assert reg["weird/model"] == 1000


class TestComposedWindowAndThreshold:
    """Incident regression: window resolution AND threshold math, composed.

    Both halves of the original bug were only ever tested with the other half
    mocked, so a broken prefix-strip or a reordered resolution chain would
    reproduce the production incident with the whole suite green. This stubs
    ONLY the HTTP fetch and runs the real resolver into the real threshold.
    """

    def test_deepseek_threshold_far_above_the_buggy_default(self):
        resp = MagicMock()
        resp.json.return_value = _OPENROUTER_PAYLOAD
        resp.raise_for_status.return_value = None
        with patch.object(mr.httpx, "get", return_value=resp):
            window = mr.resolve_context_window("openrouter/deepseek/deepseek-v4-pro")
            threshold = compaction.get_context_threshold(
                "openrouter/deepseek/deepseek-v4-pro"
            )

        # The incident: the openrouter/ prefix was not stripped, so the model
        # fell through to DEFAULT_CONTEXT_WINDOW (128k) and the threshold
        # collapsed to 107_000 — compaction fired constantly on a 1M model.
        assert window == 1048576
        assert threshold != 107_000
        assert threshold > 800_000

        # Exact composed value: 1_048_576 - 8_000 (reserved output)
        # - 83_886 (8% buffer).
        assert threshold == 956_690
        # And explicitly NOT the double-counted value that a re-introduced
        # `- max_completion_tokens` in the registry would produce:
        # (1_048_576 - 384_000) - 8_000 - 53_166 = 603_410.
        assert threshold != 603_410

    def test_composed_path_makes_exactly_one_http_call(self):
        resp = MagicMock()
        resp.json.return_value = _OPENROUTER_PAYLOAD
        resp.raise_for_status.return_value = None
        with patch.object(mr.httpx, "get", return_value=resp) as get:
            for _ in range(4):
                compaction.get_context_threshold("openrouter/deepseek/deepseek-v4-pro")
        assert get.call_count == 1


class TestUnresolvedWarningDedupe:
    def test_warns_once_per_model(self, caplog):
        with (
            patch.object(mr, "_fetch_openrouter_registry", return_value={}),
            patch.object(
                mr.litellm, "get_model_info", side_effect=Exception("unmapped")
            ),
        ):
            with caplog.at_level("WARNING"):
                for _ in range(10):
                    mr.resolve_context_window("openrouter/qwen/qwen3.7-max")
        lines = [
            r for r in caplog.records if "context_window_unresolved" in r.getMessage()
        ]
        assert len(lines) == 1

    def test_distinct_models_each_warn(self, caplog):
        with (
            patch.object(mr, "_fetch_openrouter_registry", return_value={}),
            patch.object(
                mr.litellm, "get_model_info", side_effect=Exception("unmapped")
            ),
        ):
            with caplog.at_level("WARNING"):
                mr.resolve_context_window("openrouter/a/one")
                mr.resolve_context_window("openrouter/b/two")
        lines = [
            r for r in caplog.records if "context_window_unresolved" in r.getMessage()
        ]
        assert len(lines) == 2


class TestConcurrentRefresh:
    def test_cold_cache_thundering_herd_fetches_once(self):
        # orchestration/strategies.py runs agents in parallel threads; on a
        # cold cache every one of them would otherwise issue its own 5s GET.
        calls = []
        barrier = threading.Barrier(8)

        def slow_fetch():
            calls.append(1)
            time.sleep(0.05)
            return {"deepseek/deepseek-v4-pro": 1048576}

        def worker():
            barrier.wait()  # maximize the race window
            return mr.resolve_context_window("openrouter/deepseek/deepseek-v4-pro")

        with patch.object(mr, "_fetch_openrouter_registry", side_effect=slow_fetch):
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = [f.result() for f in [pool.submit(worker) for _ in range(8)]]

        assert results == [1048576] * 8
        assert len(calls) == 1


class TestFetchOpenRouterRegistryMalformedRows:
    """The parse loop walks third-party data. Anything that escapes it
    propagates resolve_context_window -> get_context_threshold -> the agent
    loop, which has no handler — so one bad row from OpenRouter would fail
    every agent run in the fleet."""

    @staticmethod
    def _resp(payload):
        resp = MagicMock()
        resp.json.return_value = payload
        resp.raise_for_status.return_value = None
        return resp

    def test_non_numeric_context_length_does_not_raise(self):
        payload = {"data": [{"id": "bad/model", "context_length": "unknown"}]}
        with patch.object(mr.httpx, "get", return_value=self._resp(payload)):
            assert mr._fetch_openrouter_registry() == {}

    def test_data_object_instead_of_list_does_not_raise(self):
        # `{"data": {...}}` iterates as strings, so `m.get` is an AttributeError.
        payload = {"data": {"a/b": {"context_length": 100}}}
        with patch.object(mr.httpx, "get", return_value=self._resp(payload)):
            assert mr._fetch_openrouter_registry() == {}

    def test_one_bad_row_does_not_discard_the_good_rows(self):
        payload = {
            "data": [
                {"id": "good/one", "context_length": 100},
                {"id": "bad/model", "context_length": "unknown"},
                {"id": "good/two", "context_length": 200},
            ]
        }
        with patch.object(mr.httpx, "get", return_value=self._resp(payload)):
            reg = mr._fetch_openrouter_registry()
        assert reg == {"good/one": 100, "good/two": 200}

    def test_bad_row_is_logged_with_its_id(self, caplog):
        payload = {"data": [{"id": "bad/model", "context_length": "unknown"}]}
        with (
            patch.object(mr.httpx, "get", return_value=self._resp(payload)),
            caplog.at_level("WARNING"),
        ):
            mr._fetch_openrouter_registry()
        assert "bad/model" in caplog.text

    def test_malformed_row_does_not_break_resolve_context_window(self):
        payload = {"data": [{"id": "bad/model", "context_length": "unknown"}]}
        with (
            patch.object(mr.httpx, "get", return_value=self._resp(payload)),
            patch.object(
                mr.litellm, "get_model_info", side_effect=Exception("unmapped")
            ),
        ):
            assert (
                mr.resolve_context_window("openrouter/bad/model")
                == mr.DEFAULT_CONTEXT_WINDOW
            )

    def test_non_string_id_does_not_pollute_the_cache(self):
        # `{"id": 123}` would become `{123: 50}` in a `dict[str, int]` if the
        # id type isn't checked.
        payload = {
            "data": [
                {"id": 123, "context_length": 50},
                {"id": "good/one", "context_length": 100},
            ]
        }
        with patch.object(mr.httpx, "get", return_value=self._resp(payload)):
            reg = mr._fetch_openrouter_registry()
        assert reg == {"good/one": 100}
        assert 123 not in reg

    def test_backoff_is_armed_even_if_fetch_raises(self):
        """The failure timestamp must be assigned on the exception path too.
        Otherwise every subsequent call re-fetches and re-raises."""
        with patch.object(
            mr, "_fetch_openrouter_registry", side_effect=RuntimeError("boom")
        ):
            with pytest.raises(RuntimeError):
                mr._openrouter_registry()
        assert mr._openrouter_failure_ts > 0.0

        # Backoff now holds: the next call is served from cache, no re-fetch.
        with patch.object(
            mr, "_fetch_openrouter_registry", side_effect=RuntimeError("boom")
        ) as fetch:
            assert mr._openrouter_registry() == {}
        fetch.assert_not_called()


class TestFallbackLogging:
    """Falling back past the primary source must not be silent: litellm
    under-reports some openrouter ids by 6.4x, which is the exact production
    bug this module exists to fix."""

    def test_warns_when_openrouter_model_resolves_via_litellm(self, caplog):
        with (
            patch.object(mr, "_fetch_openrouter_registry", return_value={}),
            patch.object(
                mr.litellm, "get_model_info", return_value={"max_input_tokens": 163_840}
            ),
            caplog.at_level("WARNING"),
        ):
            got = mr.resolve_context_window("openrouter/deepseek/deepseek-v4-pro")
        assert got == 163_840
        assert "context_window_fallback" in caplog.text
        assert "openrouter/deepseek/deepseek-v4-pro" in caplog.text
        assert "163840" in caplog.text

    def test_fallback_warning_is_deduped(self, caplog):
        with (
            patch.object(mr, "_fetch_openrouter_registry", return_value={}),
            patch.object(
                mr.litellm, "get_model_info", return_value={"max_input_tokens": 163_840}
            ),
            caplog.at_level("WARNING"),
        ):
            for _ in range(5):
                mr.resolve_context_window("openrouter/deepseek/deepseek-v4-pro")
        assert caplog.text.count("context_window_fallback") == 1

    def test_no_warning_when_registry_answers(self, caplog):
        fake = {"deepseek/deepseek-v4-pro": 1_048_576}
        with (
            patch.object(mr, "_fetch_openrouter_registry", return_value=fake),
            caplog.at_level("WARNING"),
        ):
            mr.resolve_context_window("openrouter/deepseek/deepseek-v4-pro")
        assert "context_window_fallback" not in caplog.text

    def test_no_warning_for_non_openrouter_model(self, caplog):
        """litellm answers for a plain model id whose OpenRouter lookup misses —
        nothing was skipped, so there is nothing to warn about."""
        with (
            patch.object(mr, "_fetch_openrouter_registry", return_value={}),
            patch.object(
                mr.litellm, "get_model_info", return_value={"max_input_tokens": 200_000}
            ),
            caplog.at_level("WARNING"),
        ):
            mr.resolve_context_window("claude-opus-4-8")
        assert "context_window_fallback" not in caplog.text
