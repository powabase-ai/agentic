"""Tests for nested output field access in input mappings."""

from agentic.workflow.engine import _resolve_output_path, DAGEngine


class TestResolveOutputPath:
    """Unit tests for _resolve_output_path."""

    def test_simple_output_key(self):
        assert _resolve_output_path({"output": "hello"}, "output") == "hello"

    def test_dotted_path(self):
        data = {"output": {"message": "hi"}}
        assert _resolve_output_path(data, "output.message") == "hi"

    def test_deep_dotted_path(self):
        data = {"output": {"data": {"items": [1, 2]}}}
        assert _resolve_output_path(data, "output.data.items") == [1, 2]

    def test_missing_intermediate_key(self):
        data = {"output": {"a": 1}}
        assert _resolve_output_path(data, "output.b") is None

    def test_non_dict_source(self):
        assert _resolve_output_path("raw string", "output") is None

    def test_top_level_non_output_key(self):
        data = {"model": "gpt-4", "output": "hi"}
        assert _resolve_output_path(data, "model") == "gpt-4"


class TestApplyInputMappings:
    """Integration tests for DAGEngine._apply_input_mappings."""

    def test_dotted_path_sets_target(self):
        config = {
            "_inputMappings": [
                {
                    "sourceId": "split_3",
                    "outputField": "output.message",
                    "targetField": "prompt",
                },
            ],
            "prompt": "",
        }
        block_outputs = {"split_3": {"output": {"message": "hello world"}}}
        DAGEngine._apply_input_mappings(config, block_outputs)
        assert config["prompt"] == "hello world"

    def test_multiple_mappings_concatenate_strings(self):
        config = {
            "_inputMappings": [
                {
                    "sourceId": "a",
                    "outputField": "output",
                    "targetField": "prompt",
                },
                {
                    "sourceId": "b",
                    "outputField": "output",
                    "targetField": "prompt",
                },
            ],
            "prompt": "",
        }
        block_outputs = {"a": {"output": "line1"}, "b": {"output": "line2"}}
        DAGEngine._apply_input_mappings(config, block_outputs)
        assert config["prompt"] == "line1\nline2"

    def test_missing_source_skips_gracefully(self):
        config = {
            "_inputMappings": [
                {
                    "sourceId": "missing",
                    "outputField": "output",
                    "targetField": "prompt",
                },
            ],
            "prompt": "original",
        }
        DAGEngine._apply_input_mappings(config, {})
        assert config["prompt"] == "original"

    def test_backward_compat_fallback(self):
        """When outputField doesn't start with 'output', try output.<field>."""
        config = {
            "_inputMappings": [
                {
                    "sourceId": "split_3",
                    "outputField": "message",
                    "targetField": "prompt",
                },
            ],
            "prompt": "",
        }
        block_outputs = {"split_3": {"output": {"message": "hi"}}}
        DAGEngine._apply_input_mappings(config, block_outputs)
        assert config["prompt"] == "hi"

    def test_input_mappings_popped_from_config(self):
        config = {
            "_inputMappings": [
                {
                    "sourceId": "a",
                    "outputField": "output",
                    "targetField": "prompt",
                },
            ],
            "prompt": "",
        }
        block_outputs = {"a": {"output": "val"}}
        DAGEngine._apply_input_mappings(config, block_outputs)
        assert "_inputMappings" not in config
