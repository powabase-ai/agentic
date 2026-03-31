"""Tests for ConditionBlock — variable resolution and branching."""

import pytest

from agentic.workflow.block import BlockInput
from agentic.workflow.blocks.condition import ConditionBlock


class TestConditionBlockSpacesInName:
    """Block names with spaces must be resolved in condition expressions."""

    @pytest.mark.asyncio
    async def test_block_name_with_spaces_resolves(self):
        """<Parse Classification JSON.output.route> should resolve, not fall through to else."""
        block = ConditionBlock(
            config={
                "branches": [
                    {"expression": '<Parse Classification JSON.output.route> == "mira_policies"'},
                ]
            },
        )

        block_input = BlockInput(
            data={},
            block_outputs={
                "Parse Classification JSON": {
                    "output": {"route": "mira_policies"}
                }
            },
        )

        result = await block.execute(block_input)

        assert result.data["route"] == "if", (
            f"Expected 'if' but got '{result.data['route']}' — "
            "block name with spaces was not resolved"
        )


class TestConditionBlockDebugDiagnostics:
    """Condition block output must include _debug diagnostics."""

    @pytest.mark.asyncio
    async def test_debug_present_on_matching_branch(self):
        """When a branch matches, _debug should show resolved values and result."""
        block = ConditionBlock(
            config={
                "branches": [
                    {"expression": '<Parse Route JSON.output.route> == "master_deed"'},
                ]
            },
        )
        block_input = BlockInput(
            data={},
            block_outputs={
                "Parse Route JSON": {"output": {"route": "master_deed"}}
            },
        )

        result = await block.execute(block_input)

        assert result.data["route"] == "if"
        assert "_debug" in result.data
        debug = result.data["_debug"]
        assert len(debug) == 1
        assert debug[0]["branch"] == "if"
        assert debug[0]["raw"] == '<Parse Route JSON.output.route> == "master_deed"'
        assert debug[0]["result"] is True
        # resolved should contain the placeholder, not the raw ref
        assert "<Parse Route JSON" not in debug[0]["resolved"]

    @pytest.mark.asyncio
    async def test_debug_present_on_else_fallthrough(self):
        """When no branch matches, _debug should show all evaluated branches."""
        block = ConditionBlock(
            config={
                "branches": [
                    {"expression": '<Parse Route JSON.output.route> == "master_deed"'},
                ]
            },
        )
        block_input = BlockInput(
            data={},
            block_outputs={
                "Parse Route JSON": {"output": {"route": "general"}}
            },
        )

        result = await block.execute(block_input)

        assert result.data["route"] == "else"
        assert "_debug" in result.data
        debug = result.data["_debug"]
        assert len(debug) == 1
        assert debug[0]["branch"] == "if"
        assert debug[0]["result"] is False

    @pytest.mark.asyncio
    async def test_debug_includes_variables(self):
        """_debug should include resolved variable values for inspection."""
        block = ConditionBlock(
            config={
                "branches": [
                    {"expression": '<MyBlock.output.value> == "hello"'},
                ]
            },
        )
        block_input = BlockInput(
            data={},
            block_outputs={"MyBlock": {"output": {"value": "hello"}}},
        )

        result = await block.execute(block_input)

        assert result.data["route"] == "if"
        debug = result.data["_debug"]
        assert "variables" in debug[0]
        # The variable dict should contain the resolved value
        var_values = list(debug[0]["variables"].values())
        assert "hello" in var_values

    @pytest.mark.asyncio
    async def test_debug_on_eval_error(self):
        """When expression evaluation errors, _debug should capture the error."""
        block = ConditionBlock(
            config={
                "branches": [
                    {"expression": "<Missing.output.x> >> invalid"},
                ]
            },
        )
        block_input = BlockInput(
            data={},
            block_outputs={},
        )

        result = await block.execute(block_input)

        assert result.data["route"] == "else"
        assert "_debug" in result.data
        debug = result.data["_debug"]
        assert len(debug) == 1
        assert "error" in debug[0]

    @pytest.mark.asyncio
    async def test_debug_multiple_branches(self):
        """With multiple branches, _debug should track each one evaluated."""
        block = ConditionBlock(
            config={
                "branches": [
                    {"expression": '<X.output.v> == "a"'},
                    {"expression": '<X.output.v> == "b"'},
                ]
            },
        )
        block_input = BlockInput(
            data={},
            block_outputs={"X": {"output": {"v": "b"}}},
        )

        result = await block.execute(block_input)

        assert result.data["route"] == "elif_1"
        assert "_debug" in result.data
        debug = result.data["_debug"]
        # Both branches should be recorded
        assert len(debug) == 2
        assert debug[0]["branch"] == "if"
        assert debug[0]["result"] is False
        assert debug[1]["branch"] == "elif_1"
        assert debug[1]["result"] is True
