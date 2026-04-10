"""Tool configuration — centralized runtime constants.

All tunable defaults for agent tool definitions live here so they're
easy to find, review, and adjust without digging through tools.py.
"""

# ---------------------------------------------------------------------------
# Result truncation
# ---------------------------------------------------------------------------

# Maximum characters for CustomTool HTTP response bodies.
# Prevents a single external API call from flooding agent context.
MAX_TOOL_OUTPUT_LENGTH = 10_000

# Default max_result_chars for ToolDefinition base class.
# Applied after execute() returns; None = unlimited.
DEFAULT_MAX_RESULT_CHARS = 50_000

# ---------------------------------------------------------------------------
# Timeouts (seconds)
# ---------------------------------------------------------------------------

# Default HTTP timeout for CustomTool endpoint calls.
CUSTOM_TOOL_TIMEOUT = 30

# Default timeout for MCP tool calls.
MCP_TOOL_TIMEOUT = 30

# ---------------------------------------------------------------------------
# Delegate tool defaults
# ---------------------------------------------------------------------------

# Maximum ReAct steps for a DelegateTool sub-agent.
DELEGATE_MAX_STEPS = 10
