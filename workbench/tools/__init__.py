"""Native tool calling.

`default_registry()` builds a fresh ToolRegistry with the three safe
built-ins registered; app.py's create_app takes an explicit `tool_registry`
param (default None == tools disabled) rather than defaulting to this
registry itself -- see app.py's module docstring for why."""
from __future__ import annotations

from workbench.tools.base import (
    TOOL_ERROR_PREFIX, Tool, ToolContext, ToolRegistry, is_tool_error,
)
from workbench.tools.builtins import CalculatorTool, CurrentTimeTool, SearchContextTool


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(CurrentTimeTool())
    registry.register(SearchContextTool())
    return registry


__all__ = [
    "Tool", "ToolContext", "ToolRegistry", "default_registry",
    "TOOL_ERROR_PREFIX", "is_tool_error",
    "CalculatorTool", "CurrentTimeTool", "SearchContextTool",
]
