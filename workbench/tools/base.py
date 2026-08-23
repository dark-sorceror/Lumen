"""The Tool abstraction + registry.

`Tool` is a Protocol (structural typing, not a base class to inherit from) so any object with the right
attributes/method qualifies. `ToolContext` is the seam that lets a tool read
the live ContextObject (needed by `search_context`) without reaching for
globals; future self-editing tools will extend this same seam with an editor
handle.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from workbench.context.manager import ContextManager
    from workbench.context.model import ContextObject

# The tool error contract. A tool -- and ToolRegistry.execute
# itself -- signals failure by returning a string that begins with this EXACT
# prefix; app.py's round loop keys the wire tool_result `error` flag off it.
# Chosen approach: a single-sourced, DOCUMENTED
# sentinel constant rather than the previous bare "error:" literal sniffed in
# app.py. A full structural signal (execute returning (is_error, text) or
# tools raising a typed ToolError) would ripple through base.py + builtins.py
# (all three tools return "error: ..." strings today) + app.py + two test
# files, which is more churn than the fix is worth -- so the prefix stays,
# but is now unambiguous and single-sourced. Contract: a tool MUST NOT return a *non-error* string starting with
# this prefix.
TOOL_ERROR_PREFIX = "error: "


def is_tool_error(result: str) -> bool:
    """True iff `result` is a tool error per the TOOL_ERROR_PREFIX contract.
    The single place app.py (and anyone else) should ask 'did this tool
    fail?' -- never an ad-hoc `.startswith("error:")` literal."""
    return result.startswith(TOOL_ERROR_PREFIX)


@dataclass
class ToolContext:
    """Read access to the live session state, passed into every Tool.run()
    call. `ctx` is the ContextObject a context-reading tool (search_context)
    scans; `manager` is included for tools that need tokenization/lookup
    helpers later. Both are optional so tools can be unit-tested
    with a bare ToolContext() when they don't touch context at all (e.g.
    calculator, get_current_time)."""
    ctx: "ContextObject | None" = None
    manager: "ContextManager | None" = None


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    parameters: dict  # JSON Schema for `arguments`

    def run(self, arguments: dict, ctx: ToolContext) -> str: ...


class ToolRegistry:
    """A fixed, server-owned set of Tools -- never user-supplied code. Produces the OpenAI-function-shaped schema list
    `apply_chat_template(tools=...)` expects, and safely executes a
    model-requested call by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict]:
        """OpenAI "function" shape:
        {"type":"function","function":{"name","description","parameters"}} --
        the list `apply_chat_template(tools=...)` (and our own literal-text
        rendering below) both consume."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def execute(self, name: str, arguments: dict, ctx: ToolContext) -> str:
        """Validate `arguments` against the tool's schema (a light check:
        every key the schema lists as `required` must be present -- the registry
        doesn't pull in a full JSON-schema validator dependency for this),
        run it, and catch any exception -- a tool must never crash the
        server or the WS connection; every failure becomes an error string
        the model can see and react to. Every error string returned by this
        method (unknown tool, bad arguments, or a caught exception) begins
        with TOOL_ERROR_PREFIX -- callers (app.py) use `is_tool_error()` on
        that prefix to set the wire `tool_result` message's `error` flag."""
        tool = self._tools.get(name)
        if tool is None:
            return f"{TOOL_ERROR_PREFIX}unknown tool {name!r}"
        if not isinstance(arguments, dict):
            return f"{TOOL_ERROR_PREFIX}arguments must be a JSON object"
        schema = tool.parameters if isinstance(tool.parameters, dict) else {}
        required = schema.get("required", [])
        missing = [key for key in required if key not in arguments]
        if missing:
            return f"{TOOL_ERROR_PREFIX}missing required argument(s): {', '.join(missing)}"
        try:
            return tool.run(arguments, ctx)
        except Exception as exc:  # noqa: BLE001 -- tool failures must not crash the server
            return f"{TOOL_ERROR_PREFIX}{type(exc).__name__}: {exc}"

    # -- prompt integration ------------------------------------------------

    def render_prompt_block(self) -> str:
        """The literal text Qwen3's chat template renders into the system
        message when `tools=[...]` is passed to `apply_chat_template`
        (verified against the template) -- reproduced here
        directly rather than derived via a live apply_chat_template diff,
        since the wrapper text is a fixed constant of the Qwen3 template
        family, independent of conversation content. See app.py's docstring
        on where this gets prepended as a SYSTEM segment. Returns "" if the
        registry is empty (nothing to render -- callers should skip
        prepending a segment in that case)."""
        schemas = self.schemas()
        if not schemas:
            return ""
        tools_json = "\n".join(json.dumps(s, separators=(",", ":")) for s in schemas)
        return (
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> "
            "XML tags:\n"
            "<tools>\n" + tools_json + "\n</tools>\n\n"
            "For each function call, return a json object with function name "
            "and arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>"
        )
