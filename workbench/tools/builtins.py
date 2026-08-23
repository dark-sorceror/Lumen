"""The starter tool set: three safe, server-side
tools. NONE of them shell out, `eval`, hit the network, or touch the
filesystem -- calculator is an AST-validated evaluator (never `eval`),
get_current_time reads the local clock, and search_context is a read-only
scan over the ContextObject's own doc_chunk segments already in memory."""
from __future__ import annotations

import ast
import datetime
import operator

from workbench.context.model import SegmentKind
from workbench.tools.base import ToolContext

# -- calculator: AST-validated safe evaluator (never eval/exec) -------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

# Bounds the exponent magnitude -- applied to the RIGHT-hand operand of `**`, before
# the pow() call, so a huge exponent is rejected up front rather than
# actually computed. A cheap early-out; the real DoS guard is the cumulative
# magnitude bound below.
_MAX_EXPONENT = 1000

# Cumulative-magnitude bound on intermediate INTEGER results.
# The per-op exponent cap above does NOT stop pow-*towers* like
# ``((10**1000)**1000)**1000``: every individual exponent is <= _MAX_EXPONENT,
# yet nesting grows the BASE, so the value blows up to ~10^(10^9) and
# hangs/OOMs the CPU-bound worker thread (the 5s asyncio timeout can't rescue
# it -- it cancels the future but the thread keeps burning). So after every
# integer BinOp/UnaryOp we reject any result whose bit_length exceeds this
# ceiling. 13000 bits ~= 3913 decimal digits -- comfortably below CPython's
# ~4300-digit int->str conversion limit (so ``str(result)`` can never raise a
# ValueError) and small enough that evaluation stays fast. A normal calculator
# never needs a number anywhere near this large.
_MAX_RESULT_BITS = 13000


def _check_magnitude(result: int | float) -> None:
    """Reject an intermediate integer result that has grown past
    _MAX_RESULT_BITS. Only ints are unbounded in Python;
    floats overflow to inf (handled by the OverflowError catch in run) so they
    need no bit check. Called after each integer BinOp/UnaryOp so a pow-tower
    is short-circuited before the *next* operation can square it again."""
    if isinstance(result, int) and result.bit_length() > _MAX_RESULT_BITS:
        raise _CalculatorError("result too large")


class _CalculatorError(ValueError):
    """Raised for any expression node/operator this evaluator doesn't
    explicitly allow -- caught by CalculatorTool.run and turned into an
    error string, never propagated as a crash."""


def _safe_eval(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            # bool is an int subclass in Python -- explicitly excluded so
            # `True`/`False` literals (which ast.parse never actually
            # produces as bare Constants from an arithmetic-only grammar
            # anyway, since there's no Name access to True/False here) can't
            # sneak through as 1/0.
            raise _CalculatorError(f"unsupported constant: {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise _CalculatorError(f"unsupported operator: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if op_type is ast.Pow and abs(right) > _MAX_EXPONENT:
            raise _CalculatorError("exponent too large")
        result = _BIN_OPS[op_type](left, right)
        _check_magnitude(result)
        return result
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise _CalculatorError(f"unsupported unary operator: {op_type.__name__}")
        result = _UNARY_OPS[op_type](_safe_eval(node.operand))
        _check_magnitude(result)
        return result
    # Explicitly rejects everything else -- Name (variables/builtins like
    # __import__), Call, Attribute, Subscript, comprehensions, Lambda,
    # strings, collections, etc. This is a closed allowlist (only the node
    # types handled above are permitted), not a denylist, so no new Python
    # syntax can slip through unnoticed.
    raise _CalculatorError(f"unsupported expression: {type(node).__name__}")


class CalculatorTool:
    name = "calculator"
    description = ("Evaluate a basic arithmetic expression using "
                   "+ - * / // % ** and parentheses.")
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The arithmetic expression to evaluate, e.g. '2 + 2 * 3'.",
            },
        },
        "required": ["expression"],
    }

    def run(self, arguments: dict, ctx: ToolContext) -> str:
        expression = arguments.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            return "error: 'expression' must be a non-empty string"
        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree)
            # str() is INSIDE the try: with the magnitude
            # bound above this is unreachable in practice, but a valid-but-huge
            # int would otherwise raise CPython's int->str ValueError straight
            # out to the caller. Defense in depth -- caught below as an error
            # string, never propagated.
            return str(result)
        except _CalculatorError as exc:
            return f"error: {exc}"
        except (SyntaxError, ZeroDivisionError, OverflowError,
                TypeError, ValueError) as exc:
            return f"error: {type(exc).__name__}: {exc}"


# -- get_current_time ---------------------------------------------------

class CurrentTimeTool:
    name = "get_current_time"
    description = "Get the current local date and time."
    parameters = {"type": "object", "properties": {}, "required": []}

    def run(self, arguments: dict, ctx: ToolContext) -> str:
        return datetime.datetime.now().isoformat(timespec="seconds")


# -- search_context: read-only query over the model's own attachments ----

class SearchContextTool:
    """On-thesis tool: lets the model retrieve from its own attached
    documents on demand, via the same doc_chunk segments the Context
    Inspector already shows. Read-only, no embeddings, no
    network -- simple case-insensitive substring matching."""

    name = "search_context"
    description = ("Search the documents attached to this conversation "
                   "for a query string, returning matching snippets.")
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Text to search for."},
            "k": {"type": "integer",
                 "description": "Maximum number of snippets to return (default 3)."},
        },
        "required": ["query"],
    }

    _SNIPPET_LEN = 200

    def run(self, arguments: dict, ctx: ToolContext) -> str:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return "error: 'query' must be a non-empty string"
        k = arguments.get("k", 3)
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            k = 3
        needle = query.lower()
        results: list[str] = []
        context_obj = ctx.ctx if ctx is not None else None
        if context_obj is not None:
            for seg in context_obj.segments:
                if seg.kind != SegmentKind.DOC_CHUNK:
                    continue
                if needle in seg.text.lower():
                    snippet = seg.text.strip()
                    if len(snippet) > self._SNIPPET_LEN:
                        snippet = snippet[: self._SNIPPET_LEN] + "..."
                    results.append(snippet)
                    if len(results) >= k:
                        break
        if not results:
            return "no matching context found"
        return "\n---\n".join(results)
