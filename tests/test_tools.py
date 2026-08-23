import pytest

from workbench.context.model import ContextObject, Editor, EditEvent, Segment, SegmentKind
from workbench.tools import ToolContext, ToolRegistry, default_registry
from workbench.tools.builtins import CalculatorTool, CurrentTimeTool, SearchContextTool


# --------------------------------------------------------------- calculator

@pytest.mark.parametrize("expr, expected", [
    ("2 + 2", "4"),
    ("2+2*3", "8"),
    ("(2 + 3) * 4", "20"),
    ("2 ** 10", "1024"),
    ("7 // 2", "3"),
    ("7 % 2", "1"),
    ("-5 + 2", "-3"),
    ("7 / 2", "3.5"),
])
def test_calculator_correct_math(expr, expected):
    tool = CalculatorTool()
    assert tool.run({"expression": expr}, ToolContext()) == expected


def test_calculator_rejects_import_call_safely():
    tool = CalculatorTool()
    result = tool.run({"expression": "__import__('os').system('echo hi')"},
                      ToolContext())
    assert result.startswith("error:")


def test_calculator_rejects_attribute_access_safely():
    tool = CalculatorTool()
    result = tool.run({"expression": "().__class__"}, ToolContext())
    assert result.startswith("error:")


def test_calculator_rejects_name_reference_safely():
    tool = CalculatorTool()
    result = tool.run({"expression": "x + 1"}, ToolContext())
    assert result.startswith("error:")


def test_calculator_rejects_huge_exponent():
    tool = CalculatorTool()
    result = tool.run({"expression": "10 ** 100000000"}, ToolContext())
    assert result.startswith("error:")


def test_calculator_rejects_nested_pow_tower_quickly():
    """Nesting grows the BASE even when every individual
    exponent is <= _MAX_EXPONENT, so the per-exponent cap alone does NOT stop
    ``((10**1000)**1000)**1000`` (~10^(10^9)) -- it would hang/OOM. The
    cumulative-magnitude bound on intermediate integer results must reject it
    fast (well under the 5s tool timeout), returning an ``error: ...`` string
    rather than computing the astronomically large value."""
    import time
    tool = CalculatorTool()
    start = time.monotonic()
    result = tool.run({"expression": "((10**1000)**1000)**1000"}, ToolContext())
    elapsed = time.monotonic() - start
    assert result.startswith("error:"), result
    assert "too large" in result, result
    # Must be rejected fast -- far below the 5s executor timeout that (per the
    # brief) can't rescue a CPU-bound thread anyway.
    assert elapsed < 5.0, f"took {elapsed:.2f}s -- magnitude bound not short-circuiting"


def test_calculator_result_magnitude_bound_str_never_raises():
    """Even a valid-but-huge result must never raise an
    uncaught ValueError out of run() -- the magnitude bound keeps results
    below CPython's int->str digit limit, and str(result) is inside the
    try/except as defense in depth. A result just over the bound is reported
    as an error string, not an exception."""
    tool = CalculatorTool()
    # (2**1000)**20 -> ~20020 bits: BOTH exponents (1000, 20) are within the
    # per-op exponent cap, so ONLY the cumulative-magnitude bound (~13000
    # bits) catches it. str() of the never-computed result can't raise.
    result = tool.run({"expression": "(2 ** 1000) ** 20"}, ToolContext())
    assert result.startswith("error:"), result
    assert "too large" in result, result


def test_calculator_rejects_division_by_zero_without_crashing():
    tool = CalculatorTool()
    result = tool.run({"expression": "1 / 0"}, ToolContext())
    assert result.startswith("error:")


def test_calculator_rejects_non_string_expression():
    tool = CalculatorTool()
    assert tool.run({"expression": 5}, ToolContext()).startswith("error:")
    assert tool.run({}, ToolContext()).startswith("error:")


# ----------------------------------------------------------- get_current_time

def test_get_current_time_returns_a_string():
    tool = CurrentTimeTool()
    result = tool.run({}, ToolContext())
    assert isinstance(result, str) and result
    import datetime
    datetime.datetime.fromisoformat(result)  # must round-trip


# --------------------------------------------------------------- search_context

def _ctx_with_doc_chunks(*texts: str) -> ContextObject:
    ctx = ContextObject()
    for i, text in enumerate(texts):
        seg = Segment(id=f"chunk-{i}", kind=SegmentKind.DOC_CHUNK, text=text,
                      editable_by=Editor.USER, provenance="attachment:notes.txt")
        ctx.apply(EditEvent(op="append", segment_id=seg.id,
                            payload={"segment": seg.__dict__}, actor="server"))
    return ctx


def test_search_context_finds_matching_doc_chunk_text():
    ctx = _ctx_with_doc_chunks("the quick brown fox", "jumps over the lazy dog")
    tool = SearchContextTool()
    result = tool.run({"query": "quick"}, ToolContext(ctx=ctx))
    assert "quick" in result.lower()


def test_search_context_is_case_insensitive():
    ctx = _ctx_with_doc_chunks("The Quick Brown Fox")
    tool = SearchContextTool()
    result = tool.run({"query": "QUICK"}, ToolContext(ctx=ctx))
    assert "quick" in result.lower()


def test_search_context_misses_non_matching_query():
    ctx = _ctx_with_doc_chunks("the quick brown fox")
    tool = SearchContextTool()
    result = tool.run({"query": "elephant"}, ToolContext(ctx=ctx))
    assert "no matching" in result.lower()


def test_search_context_ignores_non_doc_chunk_segments():
    ctx = ContextObject()
    seg = Segment(id="s1", kind=SegmentKind.USER_MSG, text="quick fox",
                  editable_by=Editor.BOTH, provenance="user")
    ctx.apply(EditEvent(op="append", segment_id=seg.id,
                        payload={"segment": seg.__dict__}, actor="user"))
    tool = SearchContextTool()
    result = tool.run({"query": "quick"}, ToolContext(ctx=ctx))
    assert "no matching" in result.lower()


def test_search_context_respects_k_limit():
    ctx = _ctx_with_doc_chunks("apple pie", "apple tart", "apple cake", "apple bread")
    tool = SearchContextTool()
    result = tool.run({"query": "apple", "k": 2}, ToolContext(ctx=ctx))
    assert len(result.split("\n---\n")) == 2


def test_search_context_no_context_available():
    tool = SearchContextTool()
    result = tool.run({"query": "anything"}, ToolContext())
    assert "no matching" in result.lower()


def test_search_context_rejects_empty_query():
    tool = SearchContextTool()
    assert tool.run({"query": ""}, ToolContext()).startswith("error:")


# ------------------------------------------------------------------ registry

def test_registry_register_get_list():
    registry = ToolRegistry()
    calc = CalculatorTool()
    registry.register(calc)
    assert registry.get("calculator") is calc
    assert registry.get("nonexistent") is None
    assert registry.list() == [calc]


def test_default_registry_has_three_tools():
    registry = default_registry()
    names = {t.name for t in registry.list()}
    assert names == {"calculator", "get_current_time", "search_context"}


def test_registry_schemas_shape_matches_openai_function_format():
    registry = default_registry()
    schemas = registry.schemas()
    assert len(schemas) == 3
    for schema in schemas:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert set(fn) == {"name", "description", "parameters"}
        assert isinstance(fn["name"], str) and fn["name"]
        assert isinstance(fn["description"], str) and fn["description"]
        assert isinstance(fn["parameters"], dict)
        assert fn["parameters"]["type"] == "object"
    names = {s["function"]["name"] for s in schemas}
    assert names == {"calculator", "get_current_time", "search_context"}


def test_registry_execute_runs_registered_tool():
    registry = default_registry()
    result = registry.execute("calculator", {"expression": "2+2"}, ToolContext())
    assert result == "4"


def test_registry_execute_unknown_tool_returns_error_not_crash():
    registry = default_registry()
    result = registry.execute("nonexistent_tool", {}, ToolContext())
    assert result.startswith("error:")


def test_registry_execute_missing_required_argument_returns_error():
    registry = default_registry()
    result = registry.execute("calculator", {}, ToolContext())
    assert result.startswith("error:")


def test_registry_execute_catches_tool_exceptions():
    class ExplodingTool:
        name = "exploder"
        description = "always raises"
        parameters = {"type": "object", "properties": {}, "required": []}

        def run(self, arguments, ctx):
            raise RuntimeError("boom")

    registry = ToolRegistry()
    registry.register(ExplodingTool())
    result = registry.execute("exploder", {}, ToolContext())
    assert result.startswith("error:") and "boom" in result


def test_registry_render_prompt_block_contains_tool_names_and_tags():
    registry = default_registry()
    block = registry.render_prompt_block()
    assert "<tools>" in block and "</tools>" in block
    assert "<tool_call>" in block and "</tool_call>" in block
    assert "calculator" in block
    assert "get_current_time" in block
    assert "search_context" in block


def test_empty_registry_render_prompt_block_is_empty():
    registry = ToolRegistry()
    assert registry.render_prompt_block() == ""
