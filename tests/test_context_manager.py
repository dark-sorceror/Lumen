from hypothesis import given, strategies as st

from workbench.context.manager import ContextManager, CacheImpact
from workbench.context.model import ContextObject, EditEvent, Segment, SegmentKind


def build_ctx(texts):
    ctx = ContextObject()
    for i, t in enumerate(texts):
        s = Segment(id=f"s{i}", kind=SegmentKind.USER_MSG, text=t)
        ctx.apply(EditEvent(op="append", segment_id=s.id,
                            payload={"segment": s.__dict__}, actor="user"))
    return ctx


def test_spans_cover_tokens_exactly(fake_tokenizer):
    # FakeTokenizer.encode: "1 2 3" -> [1, 2, 3]
    ctx = build_ctx(["1 2", "3 4 5"])
    tc = ContextManager(ctx, fake_tokenizer).to_tokens()
    assert tc.tokens == [1, 2, 3, 4, 5]
    assert tc.spans == {"s0": (0, 2), "s1": (2, 5)}


@given(st.lists(st.lists(st.integers(0, 9), min_size=1, max_size=5),
                min_size=1, max_size=6))
def test_property_concat_equals_per_segment_encode(seg_token_lists):
    from tests.conftest import FakeTokenizer
    tok = FakeTokenizer()
    texts = [" ".join(str(t) for t in toks) for toks in seg_token_lists]
    tc = ContextManager(build_ctx(texts), tok).to_tokens()
    assert tc.tokens == [t for toks in seg_token_lists for t in toks]
    ends = [tc.spans[f"s{i}"][1] for i in range(len(texts))]
    starts = [tc.spans[f"s{i}"][0] for i in range(len(texts))]
    assert starts == [0] + ends[:-1]          # spans are contiguous, in order


def test_edit_first_segment_invalidates_everything(fake_tokenizer):
    ctx = build_ctx(["1 2", "3 4 5"])
    mgr = ContextManager(ctx, fake_tokenizer)
    impact = mgr.preview_edit(EditEvent(op="replace_text", segment_id="s0",
                                        payload={"text": "9 2"}, actor="user"))
    assert impact == CacheImpact(first_invalid_token=0, tokens_to_reprefill=5)
    assert ctx.segments[0].text == "1 2"      # preview did not mutate


def test_edit_last_segment_keeps_prefix(fake_tokenizer):
    ctx = build_ctx(["1 2", "3 4 5"])
    mgr = ContextManager(ctx, fake_tokenizer)
    impact = mgr.apply_edit(EditEvent(op="replace_text", segment_id="s1",
                                      payload={"text": "3 4 9 9"}, actor="user"))
    assert impact.first_invalid_token == 4    # "1 2 3 4" prefix survives
    assert impact.tokens_to_reprefill == 2    # new tokens "9 9"
    assert ctx.segments[1].text == "3 4 9 9"  # apply DID mutate


def test_append_costs_nothing(fake_tokenizer):
    ctx = build_ctx(["1 2"])
    mgr = ContextManager(ctx, fake_tokenizer)
    s = Segment(id="s9", kind=SegmentKind.USER_MSG, text="7 8")
    impact = mgr.preview_edit(EditEvent(op="append", segment_id="s9",
                                        payload={"segment": s.__dict__}, actor="user"))
    assert impact.first_invalid_token == 2
    assert impact.tokens_to_reprefill == 2
