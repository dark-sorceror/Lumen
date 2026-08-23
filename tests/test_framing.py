from workbench.server.framing import frame_tool_result
import pytest

from workbench.context.manager import ContextManager
from workbench.context.model import (ContextObject, Editor, SegmentKind,
                                     append_event)
from workbench.server.framing import (frame_message, 
                                      generation_prompt_segment)


def test_frame_message_wraps_content_in_scratch_framing(fake_tokenizer):
    prefix, content, suffix = frame_message(fake_tokenizer, "user", "1 2 3")
    assert prefix.kind == SegmentKind.SCRATCH
    assert suffix.kind == SegmentKind.SCRATCH
    assert content.kind == SegmentKind.USER_MSG
    assert content.text == "1 2 3"
    ids = {prefix.id, content.id, suffix.id}
    assert len(ids) == 3  # all fresh, distinct ids


def test_frame_message_framing_segments_are_editable_by_none(fake_tokenizer):
    prefix, content, suffix = frame_message(fake_tokenizer, "user", "1 2 3")
    assert prefix.editable_by == Editor.NONE
    assert suffix.editable_by == Editor.NONE
    assert content.editable_by == Editor.BOTH


def test_frame_message_assistant_role_uses_assistant_msg_kind(fake_tokenizer):
    _, content, _ = frame_message(fake_tokenizer, "assistant", "5 6")
    assert content.kind == SegmentKind.ASSISTANT_MSG
    assert content.provenance == "model"


def test_frame_message_user_role_provenance_is_user(fake_tokenizer):
    _, content, _ = frame_message(fake_tokenizer, "user", "5 6")
    assert content.provenance == "user"


def test_framed_tokens_match_single_message_chat_template(fake_tokenizer):
    """The concatenation of [prefix, content, suffix] token-encodes to exactly
    what apply_chat_template produces for that one message (no generation
    prompt)."""
    ctx = ContextObject()
    for seg in frame_message(fake_tokenizer, "user", "1 2 3"):
        ctx.apply(append_event(seg))
    tc = ContextManager(ctx, fake_tokenizer).to_tokens()
    expected = fake_tokenizer.apply_chat_template(
        [{"role": "user", "content": "1 2 3"}], tokenize=True, add_generation_prompt=False)
    assert tc.tokens == expected


def test_generation_prompt_segment_is_scratch_and_editable_by_none(fake_tokenizer):
    seg = generation_prompt_segment(fake_tokenizer)
    assert seg.kind == SegmentKind.SCRATCH
    assert seg.editable_by == Editor.NONE


def test_generation_prompt_segment_matches_chat_template_tail(fake_tokenizer):
    """Appending the generation-prompt segment after a framed user message must
    make manager.to_tokens() equal apply_chat_template(..., add_generation_prompt=True)."""
    ctx = ContextObject()
    for seg in frame_message(fake_tokenizer, "user", "1 2 3"):
        ctx.apply(append_event(seg))
    ctx.apply(append_event(generation_prompt_segment(fake_tokenizer)))
    tc = ContextManager(ctx, fake_tokenizer).to_tokens()
    expected = fake_tokenizer.apply_chat_template(
        [{"role": "user", "content": "1 2 3"}], tokenize=True, add_generation_prompt=True)
    assert tc.tokens == expected


# -- Tool calling: frame_tool_result --------------------------------


def test_framing_computation_is_cached_per_role(fake_tokenizer):
    """Framing is deterministic given a role, so repeated frame_message calls
    for the same (tokenizer, role) must not re-render the chat template."""
    calls = {"n": 0}
    real_apply = fake_tokenizer.apply_chat_template

    def counting_apply(*args, **kwargs):
        calls["n"] += 1
        return real_apply(*args, **kwargs)

    fake_tokenizer.apply_chat_template = counting_apply
    frame_message(fake_tokenizer, "user", "1 2 3")
    frame_message(fake_tokenizer, "user", "9 9 9")
    frame_message(fake_tokenizer, "user", "7")
    assert calls["n"] == 1


# -- id(tokenizer) cache guarded against address reuse --------


def test_framing_cache_guards_against_tokenizer_id_reuse():
    """`_framing_cache` is keyed by id(tokenizer), and `id()` is only unique
    among currently-live objects -- if a tokenizer were garbage-collected and
    a *different* tokenizer happened to be allocated at the same address, a
    naive id()-keyed cache would silently serve the OLD tokenizer's framing
    text for the NEW one. Simulate that address-reuse collision directly
    (rather than relying on GC/allocator timing, which isn't guaranteed) by
    planting a cache entry under a live tokenizer's id but backed by a
    weakref to a *different* object, and confirm the guard detects the
    mismatch and recomputes instead of returning the stale text."""
    import weakref

    from workbench.server import framing

    class T:
        def __init__(self, tag):
            self.tag = tag

        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=False, **kw):
            return f"<{self.tag}>{messages[0]['content']}</{self.tag}>"

    stale_owner = T("stale")
    fresh = T("fresh")
    # Plant a cache entry keyed by `fresh`'s id, but weakref'd to a DIFFERENT
    # object -- as if this slot were last computed for a tokenizer that has
    # since been collected and whose address `fresh` now happens to reuse.
    framing._framing_cache[(id(fresh), "user")] = (
        weakref.ref(stale_owner), "STALE_PREFIX", "STALE_SUFFIX")

    prefix, suffix = framing._framing_for_role(fresh, "user")

    assert (prefix, suffix) != ("STALE_PREFIX", "STALE_SUFFIX")
    assert prefix == "<fresh>"
    assert suffix == "</fresh>"


def test_generation_prompt_cache_guards_against_tokenizer_id_reuse():
    """Same guard, for `_generation_prompt_cache` / `generation_prompt_text`."""
    import weakref

    from workbench.server import framing

    class T:
        def __init__(self, tag):
            self.tag = tag

        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=False, **kw):
            text = messages[0]["content"]
            return text + f"<{self.tag}-gp>" if add_generation_prompt else text

    stale_owner = T("stale")
    fresh = T("fresh")
    framing._generation_prompt_cache[id(fresh)] = (
        weakref.ref(stale_owner), "STALE_GP_TEXT")

    text = framing.generation_prompt_text(fresh)

    assert text != "STALE_GP_TEXT"
    assert text == "<fresh-gp>"


@pytest.mark.slow
def test_framed_2turn_conversation_matches_real_chat_template():
    """Exact equivalence being asserted: build a ContextObject by hand out of
    frame_message()/generation_prompt_segment() calls for a 2-turn
    conversation (user1 -> assistant1 -> user2, then a trailing
    generation-prompt segment as if about to generate turn 2's reply) and
    assert ContextManager.to_tokens().tokens equals
    tokenizer.apply_chat_template([msg1, reply1, msg2], tokenize=True,
    add_generation_prompt=True) on the real test model -- i.e. framing
    reconstructs the chat template byte-for-byte at the token level,
    INCLUDING the generation-prompt tail (not excluding it)."""
    from workbench.engine.loader import TEST_MODEL, load_model

    _, tokenizer = load_model(TEST_MODEL)

    msg1 = {"role": "user", "content": "Hello there"}
    reply1 = {"role": "assistant", "content": "Hi! How can I help?"}
    msg2 = {"role": "user", "content": "Whats the weather"}

    ctx = ContextObject()
    for seg in frame_message(tokenizer, "user", msg1["content"]):
        ctx.apply(append_event(seg))
    for seg in frame_message(tokenizer, "assistant", reply1["content"]):
        ctx.apply(append_event(seg))
    for seg in frame_message(tokenizer, "user", msg2["content"]):
        ctx.apply(append_event(seg))
    ctx.apply(append_event(generation_prompt_segment(tokenizer)))

    tc = ContextManager(ctx, tokenizer).to_tokens()
    expected = tokenizer.apply_chat_template(
        [msg1, reply1, msg2], tokenize=True, add_generation_prompt=True)
    assert tc.tokens == expected


def test_frame_tool_result_wraps_content_in_scratch_framing(fake_tokenizer):
    prefix, content, suffix = frame_tool_result(fake_tokenizer, "calculator", "4")
    assert prefix.kind == SegmentKind.SCRATCH
    assert suffix.kind == SegmentKind.SCRATCH
    assert content.kind == SegmentKind.TOOL_RESULT
    assert content.text == "4"
    ids = {prefix.id, content.id, suffix.id}
    assert len(ids) == 3


def test_frame_tool_result_provenance_and_editability(fake_tokenizer):
    _, content, _ = frame_tool_result(fake_tokenizer, "calculator", "4")
    assert content.provenance == "tool:calculator"
    assert content.editable_by == Editor.USER
    prefix, _, suffix = frame_tool_result(fake_tokenizer, "calculator", "4")
    assert prefix.editable_by == Editor.NONE
    assert suffix.editable_by == Editor.NONE
    assert prefix.provenance == "framing" and suffix.provenance == "framing"


def test_frame_tool_result_uses_distinct_role_framing_from_user(fake_tokenizer):
    """The "tool" role's framing must be derived independently of "user"'s
    (distinct dict entry / cache key), even though real Qwen3 renders a tool
    response wrapped in a `user` turn -- confirming frame_tool_result doesn't
    accidentally alias the "user" cache slot."""
    user_prefix, _, user_suffix = frame_message(fake_tokenizer, "user", "hi")
    tool_prefix, _, tool_suffix = frame_tool_result(fake_tokenizer, "calculator", "4")
    assert (tool_prefix.text, tool_suffix.text) != (user_prefix.text, user_suffix.text)
