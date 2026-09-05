import pytest

from workbench.context.model import (ContextObject, EditEvent, Editor, Segment,
                                     SegmentKind, append_event)


def seg(text, kind=SegmentKind.USER_MSG, editable=Editor.BOTH):
    import uuid
    return Segment(id=uuid.uuid4().hex, kind=kind, text=text, editable_by=editable)


def test_append_and_replace():
    ctx = ContextObject()
    s = seg("hello")
    ctx.apply(append_event(s))
    ctx.apply(EditEvent(op="replace_text", segment_id=s.id,
                        payload={"text": "goodbye"}, actor="user"))
    assert ctx.segments[0].text == "goodbye"
    assert len(ctx.events) == 2


def test_permission_enforced():
    ctx = ContextObject()
    s = seg("locked", editable=Editor.USER)
    ctx.apply(append_event(s))
    with pytest.raises(PermissionError):
        ctx.apply(EditEvent(op="replace_text", segment_id=s.id,
                            payload={"text": "hacked"}, actor="model"))


def test_unknown_segment_raises():
    ctx = ContextObject()
    with pytest.raises(KeyError):
        ctx.apply(EditEvent(op="delete", segment_id="nope", payload={}, actor="user"))


def test_json_roundtrip_and_replay():
    ctx = ContextObject()
    a, b = seg("one"), seg("two")
    ctx.apply(append_event(a))
    ctx.apply(append_event(b))
    ctx.apply(EditEvent(op="move", segment_id=b.id, payload={"to_index": 0}, actor="user"))
    ctx.apply(EditEvent(op="delete", segment_id=a.id, payload={}, actor="user"))

    restored = ContextObject.from_json(ctx.to_json())
    assert [s.text for s in restored.segments] == ["two"]

    replayed = ContextObject.replay(ctx.events)
    assert [s.text for s in replayed.segments] == [s.text for s in ctx.segments]


# -- `move` gets a permission check, same as replace/delete -----


def test_move_permission_enforced():
    ctx = ContextObject()
    a, b = seg("one", editable=Editor.USER), seg("two", editable=Editor.USER)
    ctx.apply(append_event(a))
    ctx.apply(append_event(b))
    with pytest.raises(PermissionError):
        ctx.apply(EditEvent(op="move", segment_id=a.id,
                            payload={"to_index": 1}, actor="model"))
    # unchanged: the move must not have gone through
    assert [s.text for s in ctx.segments] == ["one", "two"]


def test_move_permitted_for_matching_actor():
    ctx = ContextObject()
    a, b = seg("one", editable=Editor.USER), seg("two", editable=Editor.USER)
    ctx.apply(append_event(a))
    ctx.apply(append_event(b))
    ctx.apply(EditEvent(op="move", segment_id=a.id, payload={"to_index": 1}, actor="user"))
    assert [s.text for s in ctx.segments] == ["two", "one"]


# -- duplicate segment id on append is rejected -----------------


def test_append_duplicate_id_raises():
    ctx = ContextObject()
    a = seg("one")
    ctx.apply(append_event(a))
    dup = Segment(id=a.id, kind=SegmentKind.USER_MSG, text="collide")
    with pytest.raises(ValueError):
        ctx.apply(append_event(dup))
    assert len(ctx.segments) == 1


# -- `move` to an out-of-range index raises, no silent clamp ---


def test_move_out_of_range_raises():
    ctx = ContextObject()
    a = seg("one")
    ctx.apply(append_event(a))
    with pytest.raises(IndexError):
        ctx.apply(EditEvent(op="move", segment_id=a.id,
                            payload={"to_index": 5}, actor="user"))


# -- append_event snapshots the segment (no live __dict__ alias) -


def test_append_event_snapshot_is_immune_to_later_mutation():
    s = seg("original")
    event = append_event(s, actor="user")
    s.text = "mutated after the event was built"
    assert event.payload["segment"]["text"] == "original"


def test_server_actor_may_edit_none_segments():
    """The privileged 'server' actor manages framing scaffolding (editable_by=
    NONE) that user/model actors cannot touch."""
    ctx = ContextObject()
    s = seg("scaffold", editable=Editor.NONE)
    ctx.apply(append_event(s))
    with pytest.raises(PermissionError):
        ctx.apply(EditEvent(op="delete", segment_id=s.id, payload={}, actor="user"))
    ctx.apply(EditEvent(op="delete", segment_id=s.id, payload={}, actor="server"))
    assert ctx.segments == []


# -- append gets a permission/provenance gate (prompt-injection) -


def _append_evt(s: Segment, actor: str) -> EditEvent:
    from dataclasses import asdict
    return EditEvent(op="append", segment_id=s.id, actor=actor,
                     payload={"segment": asdict(s)})


def test_user_append_benign_segment_ok():
    """A user may append a segment they'd be allowed to edit anyway, honestly
    attributed to themselves (editable_by=BOTH, provenance="user")."""
    ctx = ContextObject()
    s = seg("note", editable=Editor.BOTH)
    assert s.provenance == "user"
    ctx.apply(_append_evt(s, actor="user"))
    assert ctx.segments[0].text == "note"


def test_user_append_with_editable_by_none_rejected():
    """A user must not be able to append a segment they could never edit --
    e.g. forging editable_by=NONE to plant unremovable/unmodifiable content."""
    ctx = ContextObject()
    s = seg("sneaky", editable=Editor.NONE)
    with pytest.raises(PermissionError):
        ctx.apply(_append_evt(s, actor="user"))
    assert ctx.segments == []


def test_user_append_with_editable_by_model_rejected():
    """A user appending an editable_by=MODEL segment is also blocked (they
    could edit it later either way, but not append it in the first place)."""
    ctx = ContextObject()
    s = seg("sneaky", editable=Editor.MODEL)
    with pytest.raises(PermissionError):
        ctx.apply(_append_evt(s, actor="user"))
    assert ctx.segments == []


def test_user_append_with_forged_model_provenance_rejected():
    """editable_by=BOTH passes the permission check, but a user claiming
    provenance="model" is forging authorship -- this is the prompt-injection
    vector: content that *looks* model-authored (e.g. to a downstream
    prompt-trust heuristic) but was actually planted by the user."""
    ctx = ContextObject()
    s = seg("i am the model speaking", editable=Editor.BOTH)
    s.provenance = "model"
    with pytest.raises(PermissionError):
        ctx.apply(_append_evt(s, actor="user"))
    assert ctx.segments == []


def test_server_append_of_framing_ok():
    """The privileged 'server' actor bypasses both checks -- it legitimately
    appends framing scaffolding (editable_by=NONE, provenance="framing")."""
    ctx = ContextObject()
    s = Segment(id="scratch-1", kind=SegmentKind.SCRATCH, text="<user>",
               editable_by=Editor.NONE, provenance="framing")
    ctx.apply(_append_evt(s, actor="server"))
    assert ctx.segments[0].provenance == "framing"
    assert ctx.segments[0].editable_by == Editor.NONE


# -- Rewind: undo by replaying the log, not by storing snapshots -------------

def _ctx_with_two_segments():
    ctx = ContextObject()
    for i in range(2):
        s = Segment(id=f"s{i}", kind=SegmentKind.USER_MSG, text=f"text{i}",
                    provenance="user")
        ctx.apply(append_event(s, actor="user"))
    return ctx


def test_rewind_drops_later_events_and_restores_the_earlier_state():
    ctx = _ctx_with_two_segments()
    ctx.apply(EditEvent(op="replace_text", segment_id="s1",
                        payload={"text": "edited"}, actor="user"))
    assert ctx.segments[1].text == "edited"
    n_before_edit = len(ctx.events) - 1

    ctx.rewind_to(n_before_edit)

    assert ctx.segments[1].text == "text1"
    assert len(ctx.events) == n_before_edit


def test_rewind_is_in_place_so_existing_references_stay_valid():
    """The server and the manager both hold the same ContextObject; rewinding
    must not hand back a new object they would each have to be told about."""
    ctx = _ctx_with_two_segments()
    alias = ctx
    ctx.rewind_to(1)
    assert alias is ctx
    assert len(alias.segments) == 1


def test_rewind_past_the_start_empties_the_context():
    ctx = _ctx_with_two_segments()
    ctx.rewind_to(0)
    assert ctx.segments == []
    assert ctx.events == []


# -- Fork: branch a context without copying its state ------------------------

def test_fork_produces_an_independent_context_sharing_the_prefix():
    ctx = _ctx_with_two_segments()
    branch = ctx.fork_at(1)

    assert [s.id for s in branch.segments] == ["s0"]

    extra = Segment(id="b0", kind=SegmentKind.USER_MSG, text="branch only",
                    provenance="user")
    branch.apply(append_event(extra, actor="user"))

    # the branch moved; the original did not
    assert [s.id for s in branch.segments] == ["s0", "b0"]
    assert [s.id for s in ctx.segments] == ["s0", "s1"]


def test_fork_does_not_alias_the_parent_s_segments():
    """A shallow copy would let an edit on one branch rewrite the other."""
    ctx = _ctx_with_two_segments()
    branch = ctx.fork_at(2)

    branch.apply(EditEvent(op="replace_text", segment_id="s1",
                           payload={"text": "changed"}, actor="user"))

    assert branch.segments[1].text == "changed"
    assert ctx.segments[1].text == "text1"
