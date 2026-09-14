"""Structured provenance: who wrote a segment, where it came from, and what
produced it. The string form every existing call site passes is coerced, so
this is a type change and not a behaviour change."""
from dataclasses import asdict

from workbench.context.model import (
    ContextObject,
    Editor,
    Provenance,
    Segment,
    SegmentKind,
    append_event,
)


def test_a_bare_string_coerces_to_an_author():
    p = Provenance.coerce("user")

    assert p.author == "user"
    assert p.source == ""
    assert p.derived_from == ()


def test_an_attachment_string_keeps_its_whole_form_as_the_source():
    p = Provenance.coerce("attachment:notes.pdf")

    assert p.author == "attachment"
    assert p.source == "attachment:notes.pdf"


def test_a_provenance_passes_through_unchanged():
    p = Provenance(author="mira", source="spec.md", revision=2)

    assert Provenance.coerce(p) is p


def test_a_segment_built_with_a_string_still_works():
    seg = Segment(id="s1", kind=SegmentKind.USER_MSG, text="hi", provenance="user")

    assert seg.provenance.author == "user"


def test_provenance_survives_a_json_round_trip():
    ctx = ContextObject()
    seg = Segment(
        id="s1",
        kind=SegmentKind.USER_MSG,
        text="hi",
        provenance=Provenance(author="mira", source="spec.md", revision=2,
                              derived_from=("s0",)),
    )
    ctx.apply(append_event(seg))

    back = ContextObject.from_json(ctx.to_json())

    assert back.segments[0].provenance == seg.provenance


def test_the_append_gate_still_reads_the_author():
    ctx = ContextObject()
    seg = Segment(id="s1", kind=SegmentKind.USER_MSG, text="hi",
                  editable_by=Editor.USER, provenance="user")

    ctx.apply(append_event(seg, actor="user"))

    assert ctx.segments[0].provenance.author == "user"


def test_assigning_a_string_later_still_gives_a_record():
    seg = Segment(id="s1", kind=SegmentKind.USER_MSG, text="hi", provenance="user")

    seg.provenance = "model"

    assert isinstance(seg.provenance, Provenance)
    assert seg.provenance.author == "model"
