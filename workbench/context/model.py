"""The Context Object: one event-sourced data structure that the chat UI, the editor,
emphasis sliders, and the model's self-edit tools all operate on."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from enum import Enum


class SegmentKind(str, Enum):
    SYSTEM = "system"
    USER_MSG = "user_msg"
    ASSISTANT_MSG = "assistant_msg"
    THOUGHT = "thought"
    DOC_CHUNK = "doc_chunk"
    SCRATCH = "scratch"
    # Tool calling: a tool's return string, framed with the
    # tokenizer's <tool_response> wrapper (see server/framing.py's
    # frame_tool_result). Distinct from DOC_CHUNK (user-attached reference
    # data) and SCRATCH (framing-only, editable_by=NONE) -- a tool result is
    # real content the user may want to inspect/edit, and the Inspector can
    # badge it distinctly. provenance is f"tool:{name}"; editable_by=USER.
    TOOL_RESULT = "tool_result"


class Editor(str, Enum):
    USER = "user"
    MODEL = "model"
    BOTH = "both"
    NONE = "none"


_MAY_EDIT = {
    "user": {Editor.USER, Editor.BOTH},
    "model": {Editor.MODEL, Editor.BOTH},
    # The server owns segment lifecycle (framing scaffolding like the
    # generation-prompt SCRATCH segment, which is editable_by=NONE for
    # user/model). Never accept "server" as an actor from the wire.
    "server": {Editor.USER, Editor.MODEL, Editor.BOTH, Editor.NONE},
}


@dataclass(frozen=True)
class Provenance:
    """Where a segment came from, as structure rather than a sentence.

    This was a free string (`"user"`, `"attachment:notes.pdf"`, `"tool:calc"`).
    The string is kept verbatim as `source` so nothing that formatted or
    matched on it changes, and the parts worth ranking on are lifted out.

    `derived_from` is the edge no hosted API can offer: the segments that were
    in the projection when a generation produced this one. It makes taint
    propagate -- if a segment is later superseded, everything generated from it
    is suspect -- without any semantic analysis."""
    author: str = "user"
    source: str = ""
    revision: int = 0
    derived_from: tuple[str, ...] = ()
    op_seq: int = -1

    @classmethod
    def coerce(cls, value: object) -> "Provenance":
        """Accept a Provenance, the legacy string, or a replayed dict.

        Replay and from_json both hand back plain dicts, and every existing
        call site passes a string; coercing in one place is what makes this a
        type change rather than a rewrite of the call sites."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            author, _, _rest = value.partition(":")
            return cls(author=author, source=value if _rest else "")
        if isinstance(value, dict):
            data = dict(value)
            author = data.get("author", "user")
            if not isinstance(author, str):
                raise TypeError("provenance author must be a string")
            derived = data.get("derived_from", ())
            if isinstance(derived, str) or not all(
                    isinstance(sid, str) for sid in derived):
                raise TypeError(
                    "provenance derived_from must be a sequence of segment ids")
            data["derived_from"] = tuple(derived)
            # bool is an int subclass; True would pass as revision 1.
            revision = data.get("revision", 0)
            if isinstance(revision, bool) or not isinstance(revision, int):
                raise TypeError("provenance revision must be an int")
            unknown = set(data) - {f.name for f in fields(cls)}
            if unknown:
                raise TypeError(
                    f"unknown provenance key {sorted(unknown)[0]!r}")
            return cls(**data)
        raise TypeError(f"cannot read provenance from {type(value).__name__}")

    @property
    def legacy(self) -> str:
        """The free string this record replaced: the source when there is one,
        else the author. Wire and display code that predates the record reads it."""
        return self.source or self.author


@dataclass
class Segment:
    id: str
    kind: SegmentKind
    text: str
    emphasis: float = 0.0
    editable_by: Editor = Editor.BOTH
    provenance: Provenance = field(default_factory=Provenance)

    def __setattr__(self, name: str, value: object) -> None:
        # Coerce on ASSIGNMENT, not only in __post_init__: the dataclass's own
        # __init__ assigns through here too, so this one mechanism covers
        # construction and later writes alike. Without it `seg.provenance =
        # "model"` leaves a bare string on a field every later reader treats
        # as a record, and the AttributeError surfaces somewhere else entirely.
        if name == "provenance":
            value = Provenance.coerce(value)
        object.__setattr__(self, name, value)


@dataclass
class EditEvent:
    op: str
    segment_id: str
    payload: dict
    actor: str


@dataclass
class ContextObject:
    segments: list[Segment] = field(default_factory=list)
    events: list[EditEvent] = field(default_factory=list)

    def _index_of(self, segment_id: str) -> int:
        for i, s in enumerate(self.segments):
            if s.id == segment_id:
                return i
        raise KeyError(segment_id)

    def _check_permission(self, segment: Segment, actor: str) -> None:
        if segment.editable_by not in _MAY_EDIT.get(actor, set()):
            raise PermissionError(
                f"{actor} may not edit segment {segment.id} ({segment.editable_by})")

    def apply(self, event: EditEvent) -> None:
        if event.op == "append":
            try:
                data = dict(event.payload["segment"])
                data["kind"] = SegmentKind(data["kind"])
                data["editable_by"] = Editor(data["editable_by"])
                segment = Segment(**data)
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"malformed segment payload: {e}") from e
            if any(s.id == segment.id for s in self.segments):
                raise ValueError(f"duplicate segment id: {segment.id}")
            # Prompt-injection gate: a non-"server" actor may only append a
            # segment it would itself be allowed to edit (blocks e.g. a user
            # appending an editable_by=NONE/MODEL segment), AND the segment's
            # claimed provenance must equal the actor (blocks forging
            # provenance="model"/"system"/"framing" to smuggle content that
            # looks model- or system-authored). "server" is the privileged
            # actor that legitimately appends framing scaffolding (provenance
            # "framing"/"model", editable_by NONE) and is exempt from both
            # checks -- see _MAY_EDIT's comment.
            if event.actor != "server":
                if segment.editable_by not in _MAY_EDIT.get(event.actor, set()):
                    raise PermissionError(
                        f"{event.actor} may not append a segment with "
                        f"editable_by={segment.editable_by.value}")
                # The whole record must be the actor's own, not merely its
                # author. Checking .author alone lets a user append
                # {"author": "user", "source": "tool:calculator"}, which passes
                # and then reaches every client as tool-authored. Comparing
                # against Provenance(author=actor) is the old whole-string
                # semantics exactly, and also refuses a forged source,
                # derived_from, revision or op_seq from the wire.
                if segment.provenance != Provenance(author=event.actor):
                    raise PermissionError(
                        f"{event.actor} may not append a segment with "
                        f"provenance {segment.provenance.legacy!r}")
            self.segments.append(segment)
        elif event.op == "replace_text":
            i = self._index_of(event.segment_id)
            self._check_permission(self.segments[i], event.actor)
            self.segments[i].text = event.payload["text"]
        elif event.op == "delete":
            i = self._index_of(event.segment_id)
            self._check_permission(self.segments[i], event.actor)
            del self.segments[i]
        elif event.op == "move":
            i = self._index_of(event.segment_id)
            self._check_permission(self.segments[i], event.actor)
            to_index = event.payload["to_index"]
            if not (0 <= to_index < len(self.segments)):
                raise IndexError(f"move to_index out of range: {to_index}")
            seg = self.segments.pop(i)
            self.segments.insert(to_index, seg)
        else:
            raise ValueError(f"unknown op: {event.op}")
        self.events.append(event)

    def rewind_to(self, n_events: int) -> None:
        """Rebuild this context from the first `n_events` of its own log.

        Undo by REPLAY rather than by snapshot: the log is the cheap thing to
        keep, and rebuilding from it is exact by construction. Mutates in
        place, because the server and the ContextManager both hold this same
        object -- handing back a new one would leave one of them stale."""
        rebuilt = ContextObject.replay(self.events[:n_events])
        self.segments = rebuilt.segments
        self.events = rebuilt.events

    def fork_at(self, n_events: int) -> "ContextObject":
        """An independent context replayed from the first `n_events` of this log.

        Branching is cheap for the same reason undo is: the log is small and
        replay is exact, so there is nothing to snapshot. Because the branch
        shares a token prefix with its parent, the engine's existing
        common-prefix cache reuse makes switching between them cheap too --
        only the diverging tail re-prefills."""
        return ContextObject.replay(self.events[:n_events])

    def to_json(self) -> str:
        return json.dumps({"events": [asdict(e) for e in self.events]})

    @classmethod
    def from_json(cls, s: str) -> "ContextObject":
        events = [EditEvent(**e) for e in json.loads(s)["events"]]
        return cls.replay(events)

    @classmethod
    def replay(cls, events: list[EditEvent]) -> "ContextObject":
        ctx = cls()
        for e in events:
            ctx.apply(e)
        return ctx


def append_event(segment: Segment, actor: str = "server") -> EditEvent:
    """Build an 'append' EditEvent for `segment`, snapshotting it via
    `dataclasses.asdict` rather than aliasing `segment.__dict__`. Callers
    (server/framing code) that build segments and then keep a reference to
    them must not be able to retroactively mutate an already-recorded event's
    payload -- to_json()/replay() must reflect the segment as it was at the
    moment this event was created, not whatever it later becomes.

    Defaults to the privileged "server" actor (bypasses the append
    permission/provenance gate in `ContextObject.apply`) since this helper is
    typically used to bootstrap/replay segments directly rather than to
    model a real user- or model-originated wire edit; pass an explicit
    `actor` to exercise the gate."""
    return EditEvent(op="append", segment_id=segment.id, actor=actor,
                     payload={"segment": asdict(segment)})
