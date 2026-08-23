"""The Context Object: one event-sourced data structure that the chat UI, the editor,
emphasis sliders, and the model's self-edit tools all operate on."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
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


@dataclass
class Segment:
    id: str
    kind: SegmentKind
    text: str
    emphasis: float = 0.0
    editable_by: Editor = Editor.BOTH
    provenance: str = "user"


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
                if segment.provenance != event.actor:
                    raise PermissionError(
                        f"{event.actor} may not append a segment with "
                        f"provenance {segment.provenance!r}")
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
