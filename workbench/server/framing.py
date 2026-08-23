"""Chat-template framing: wraps a message's bare text in the SCRATCH segments
the tokenizer's chat template would otherwise add invisibly (role tags,
turn separators, the assistant generation-prompt header).

The framing text for a given (tokenizer, role) pair is derived once by
rendering `apply_chat_template` for a single placeholder message and diffing
the rendered text against the placeholder -- this is deterministic given the
role, so it's computed once and cached by design rather than
re-rendered on every message.
"""
from __future__ import annotations

import uuid
import weakref

from workbench.context.model import Editor, Segment, SegmentKind

_ROLE_KIND = {
    "user": SegmentKind.USER_MSG,
    "assistant": SegmentKind.ASSISTANT_MSG,
    "system": SegmentKind.SYSTEM,
}

_ROLE_PROVENANCE = {"user": "user", "assistant": "model", "system": "system"}

# A sentinel unlikely to collide with real message content, used to locate
# where the role's framing prefix ends and suffix begins in the rendered
# template (diffing against the bare text, using a
# placeholder instead of the real per-call text so the result -- prefix and
# suffix strings -- can be cached across calls with different real content).
_PLACEHOLDER = "FRAME_PLACEHOLDER"

# Cache keyed by (id(tokenizer), role) -> (prefix_text, suffix_text). Framing
# is deterministic per (tokenizer, role); this avoids re-rendering the chat
# template for every message.
#
# `id()` is only unique among currently-live objects: if a tokenizer is
# garbage-collected and a *different* tokenizer object happens to be
# allocated at the same address, a naive id()-keyed cache would silently
# serve stale framing text for the wrong tokenizer. Each cache entry also
# stores a weakref to the tokenizer it was computed for; a lookup only
# counts as a hit if that weakref still resolves to the exact object passed
# in, otherwise the entry is treated as stale and recomputed.
_framing_cache: dict[tuple[int, str], tuple["weakref.ReferenceType", str, str]] = {}
_generation_prompt_cache: dict[int, tuple["weakref.ReferenceType", str]] = {}


def _guard_ref(obj):
    try:
        return weakref.ref(obj)
    except TypeError:
        # Object doesn't support weakrefs (rare for tokenizer-like objects).
        # Fall back to an always-matching "ref": this keeps `obj` alive for
        # the life of the cache (same as the old id()-only behavior) and
        # accepts the narrow id-reuse risk for such objects.
        return lambda: obj


def _framing_for_role(tokenizer, role: str) -> tuple[str, str]:
    key = (id(tokenizer), role)
    cached = _framing_cache.get(key)
    if cached is not None:
        ref, prefix, suffix = cached
        if ref() is tokenizer:
            return prefix, suffix
    rendered = tokenizer.apply_chat_template(
        [{"role": role, "content": _PLACEHOLDER}],
        tokenize=False,
        add_generation_prompt=False,
    )
    idx = rendered.index(_PLACEHOLDER)
    prefix, suffix = rendered[:idx], rendered[idx + len(_PLACEHOLDER):]
    _framing_cache[key] = (_guard_ref(tokenizer), prefix, suffix)
    return prefix, suffix


def frame_message(tokenizer, role: str, text: str) -> list[Segment]:
    """Returns [prefix SCRATCH, content <role>_MSG, suffix SCRATCH] segments.
    The two SCRATCH segments carry the chat-template framing around `text`
    and are not editable by anyone (editable_by=NONE); the content segment
    is editable_by=BOTH regardless of role."""
    prefix_text, suffix_text = _framing_for_role(tokenizer, role)
    kind = _ROLE_KIND[role]
    provenance = _ROLE_PROVENANCE[role]
    return [
        Segment(id=uuid.uuid4().hex, kind=SegmentKind.SCRATCH, text=prefix_text,
               editable_by=Editor.NONE, provenance="framing"),
        Segment(id=uuid.uuid4().hex, kind=kind, text=text,
               editable_by=Editor.BOTH, provenance=provenance),
        Segment(id=uuid.uuid4().hex, kind=SegmentKind.SCRATCH, text=suffix_text,
               editable_by=Editor.NONE, provenance="framing"),
    ]


def frame_tool_result(tokenizer, name: str, result_text: str) -> list[Segment]:
    """Returns [prefix SCRATCH, content TOOL_RESULT, suffix SCRATCH] segments
    for a tool's return value, mirroring frame_message() but for the
    Hermes/Qwen3 tool-response wrapper (role "tool" -> rendered as a `user`
    turn containing `<tool_response>...</tool_response>`). The prefix/suffix text is derived the same way frame_message
    derives its role framing: render apply_chat_template for a single
    placeholder {"role": "tool", "content": PLACEHOLDER} message and diff
    against the placeholder -- reusing `_framing_for_role`'s cache (keyed by
    (tokenizer, "tool"), distinct from the "user"/"assistant"/"system" role
    entries in _ROLE_KIND/_ROLE_PROVENANCE, which frame_message uses).

    Unlike frame_message, the content segment's provenance is NOT a fixed
    per-role value -- it's f"tool:{name}" (so the Inspector can badge which
    tool produced it), and editable_by=USER (a tool result is real content,
    not framing scaffolding). Both are appended by the server with the
    privileged actor="server" (see app.py's _append_tool_result_segments),
    which is exempt from the provenance==actor gate in ContextObject.apply,
    so a provenance of "tool:calculator" (neither "server" nor "user") is
    legitimate."""
    prefix_text, suffix_text = _framing_for_role(tokenizer, "tool")
    return [
        Segment(id=uuid.uuid4().hex, kind=SegmentKind.SCRATCH, text=prefix_text,
               editable_by=Editor.NONE, provenance="framing"),
        Segment(id=uuid.uuid4().hex, kind=SegmentKind.TOOL_RESULT, text=result_text,
               editable_by=Editor.USER, provenance=f"tool:{name}"),
        Segment(id=uuid.uuid4().hex, kind=SegmentKind.SCRATCH, text=suffix_text,
               editable_by=Editor.NONE, provenance="framing"),
    ]


def generation_prompt_text(tokenizer) -> str:
    """The literal text the chat template appends when `add_generation_prompt=
    True` (e.g. the assistant turn's opening header) -- derived by diffing the
    template rendered with and without it, for a placeholder single-message
    conversation. Cached per tokenizer (role-independent: chat templates emit
    one fixed generation-prompt tail regardless of what came before)."""
    cached = _generation_prompt_cache.get(id(tokenizer))
    if cached is not None:
        ref, text = cached
        if ref() is tokenizer:
            return text
    message = [{"role": "user", "content": _PLACEHOLDER}]
    with_gp = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    without_gp = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
    if not with_gp.startswith(without_gp):
        raise ValueError(
            "generation_prompt_text: chat template's add_generation_prompt=True "
            "output is not a pure suffix of its add_generation_prompt=False output "
            "-- this tokenizer's template needs a different framing strategy")
    text = with_gp[len(without_gp):]
    _generation_prompt_cache[id(tokenizer)] = (_guard_ref(tokenizer), text)
    return text


def generation_prompt_segment(tokenizer) -> Segment:
    """A SCRATCH, editable_by=NONE segment holding the assistant
    generation-prompt header, to be appended right before starting
    generation and later replaced by the real assistant frame_message()
    segments once the reply text is known."""
    return Segment(id=uuid.uuid4().hex, kind=SegmentKind.SCRATCH,
                  text=generation_prompt_text(tokenizer),
                  editable_by=Editor.NONE, provenance="framing")
