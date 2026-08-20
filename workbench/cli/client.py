"""Input parsing and event rendering for the terminal client.

Kept pure and separate from the socket: everything here is dict-in, string-out,
so the entire protocol surface is testable without a server or a model."""
from dataclasses import dataclass, field

# Sentinel for "/quit" -- distinct from None, which means "nothing to send".
QUIT = object()

_BARE = {"/context": "get_context", "/abort": "abort",
         "/pause": "pause", "/resume": "resume"}

HELP = """commands:
  /context           list the live context segments
  /edit <sel> <text> preview replacing a segment; <sel> is #N, @kind, or an id
  /apply             apply the previewed edit
  /logprobs          toggle per-token alternatives
  /pause /resume /abort
  /help /quit
anything else is sent to the model."""


@dataclass
class RenderState:
    """What the renderer needs to remember between events."""
    show_logprobs: bool = False
    pending_edit: dict | None = None
    # Last /context listing, so an edit can name a segment by position (#2) or
    # kind (@user_msg) instead of a uuid the user has not seen yet.
    segments: list = field(default_factory=list)
    streaming: bool = False
    output_tokens: int = 0
    notes: list[str] = field(default_factory=list)


def parse_input(line: str, state: RenderState | None = None):
    """Turn a line of typed input into a wire message.

    Returns a dict to send, None to send nothing (blank line, unknown command,
    or a local refusal), or QUIT."""
    text = line.strip()
    if not text:
        return None
    if not text.startswith("/"):
        msg = {"type": "user_message", "text": text}
        if state is not None and state.show_logprobs:
            # Asked for per-turn, not globally: the server computes a sort over
            # the vocabulary per token when this is on.
            msg["top_k_logprobs"] = 5
        return msg

    word, _, rest = text.partition(" ")
    if word == "/quit":
        return QUIT
    if word in _BARE:
        return {"type": _BARE[word]}
    if word == "/edit":
        selector, _, new_text = rest.strip().partition(" ")
        if not selector or not new_text.strip():
            return None
        segment_id = resolve_segment(selector, state)
        if segment_id is None:
            return None
        return {"type": "preview_edit",
                "event": {"op": "replace_text", "segment_id": segment_id,
                          "payload": {"text": new_text.strip()}, "actor": "user"}}
    if word == "/apply":
        # Reuses the previewed event verbatim rather than rebuilding it, so the
        # thing priced is necessarily the thing applied.
        if state is None or state.pending_edit is None:
            return None
        return {"type": "apply_edit", "event": state.pending_edit}
    return None


# Which server event marks a request finished. Used by batch mode to send one
# request per turn instead of racing the whole script onto the socket at once.
# "error" is in every set: a refused request must complete the wait, or a piped
# script hangs. pause/resume have no reply of their own, so they wait for none.
_COMPLETES = {
    "user_message": {"done"},
    "get_context": {"context"},
    "preview_edit": {"cache_impact", "edit_rejected"},
    "apply_edit": {"cache_impact", "edit_rejected"},
    "abort": {"done"},
    "pause": set(),
    "resume": set(),
}


def completes(request_type: str) -> set[str]:
    """The events that end a wait for `request_type`."""
    waits = _COMPLETES.get(request_type, set())
    return (waits | {"error"}) if waits else set()


def resolve_segment(selector: str, state: RenderState | None) -> str | None:
    """Turn '#2', '@user_msg' or a raw id into a segment id.

    Positional and kind selectors resolve against the last context listing;
    anything else is passed through untouched, so a full id still works with no
    listing at all."""
    segments = state.segments if state else []
    if selector.startswith("#"):
        try:
            index = int(selector[1:])
        except ValueError:
            return None
        if 1 <= index <= len(segments):
            return segments[index - 1].get("id")
        return None
    if selector.startswith("@"):
        kind = selector[1:]
        matches = [s for s in segments if s.get("kind") == kind]
        return matches[-1].get("id") if matches else None   # last, i.e. newest
    return selector


def _fmt_logprobs(top: dict) -> str:
    pairs = sorted(top.items(), key=lambda kv: -kv[1])[:5]
    return " ".join(f"{tid}:{lp:.2f}" for tid, lp in pairs)


def render_event(event: dict, state: RenderState) -> str | None:
    """Turn one server event into terminal output, or None to print nothing.

    Unknown event types return None rather than raising: the protocol is
    versioned and additive, so an older client meeting a newer server should go
    quiet on what it does not understand, not die."""
    kind = event.get("type")

    if kind == "token":
        state.streaming = True
        state.output_tokens += 1
        text = event.get("text", "")
        if state.show_logprobs and event.get("top_logprobs"):
            # One row per token. Inline brackets split the stream mid-word.
            # repr() so whitespace and newline tokens stay visible.
            return f"\n{text!r:>12}  {_fmt_logprobs(event['top_logprobs'])}"
        return text

    if kind == "gen_stats":
        prompt = event.get("prompt_tokens", 0)
        cached = event.get("cached_tokens", 0)
        reuse = (100.0 * cached / prompt) if prompt else 0.0
        return f"[prompt {prompt} tok, {cached} reused from cache ({reuse:.0f}%)]"

    if kind == "done":
        state.streaming = False
        return f"\n[done: {event.get('finish_reason')}, {state.output_tokens} tokens]"

    if kind == "context":
        segments = event.get("segments", [])
        state.segments = segments
        lines = [f"context: {len(segments)} segments"]
        for n, seg in enumerate(segments, 1):
            text = " ".join((seg.get("text") or "").split())
            if len(text) > 60:
                text = text[:57] + "..."
            # Both the index (for "#2") and a short id, since the id is what
            # the server actually keys on and what an error will quote back.
            sid = str(seg.get("id") or "")
            lines.append(f"  #{n:<3} {sid[:8]:<10} {seg.get('kind'):<14} "
                         f"[{seg.get('editable_by'):<4}] {text}")
        return "\n".join(lines)

    if kind == "cache_impact":
        first = event.get("first_invalid_token")
        refill = event.get("tokens_to_reprefill")
        if event.get("preview"):
            return (f"[preview: cache valid to token {first}; "
                    f"{refill} tokens would be re-prefilled -- /apply to commit]")
        state.pending_edit = None
        return f"[applied: cache valid to token {first}; {refill} tokens re-prefilled]"

    if kind == "edit_rejected":
        state.pending_edit = None
        return f"[edit rejected: {event.get('message')}]"

    if kind == "error":
        return f"[error: {event.get('message')}]"

    return None
