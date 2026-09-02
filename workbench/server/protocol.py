"""v1(.1) WS wire protocol. New message types may be added; existing fields are frozen."""
from __future__ import annotations

import json

from workbench.context.manager import CacheImpact
from workbench.context.model import ContextObject
from workbench.engine.engine import TokenEvent

CLIENT_TYPES = {"user_message", "pause", "resume", "abort", "inspect",
                "get_context", "preview_edit", "apply_edit"}

_EVENT_FIELDS = {"op": str, "segment_id": str, "payload": dict, "actor": str}


def _validate_event(event: object) -> None:
    if not isinstance(event, dict):
        raise ValueError("event must be an object")
    if set(event.keys()) != set(_EVENT_FIELDS):
        raise ValueError(
            f"event has unexpected keys: {sorted(set(event.keys()) - set(_EVENT_FIELDS))}")
    for key, typ in _EVENT_FIELDS.items():
        if key not in event or not isinstance(event[key], typ):
            raise ValueError(f"event.{key} must be a {typ.__name__}")


def parse_client_msg(raw: str) -> dict:
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e
    if not isinstance(msg, dict) or msg.get("type") not in CLIENT_TYPES:
        raise ValueError(f"unknown message type: {msg!r}")
    if msg["type"] == "user_message":
        if not isinstance(msg.get("text"), str):
            raise ValueError("user_message requires a string 'text'")
    if msg["type"] == "user_message":
        if "top_k_logprobs" in msg:
            # Opt-in per turn: the token event has always carried
            # `top_logprobs`, but generation ran with top_k_logprobs=0, so the
            # field was always empty. Off by default -- it costs a sort over
            # the vocabulary per token.
            k = msg["top_k_logprobs"]
            if not isinstance(k, int) or isinstance(k, bool) or k < 0:
                raise ValueError(
                    "user_message.top_k_logprobs must be a non-negative int")
        if "attachment_ids" in msg:
            ids = msg["attachment_ids"]
            if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
                raise ValueError(
                    "user_message.attachment_ids must be a list of strings")
    if msg["type"] == "inspect" and "layers" in msg:
        layers = msg["layers"]
        if not isinstance(layers, list) or not all(isinstance(i, int) for i in layers):
            raise ValueError("inspect.layers must be a list of integers")
    if msg["type"] in ("preview_edit", "apply_edit"):
        _validate_event(msg.get("event"))
    return msg


def gen_stats_msg(prompt_tokens: int, cached_tokens: int) -> dict:
    """Sent once, at the start of a turn's generation (before/at the first
    token): the INPUT-side stats the client cannot derive itself (prompt
    length, and how much of it was reused from the KV cache). The client
    counts OUTPUT tokens and derives tokens/sec itself from the `token`
    stream it already receives."""
    return {
        "type": "gen_stats",
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
    }


def token_msg(e: TokenEvent, text: str | None = None) -> dict:
    """`text` overrides `e.text` when given (additive, v1.2): the
    tool-calling round loop in app.py suppresses raw `<tool_call>...
    </tool_call>` tags from the token stream by forwarding a filtered
    substring of the event's text rather than the text itself -- see
    run_generation's tag-suppression state machine. Every existing caller
    (the non-tool-calling path, and every pre-tool-calling test) omits `text` and
    gets exactly the old behavior."""
    return {
        "type": "token",
        "token_id": e.token_id,
        "text": e.text if text is None else text,
        "top_logprobs": {str(k): v for k, v in e.top_logprobs.items()},
    }


def done_msg(finish_reason: str) -> dict:
    """finish_reason is one of "stop"/"length"/"aborted" (pre-existing) or,
    additively (v1.2), "tool_limit" -- emitted when the agentic tool loop in
    app.py hits MAX_TOOL_ROUNDS while the model still wants to call a tool."""
    return {"type": "done", "finish_reason": finish_reason}


def tool_call_msg(call_id: str, name: str, arguments: dict) -> dict:
    """v1.2: emitted once a round's `<tool_call>{...}</tool_call>` block has
    been parsed, before the tool executes -- lets the client render a
    structured "calling tool X" step instead of the raw tags (which are
    suppressed from the `token` stream; see token_msg)."""
    return {"type": "tool_call", "call_id": call_id, "name": name, "arguments": arguments}


def tool_result_msg(call_id: str, name: str, result: str, error: bool) -> dict:
    """v1.2: emitted once the tool named in a prior tool_call_msg has run.
    `error=True` means `result` is a human-readable failure message (unknown
    tool, malformed arguments, or a caught exception/timeout) rather than the
    tool's real return value -- the model still gets it fed back as the next
    round's tool-response context, so it can self-correct."""
    return {"type": "tool_result", "call_id": call_id, "name": name,
            "result": result, "error": error}


def error_msg(message: str) -> dict:
    return {"type": "error", "message": message}


def segment_msg(seg) -> dict:
    return {
        "id": seg.id,
        "kind": seg.kind.value,
        "text": seg.text,
        "emphasis": seg.emphasis,
        "editable_by": seg.editable_by.value,
        "provenance": seg.provenance,
    }


def context_msg(ctx: ContextObject) -> dict:
    return {"type": "context", "segments": [segment_msg(s) for s in ctx.segments]}


def cache_impact_msg(impact: CacheImpact, preview: bool) -> dict:
    return {
        "type": "cache_impact",
        "first_invalid_token": impact.first_invalid_token,
        "tokens_to_reprefill": impact.tokens_to_reprefill,
        "preview": preview,
    }


def edit_rejected_msg(message: str) -> dict:
    return {"type": "edit_rejected", "message": message}


def inspection_msg(layers: list[dict]) -> dict:
    """One-shot measurement of the live context (v1.3).

    Requested rather than streamed: a hidden state is d_model floats per layer
    per token, so pushing activations alongside the token stream is not viable.
    Each entry carries what that depth would predict (`lens`) and where its
    attention actually went, aggregated to context SEGMENTS rather than raw
    token positions -- segments being the unit the client can edit."""
    return {"type": "inspection", "layers": layers}
