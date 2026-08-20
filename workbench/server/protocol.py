"""v1(.1) WS wire protocol. New message types may be added; existing fields are frozen."""
from __future__ import annotations

import json

from workbench.context.manager import CacheImpact
from workbench.context.model import ContextObject
from workbench.engine.engine import TokenEvent

CLIENT_TYPES = {"user_message", "pause", "resume", "abort",
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
    if msg["type"] == "user_message" and "top_k_logprobs" in msg:
        # Opt-in per turn: the token event has always carried `top_logprobs`,
        # but generation ran with top_k_logprobs=0, so the field was always
        # empty. Off by default -- it costs a sort over the vocabulary per
        # token.
        k = msg["top_k_logprobs"]
        if not isinstance(k, int) or isinstance(k, bool) or k < 0:
            raise ValueError("user_message.top_k_logprobs must be a non-negative int")
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


def token_msg(e: TokenEvent) -> dict:
    return {
        "type": "token",
        "token_id": e.token_id,
        "text": e.text,
        "top_logprobs": {str(k): v for k, v in e.top_logprobs.items()},
    }


def done_msg(finish_reason: str) -> dict:
    """finish_reason is one of "stop"/"length"/"aborted" (pre-existing) or,
    additively (v1.2), "tool_limit" -- emitted when the agentic tool loop in
    app.py hits MAX_TOOL_ROUNDS while the model still wants to call a tool."""
    return {"type": "done", "finish_reason": finish_reason}


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
