import pytest

from workbench.context.manager import CacheImpact
from workbench.context.model import ContextObject, EditEvent, Editor, Segment, SegmentKind
from workbench.engine.engine import TokenEvent
from workbench.server.protocol import (cache_impact_msg, context_msg, done_msg,
                                       edit_rejected_msg, parse_client_msg, token_msg,
                                       )


def test_token_msg_from_event():
    e = TokenEvent(token_id=5, text="hi", top_logprobs={5: -0.1, 7: -2.3})
    msg = token_msg(e)
    assert msg == {"type": "token", "token_id": 5, "text": "hi",
                   "top_logprobs": {"5": -0.1, "7": -2.3}}


def test_done_msg():
    assert done_msg("stop") == {"type": "done", "finish_reason": "stop"}


# -- v1.2: tool calling -------------------------------------------------


def test_token_msg_without_override_uses_event_text():
    e = TokenEvent(token_id=9, text="hello")
    assert token_msg(e)["text"] == "hello"


def test_parse_client_msg_valid():
    assert parse_client_msg('{"type": "pause"}') == {"type": "pause"}
    m = parse_client_msg('{"type": "user_message", "text": "hey"}')
    assert m == {"type": "user_message", "text": "hey"}


def test_parse_client_msg_invalid():
    import pytest
    with pytest.raises(ValueError):
        parse_client_msg('{"type": "rm -rf"}')
    with pytest.raises(ValueError):
        parse_client_msg("not json")
    with pytest.raises(ValueError):
        parse_client_msg('{"type": "user_message"}')  # missing text


# -- Attachments: optional attachment_ids on user_message -----------


# -- v1.1: get_context / preview_edit / apply_edit ---------------------------


def test_parse_get_context():
    assert parse_client_msg('{"type": "get_context"}') == {"type": "get_context"}


_VALID_EVENT = {"op": "replace_text", "segment_id": "s1",
                "payload": {"text": "hi"}, "actor": "user"}


@pytest.mark.parametrize("msg_type", ["preview_edit", "apply_edit"])
def test_parse_edit_msg_valid(msg_type):
    import json
    raw = json.dumps({"type": msg_type, "event": _VALID_EVENT})
    msg = parse_client_msg(raw)
    assert msg == {"type": msg_type, "event": _VALID_EVENT}


@pytest.mark.parametrize("msg_type", ["preview_edit", "apply_edit"])
def test_parse_edit_msg_missing_event_raises(msg_type):
    import json
    with pytest.raises(ValueError):
        parse_client_msg(json.dumps({"type": msg_type}))


@pytest.mark.parametrize("msg_type", ["preview_edit", "apply_edit"])
def test_parse_edit_msg_malformed_event_raises(msg_type):
    import json
    bad_events = [
        "not a dict",
        {"segment_id": "s1", "payload": {}, "actor": "user"},  # missing op
        {"op": "move", "payload": {}, "actor": "user"},         # missing segment_id
        {"op": "move", "segment_id": "s1", "actor": "user"},    # missing payload
        {"op": "move", "segment_id": "s1", "payload": "nope", "actor": "user"},  # payload not a dict
        {"op": "move", "segment_id": "s1", "payload": {}, "actor": 5},  # actor not a string
    ]
    for event in bad_events:
        with pytest.raises(ValueError):
            parse_client_msg(json.dumps({"type": msg_type, "event": event}))


@pytest.mark.parametrize("msg_type", ["preview_edit", "apply_edit"])
def test_parse_edit_msg_extra_key_in_event_raises(msg_type):
    """An event dict with an unexpected extra key must be
    rejected up front, not silently accepted and later crash EditEvent(**...)
    with a TypeError."""
    import json
    event = dict(_VALID_EVENT, extra_field="sneaky")
    with pytest.raises(ValueError):
        parse_client_msg(json.dumps({"type": msg_type, "event": event}))


def test_context_msg_from_context_object():
    ctx = ContextObject()
    s = Segment(id="s1", kind=SegmentKind.USER_MSG, text="hi", emphasis=0.5,
               editable_by=Editor.BOTH, provenance="user")
    ctx.apply(EditEvent(op="append", segment_id=s.id,
                        payload={"segment": s.__dict__}, actor="user"))
    msg = context_msg(ctx)
    assert msg == {
        "type": "context",
        "segments": [
            {"id": "s1", "kind": "user_msg", "text": "hi", "emphasis": 0.5,
             "editable_by": "both", "provenance": "user"},
        ],
    }


def test_cache_impact_msg():
    impact = CacheImpact(first_invalid_token=3, tokens_to_reprefill=7)
    assert cache_impact_msg(impact, preview=True) == {
        "type": "cache_impact", "first_invalid_token": 3,
        "tokens_to_reprefill": 7, "preview": True,
    }
    assert cache_impact_msg(impact, preview=False)["preview"] is False


def test_edit_rejected_msg():
    assert edit_rejected_msg("pause or wait first") == {
        "type": "edit_rejected", "message": "pause or wait first",
    }
