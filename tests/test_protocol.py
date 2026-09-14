import json
from dataclasses import asdict
import pytest

from workbench.context.manager import CacheImpact
from workbench.context.model import ContextObject, EditEvent, Editor, Segment, SegmentKind
from workbench.engine.engine import TokenEvent
from workbench.config import SteeringSpec, TaskConfig
from workbench.server.protocol import (
    config_msg,
    inspection_msg,cache_impact_msg, context_msg, done_msg,
                                       edit_rejected_msg, parse_client_msg, token_msg,
                                       tool_call_msg, tool_result_msg)


def test_token_msg_from_event():
    e = TokenEvent(token_id=5, text="hi", top_logprobs={5: -0.1, 7: -2.3})
    msg = token_msg(e)
    assert msg == {"type": "token", "token_id": 5, "text": "hi",
                   "top_logprobs": {"5": -0.1, "7": -2.3}}


def test_done_msg():
    assert done_msg("stop") == {"type": "done", "finish_reason": "stop"}


def test_done_msg_tool_limit_finish_reason():
    assert done_msg("tool_limit") == {"type": "done", "finish_reason": "tool_limit"}


# -- v1.2: tool calling -------------------------------------------------

def test_token_msg_text_override_suppresses_raw_text():
    e = TokenEvent(token_id=9, text="<tool_call>{...}</tool_call>",
                   top_logprobs={1: -0.5})
    msg = token_msg(e, text="")
    assert msg == {"type": "token", "token_id": 9, "text": "",
                   "top_logprobs": {"1": -0.5}}


def test_token_msg_without_override_uses_event_text():
    e = TokenEvent(token_id=9, text="hello")
    assert token_msg(e)["text"] == "hello"


def test_tool_call_msg():
    msg = tool_call_msg("tc_0", "calculator", {"expression": "2+2"})
    assert msg == {"type": "tool_call", "call_id": "tc_0", "name": "calculator",
                   "arguments": {"expression": "2+2"}}


def test_tool_result_msg_success():
    msg = tool_result_msg("tc_0", "calculator", "4", error=False)
    assert msg == {"type": "tool_result", "call_id": "tc_0", "name": "calculator",
                   "result": "4", "error": False}


def test_tool_result_msg_error():
    msg = tool_result_msg("tc_0", "calculator", "error: bad expression", error=True)
    assert msg["error"] is True
    assert msg["result"] == "error: bad expression"


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


def test_parse_user_message_with_attachment_ids():
    import json
    raw = json.dumps({"type": "user_message", "text": "hey",
                      "attachment_ids": ["abc123", "def456"]})
    msg = parse_client_msg(raw)
    assert msg == {"type": "user_message", "text": "hey",
                   "attachment_ids": ["abc123", "def456"]}


def test_parse_user_message_with_empty_attachment_ids():
    assert parse_client_msg(
        '{"type": "user_message", "text": "hi", "attachment_ids": []}'
    ) == {"type": "user_message", "text": "hi", "attachment_ids": []}


def test_parse_user_message_without_attachment_ids_still_valid():
    # Existing wire clients that never send attachment_ids must stay valid.
    assert parse_client_msg('{"type": "user_message", "text": "hey"}') == {
        "type": "user_message", "text": "hey"}


def test_parse_user_message_non_list_attachment_ids_rejected():
    import json
    import pytest
    for bad in ("abc123", 5, {"a": 1}, None):
        with pytest.raises(ValueError):
            parse_client_msg(json.dumps(
                {"type": "user_message", "text": "hey", "attachment_ids": bad}))


def test_parse_user_message_non_string_items_in_attachment_ids_rejected():
    import json
    import pytest
    with pytest.raises(ValueError):
        parse_client_msg(json.dumps(
            {"type": "user_message", "text": "hey", "attachment_ids": ["ok", 5]}))


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


# -- Inspection: on-demand measurement of the live context -------------------

def test_parse_inspect_message_with_layers():
    msg = parse_client_msg('{"type": "inspect", "layers": [0, 5]}')
    assert msg["type"] == "inspect"
    assert msg["layers"] == [0, 5]


def test_parse_inspect_defaults_to_no_explicit_layers():
    assert "layers" not in parse_client_msg('{"type": "inspect"}')


def test_parse_inspect_rejects_non_integer_layers():
    with pytest.raises(ValueError):
        parse_client_msg('{"type": "inspect", "layers": ["oops"]}')


def test_inspection_msg_carries_lens_and_attention_per_layer():
    m = inspection_msg([
        {"layer": 0,
         "lens": [{"token_id": 7, "text": " Paris", "logprob": -0.2}],
         "attention_mass": [{"segment_id": "s1", "mass": 0.81}]},
    ])
    assert m["type"] == "inspection"
    assert m["layers"][0]["layer"] == 0
    assert m["layers"][0]["lens"][0]["text"] == " Paris"
    assert m["layers"][0]["attention_mass"][0]["mass"] == 0.81


def test_context_msg_carries_per_segment_edit_cost_when_available():
    ctx = ContextObject()
    seg = Segment(id="s0", kind=SegmentKind.USER_MSG, text="hi", provenance="user")
    ctx.apply(EditEvent(op="append", segment_id="s0",
                        payload={"segment": asdict(seg)}, actor="user"))

    m = context_msg(ctx, costs={"s0": CacheImpact(first_invalid_token=3,
                                                  tokens_to_reprefill=9)})

    assert m["segments"][0]["edit_cost"] == {"first_invalid_token": 3,
                                             "tokens_to_reprefill": 9}


def test_context_msg_omits_edit_cost_when_not_supplied():
    ctx = ContextObject()
    seg = Segment(id="s0", kind=SegmentKind.USER_MSG, text="hi", provenance="user")
    ctx.apply(EditEvent(op="append", segment_id="s0",
                        payload={"segment": asdict(seg)}, actor="user"))
    assert "edit_cost" not in context_msg(ctx)["segments"][0]


# -- Rewind: undo, over the wire --------------------------------------------

def test_parse_rewind_with_explicit_target():
    msg = parse_client_msg('{"type": "rewind", "to_event": 3}')
    assert msg["to_event"] == 3


def test_parse_rewind_without_target_means_undo_one():
    assert "to_event" not in parse_client_msg('{"type": "rewind"}')


def test_parse_rewind_rejects_a_non_integer_target():
    with pytest.raises(ValueError):
        parse_client_msg('{"type": "rewind", "to_event": "two"}')


def test_parse_rewind_rejects_a_negative_target():
    with pytest.raises(ValueError):
        parse_client_msg('{"type": "rewind", "to_event": -1}')


# -- Steering configuration over the wire ------------------------------------

def test_parse_derive_steering():
    msg = parse_client_msg(json.dumps({
        "type": "derive_steering", "positive": ["a"], "negative": ["b"],
        "layer": 18, "strength": 0.15}))
    assert msg["layer"] == 18


def test_parse_derive_steering_requires_both_sides():
    with pytest.raises(ValueError):
        parse_client_msg(json.dumps({"type": "derive_steering",
                                     "positive": ["a"], "negative": [],
                                     "layer": 1, "strength": 0.1}))


def test_parse_derive_steering_rejects_non_string_prompts():
    with pytest.raises(ValueError):
        parse_client_msg(json.dumps({"type": "derive_steering",
                                     "positive": [3], "negative": ["b"],
                                     "layer": 1, "strength": 0.1}))


def test_config_msg_summarises_without_shipping_the_vector():
    """A d_model-length direction is not something to push over a socket on
    every state change; the client needs the shape, not the numbers."""
    cfg = TaskConfig(name="tone", steering=[
        SteeringSpec(layer=18, vector=[0.1] * 8, strength=0.15, label="warm")])

    m = config_msg(cfg)

    assert m["type"] == "config"
    assert m["name"] == "tone"
    entry = m["steering"][0]
    assert entry == {"layer": 18, "strength": 0.15, "label": "warm", "dim": 8}


# -- opt-in per-token logprobs ----------------------------------------------

def test_user_message_may_request_top_logprobs():
    """The wire already carries `top_logprobs` on every token event, but the
    server hardcoded top_k_logprobs=0, so the field was always empty and the
    README's "per-token logprobs on the wire" was not actually reachable by any
    client. Requesting it per turn keeps the default cheap."""
    msg = parse_client_msg(
        '{"type": "user_message", "text": "hi", "top_k_logprobs": 5}')

    assert msg["top_k_logprobs"] == 5


def test_top_logprobs_request_must_be_a_non_negative_int():
    for bad in ('"5"', "-1", "null"):
        with pytest.raises(ValueError):
            parse_client_msg(
                '{"type": "user_message", "text": "hi", "top_k_logprobs": %s}' % bad)


def test_omitting_it_stays_valid_and_unset():
    msg = parse_client_msg('{"type": "user_message", "text": "hi"}')

    assert "top_k_logprobs" not in msg
