"""The terminal client: input parsing and event rendering.

Both halves are pure functions over dicts, so the whole protocol surface is
testable without a server, a socket, or a model. The asyncio plumbing that
carries them is thin by design -- see workbench/cli/session.py."""
import pytest

from workbench.cli.client import QUIT, RenderState, parse_input, render_event


# -- input -> wire message ---------------------------------------------------

def test_plain_text_becomes_a_user_message():
    assert parse_input("hello there") == {"type": "user_message", "text": "hello there"}


def test_a_leading_slash_is_a_command_not_a_message():
    assert parse_input("/context") == {"type": "get_context"}
    assert parse_input("/abort") == {"type": "abort"}
    assert parse_input("/pause") == {"type": "pause"}
    assert parse_input("/resume") == {"type": "resume"}


def test_quit_is_a_sentinel_not_a_wire_message():
    """/quit is the one input that must not be sent to the server."""
    assert parse_input("/quit") is QUIT


def test_edit_builds_a_valid_event_envelope():
    """The server validates `event` keys exactly, so the client must produce
    all four fields and no others (see protocol._validate_event)."""
    msg = parse_input("/edit seg-3 the new text")

    assert msg["type"] == "preview_edit"
    assert set(msg["event"]) == {"op", "segment_id", "payload", "actor"}
    assert msg["event"]["op"] == "replace_text"
    assert msg["event"]["segment_id"] == "seg-3"
    assert msg["event"]["payload"] == {"text": "the new text"}
    assert msg["event"]["actor"] == "user"


def test_apply_reuses_the_pending_edit_so_preview_and_apply_cannot_diverge():
    state = RenderState()
    state.pending_edit = {"op": "replace_text", "segment_id": "seg-3",
                          "payload": {"text": "x"}, "actor": "user"}

    msg = parse_input("/apply", state)

    assert msg == {"type": "apply_edit", "event": state.pending_edit}


def test_apply_without_a_preview_is_refused_locally():
    """Better to say so than to send an apply the server will reject."""
    assert parse_input("/apply", RenderState()) is None


def test_an_unknown_command_is_refused_rather_than_sent_as_chat():
    """Otherwise a typo'd command is silently said to the model."""
    assert parse_input("/nonsense") is None


def test_blank_input_sends_nothing():
    assert parse_input("   ") is None


# -- server event -> terminal output -----------------------------------------

def test_tokens_stream_without_newlines_between_them():
    state = RenderState()

    a = render_event({"type": "token", "token_id": 1, "text": "Hel", "top_logprobs": {}}, state)
    b = render_event({"type": "token", "token_id": 2, "text": "lo", "top_logprobs": {}}, state)

    assert a == "Hel" and b == "lo"


def test_logprobs_are_hidden_by_default_and_shown_when_asked():
    """The white-box view: per-token alternatives are on the wire already, so
    showing them is a client-side switch, not a new request."""
    event = {"type": "token", "token_id": 1, "text": " Paris",
             "top_logprobs": {"1": -0.11, "2": -2.4}}

    assert render_event(event, RenderState()) == " Paris"

    verbose = RenderState(show_logprobs=True)
    out = render_event(event, verbose)
    assert " Paris" in out and "-0.11" in out


def test_gen_stats_reports_cache_reuse():
    out = render_event({"type": "gen_stats", "prompt_tokens": 120,
                        "cached_tokens": 100}, RenderState())

    assert "120" in out and "100" in out


def test_context_lists_every_segment_with_its_id_and_kind():
    event = {"type": "context", "segments": [
        {"id": "s1", "kind": "system", "text": "You are terse.",
         "emphasis": 1.0, "editable_by": "user", "provenance": {}},
        {"id": "s2", "kind": "user", "text": "hi",
         "emphasis": 1.0, "editable_by": "user", "provenance": {}},
    ]}

    out = render_event(event, RenderState())

    assert "s1" in out and "system" in out and "You are terse." in out
    assert "s2" in out and "user" in out


def test_cache_impact_states_the_cost_of_the_edit():
    out = render_event({"type": "cache_impact", "first_invalid_token": 40,
                        "tokens_to_reprefill": 85, "preview": True}, RenderState())

    assert "40" in out and "85" in out


def test_a_preview_records_the_pending_edit_for_apply():
    state = RenderState()
    state.pending_edit = {"op": "replace_text", "segment_id": "s1",
                          "payload": {"text": "x"}, "actor": "user"}

    render_event({"type": "cache_impact", "first_invalid_token": 0,
                  "tokens_to_reprefill": 9, "preview": True}, state)

    assert state.pending_edit is not None      # survives a preview
    render_event({"type": "cache_impact", "first_invalid_token": 0,
                  "tokens_to_reprefill": 9, "preview": False}, state)
    assert state.pending_edit is None          # cleared once applied


def test_errors_and_rejections_are_surfaced():
    assert "boom" in render_event({"type": "error", "message": "boom"}, RenderState())
    assert "nope" in render_event({"type": "edit_rejected", "message": "nope"},
                                  RenderState())


def test_done_reports_the_finish_reason():
    assert "length" in render_event({"type": "done", "finish_reason": "length"},
                                    RenderState())


def test_an_unrecognised_event_does_not_crash_the_client():
    """The server's protocol grows; an old client must not die on a new event."""
    out = render_event({"type": "something_new_in_v2", "payload": 1}, RenderState())

    assert out is None or isinstance(out, str)


# -- batch mode: one turn at a time ------------------------------------------

from workbench.cli.client import completes  # noqa: E402


def test_each_request_knows_which_event_completes_it():
    """Piped input outruns the server otherwise: every line is read and /quit
    fires before the first token arrives. Batch mode waits for the reply, so a
    script is a sequence of turns rather than a race."""
    assert "done" in completes("user_message")
    assert "context" in completes("get_context")
    assert completes("preview_edit") >= {"cache_impact", "edit_rejected"}
    assert completes("apply_edit") >= {"cache_impact", "edit_rejected"}


def test_an_error_completes_any_request():
    """Otherwise a rejected request hangs a script forever."""
    for kind in ("user_message", "get_context", "preview_edit", "apply_edit"):
        assert "error" in completes(kind), kind


def test_fire_and_forget_requests_wait_for_nothing():
    """pause/resume have no reply of their own; waiting on one would hang."""
    assert completes("pause") == set()
    assert completes("resume") == set()


def test_logprobs_mode_asks_the_server_for_them():
    """The client cannot render alternatives the server never sent."""
    assert "top_k_logprobs" not in parse_input("hi", RenderState())
    assert parse_input("hi", RenderState(show_logprobs=True))["top_k_logprobs"] == 5


def test_logprob_rows_start_on_their_own_line():
    """Inline brackets split the token stream mid-word: a real run rendered
    '</t  [ids...]hink>'. In logprobs mode the stream is a table, not prose."""
    state = RenderState(show_logprobs=True)

    out = render_event({"type": "token", "token_id": 9, "text": "lo",
                        "top_logprobs": {"9": -0.1}}, state)

    assert out.startswith("\n")
    assert "'lo'" in out          # quoted, so whitespace tokens are visible


# -- addressing a segment without knowing its uuid ---------------------------

def test_context_is_remembered_so_edits_can_name_a_segment_positionally():
    """A piped script cannot paste a uuid it has not seen yet. The client keeps
    the last /context listing so a segment can be addressed by position or
    kind."""
    state = RenderState()
    render_event({"type": "context", "segments": [
        {"id": "aaa", "kind": "system", "text": "sys", "emphasis": 1.0,
         "editable_by": "both", "provenance": {}},
        {"id": "bbb", "kind": "user_msg", "text": "hi", "emphasis": 1.0,
         "editable_by": "both", "provenance": {}},
    ]}, state)

    assert parse_input("/edit #2 new", state)["event"]["segment_id"] == "bbb"
    assert parse_input("/edit @system new", state)["event"]["segment_id"] == "aaa"
    assert parse_input("/edit @user_msg new", state)["event"]["segment_id"] == "bbb"


def test_a_kind_selector_picks_the_last_segment_of_that_kind():
    """With several user turns, '@user_msg' means the most recent one."""
    state = RenderState()
    render_event({"type": "context", "segments": [
        {"id": "u1", "kind": "user_msg", "text": "first", "emphasis": 1.0,
         "editable_by": "both", "provenance": {}},
        {"id": "u2", "kind": "user_msg", "text": "second", "emphasis": 1.0,
         "editable_by": "both", "provenance": {}},
    ]}, state)

    assert parse_input("/edit @user_msg x", state)["event"]["segment_id"] == "u2"


def test_an_unresolvable_selector_is_refused_locally():
    state = RenderState()
    assert parse_input("/edit #9 x", state) is None
    assert parse_input("/edit @nosuchkind x", state) is None


def test_a_full_id_still_works_without_any_context_listing():
    msg = parse_input("/edit abc123 x", RenderState())

    assert msg["event"]["segment_id"] == "abc123"
