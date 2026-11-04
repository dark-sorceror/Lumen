"""The hero's app drawing: its numbers add up, its reply is honest, and its
layers close one at a time."""
from __future__ import annotations

from pathlib import Path

from tests.node_ts import needs_node, run_ts

CHAT = "components/landing/appChat.ts"
LAYERS = "components/landing/layers.ts"
PLAY = "components/landing/appPlay.ts"
SRC = Path(__file__).resolve().parent.parent / "frontend" / "src"


@needs_node
def test_what_the_newest_turn_added_adds_up_to_its_sixteen_point_seven_k():
    rows = run_ts(CHAT, "return m.TURN_ADDED;")
    assert [(r["label"], r["k"]) for r in rows] == [
        ("Ran the test twice", 8.7), ("Read clock.ts", 3.4), ("Read upload.test.ts again", 2.8),
        ("Its reply", 1.8), ("Your message and what you put back", 0.022)]
    assert round(sum(r["k"] for r in rows), 1) == 16.7
    # Your message is 22 tokens: under 0.1k it shows as the count itself.
    assert run_ts(CHAT, "return m.TURN_ADDED.map((r) => m.kText(r.k));") == [
        "8.7k", "3.4k", "2.8k", "1.8k", "22"]
    assert run_ts(PLAY, "return m.TURNS[m.TURNS.length - 1].delta;") == 16.7


@needs_node
def test_the_two_gone_messages_are_the_ones_the_chat_shows_faded():
    gone = run_ts(CHAT, "return m.GONE.map((g) => g.text);")
    assert gone == ["Don't touch the retry logic. The team wants it kept as it is.",
                    "Run the full suite ten times before you call it fixed."]


@needs_node
def test_gone_says_one_thing_is_and_two_things_are():
    assert run_ts(CHAT, "return [m.goneWords(1), m.goneWords(2)];") == [
        "1 thing you typed is gone", "2 things you typed are gone"]


@needs_node
def test_the_composer_takes_a_click_like_the_apps_but_types_and_sends_nothing():
    got = run_ts(CHAT, "return [typeof m.COMPOSER_HINT, typeof m.SEND_OFF, typeof m.replyTo];")
    assert got == ["undefined", "undefined", "undefined"]
    drawing = (SRC / "components" / "landing" / "AppDrawing.tsx").read_text()
    start = drawing.index("<textarea")
    field = drawing[start:drawing.index("/>", start)]
    # Its placeholder is the app's; it focuses on a click and takes no letters.
    assert 'placeholder="Message Claude Code"' in field and "readOnly" in field
    assert "onClick" not in field and "onKeyDown" not in field
    assert "this drawing" not in drawing.lower()


@needs_node
def test_esc_closes_the_top_layer_only_and_names_its_opener():
    got = run_ts(LAYERS, """
      let s = m.openLayer([], "island", "ring");
      s = m.openLayer(s, "aside", "details");
      const top = m.topLayer(s);
      s = m.closeLayer(s, top.id);
      return [top, m.topLayer(s), m.openLayer(s, "island", "ring").length];""")
    assert got == [{"id": "aside", "opener": "details"}, {"id": "island", "opener": "ring"}, 1]


ORDER = ["pre", "summarising", "summarised", "typingA", "sentA", "workingA", "replyingA", "repliedA",
         "revealing", "puttingBack", "typingB", "sentB", "workingB", "replyingB", "done"]


@needs_node
def test_the_play_runs_the_summary_the_mistake_the_put_back_and_the_fix_in_order():
    assert run_ts(PLAY, "return m.APP_TIMELINE.map((s) => s.phase);") == ORDER
    beats = run_ts(PLAY, "return m.appScript();")
    assert [b["phase"] for b in beats if "phase" in b] == ORDER
    ats = [b["at"] for b in beats]
    assert ats == sorted(ats) and ats[0] == 0


@needs_node
def test_the_ring_goes_86_10_13_21_and_one_thing_comes_back_with_undo_that():
    got = run_ts(PLAY, """
      const seen = [];
      for (const { phase } of m.APP_TIMELINE) {
        const r = m.ringFor(phase), last = seen[seen.length - 1];
        if (!last || last.pct !== r.pct || last.gone !== r.gone) seen.push(r);
      }
      return [seen, ["summarised", "repliedA", "typingB", "sentB", "done"].map(m.ringFor)];""")
    assert got[0] == [{"pct": 86, "gone": 0}, {"pct": 10, "gone": 2}, {"pct": 13, "gone": 2},
                      {"pct": 13, "gone": 1}, {"pct": 21, "gone": 1}]
    assert got[1] == [{"pct": 10, "gone": 2}, {"pct": 13, "gone": 2}, {"pct": 13, "gone": 2},
                      {"pct": 13, "gone": 1}, {"pct": 21, "gone": 1}]


@needs_node
def test_the_two_turns_add_up_to_its_work_and_fill_the_ring_as_it_says():
    turns = run_ts(PLAY, "return m.TURNS.map((t) => [t.delta, t.pct]);")
    final = run_ts(CHAT, "return m.FINAL;")
    assert [d for d, _ in turns] == [6.1, 16.7]
    assert round(sum(d for d, _ in turns), 1) == 22.8  # "Its work", in the island and the panel
    used = 19.8  # after the summary
    for delta, pct in turns:
        used = round(used + delta, 1)
        assert pct == round(used / 200 * 100)
    assert (used, [p for _, p in turns]) == (42.6, [13, 21])
    assert (final["to"], final["used"], final["pct"]) == ("19.8k", "42.6k", 21)


@needs_node
def test_it_types_both_messages_and_the_turns_say_what_the_spec_says():
    assert run_ts(PLAY, "return [m.TYPED, m.TYPED_REPLY];") == ["Now make the fix.", "Undo that."]
    a, b = run_ts(PLAY, "return m.TURNS;")
    assert (a["typed"], a["putBack"], b["typed"], b["putBack"]) == (
        "Now make the fix.", None, "Undo that.", "retry")
    assert a["steps"] == ["Reading src/retry.ts", "Editing src/retry.ts", "Running the upload test"]
    assert a["work"] == "Worked for 31s · read 1 file, edited 1, ran the test once"
    assert a["reply"] == ("Done. I cut the retry delay in src/retry.ts from 200 ms to 50 ms, so the "
                          "retry lands before the assertion. The upload test passes.")
    assert a["drew"] == [{"file": "retry.ts"}]
    assert b["steps"] == ["Reverting src/retry.ts", "Reading test/helpers/clock.ts",
                          "Editing test/upload.test.ts", "Running the test twice"]
    assert b["work"] == "Worked for 48s · reverted 1 file, read 2, edited 1, ran the test twice"
    assert b["reply"] == ("Sorry, I've put src/retry.ts back as it was. The fix is in the test instead: "
                          "upload.test.ts now moves a fake clock past the retry delay, so it no longer "
                          "depends on how fast the machine is.")
    assert b["drew"] == [{"file": "upload.test.ts", "again": True}, {"file": "clock.ts"}]


@needs_node
def test_it_keeps_the_rhythm_and_gives_illumen_its_beats():
    beats = run_ts(PLAY, "return m.appScript();")
    at = {b["phase"]: b["at"] for b in beats if "phase" in b}
    turns = run_ts(PLAY, "return m.TURNS.map((t) => [t.typed.length, t.steps.length, m.replyWords(t.reply)]);")

    def gaps(key: str, start: str, end: str) -> list[int]:
        """The time between one beat that sets `key` and the next, from `start` to `end`."""
        times = [b["at"] for b in beats if key in b and at[start] <= b["at"] < at[end]]
        return [later - sooner for sooner, later in zip(times, times[1:])]

    for (letters, steps, words), (typing, sent, working, replying, done) in zip(turns, [
            ("typingA", "sentA", "workingA", "replyingA", "repliedA"),
            ("typingB", "sentB", "workingB", "replyingB", "done")]):
        assert gaps("typed", typing, sent) == [55] * (letters - 1)  # 55 ms a letter
        assert at[sent] - at[typing] == 55 * letters + 350
        assert gaps("step", working, replying) == [900] * (steps - 1)  # 900 ms a step
        assert at[replying] - at[working] == 900 * steps
        assert gaps("words", replying, done) == [60] * (words - 1)  # 60 ms a word
    # A beat after the mistake, illumen opens the seam and marks the first
    # gone message for 1.2 s, then its Put back is pressed for 180 ms.
    assert at["revealing"] - at["repliedA"] == 900
    assert [b for b in beats if b.get("reveal")] == [{"at": at["revealing"], "phase": "revealing", "reveal": True}]
    assert at["puttingBack"] - at["revealing"] == 1200
    presses = [(b["at"], b["press"]) for b in beats if "press" in b]
    assert presses == [(at["sentA"], "send"), (at["sentA"] + 180, None),
                       (at["puttingBack"], "putBack"), (at["puttingBack"] + 180, None),
                       (at["sentB"], "send"), (at["sentB"] + 180, None)]


@needs_node
def test_each_turn_shows_from_its_message_on_and_the_draft_is_the_one_being_typed():
    phases = ["typingA", "sentA", "workingA", "replyingA", "repliedA", "typingB", "sentB",
              "workingB", "replyingB", "done"]
    got = run_ts(PLAY, "return input.map((p) => [m.turnStage(p, 0), m.turnStage(p, 1)]);", phases)
    assert got == [["hidden", "hidden"], ["sent", "hidden"], ["working", "hidden"], ["replying", "hidden"],
                   ["done", "hidden"], ["done", "hidden"], ["done", "sent"], ["done", "working"],
                   ["done", "replying"], ["done", "done"]]
    drafts = run_ts(PLAY, 'return [m.draftAt("typingA", 4), m.draftAt("typingB", 4), m.draftAt("sentB", 4)];')
    assert drafts == ["Now ", "Undo", ""]


@needs_node
def test_the_first_gone_message_waits_in_the_composer_then_goes_back_in_with_undo_that():
    got = run_ts(PLAY, 'return ["repliedA", "revealing", "puttingBack", "typingB", "sentB", "done"]'
                       '.map((p) => [m.playedBack(p, "retry"), m.playedBack(p, "clock")]);')
    assert got == [["gone", "gone"], ["gone", "gone"], ["queued", "gone"], ["queued", "gone"],
                   ["sent", "gone"], ["sent", "gone"]]
    assert run_ts(PLAY, "return m.PUT_BACK;") == "retry"


@needs_node
def test_file_names_and_the_test_command_are_set_apart_as_code_and_words_stay_whole():
    got = run_ts(PLAY, 'return m.spans("Ran npm test -- upload twice, read src/retry.ts and upload.test.ts again");')
    assert got == [{"text": "Ran ", "code": False}, {"text": "npm test -- upload", "code": True},
                   {"text": " twice, read ", "code": False}, {"text": "src/retry.ts", "code": True},
                   {"text": " and ", "code": False}, {"text": "upload.test.ts", "code": True},
                   {"text": " again", "code": False}]
    # Code is lit word by word with the rest, so a reply counts the same either way.
    for words, reply in run_ts(PLAY, "return m.TURNS.map((t) => [m.replyWords(t.reply), t.reply]);"):
        assert words == len(reply.split())


@needs_node
def test_a_phase_has_reached_the_ones_before_it_and_not_the_ones_after():
    got = run_ts(PLAY, 'return [m.reached("typingA", "summarised"), m.reached("summarising", "summarised"), '
                       'm.reached("sentB", "revealing"), m.reached("repliedA", "puttingBack"), m.reached("done", "done")];')
    assert got == [True, False, True, False, True]


def test_the_overview_counts_what_is_gone_now_not_what_the_summary_dropped():
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "frontend/src/components/landing/AppAside.tsx").read_text()
    assert "{forgotten.length} gone" in src
    assert "2 gone" not in src
