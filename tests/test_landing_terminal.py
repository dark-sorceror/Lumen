"""The terminal drawing's session: what the status line says at each moment,
which commands match, and what the prompt answers."""
from __future__ import annotations

from tests.node_ts import needs_node, run_ts

T = "components/landing/terminal.ts"


@needs_node
def test_the_status_line_follows_the_session():
    got = run_ts(T, """
      const at = (n, o) => m.statusText(m.sessionAt(n), Object.assign({ forgotSeen: false, putBack: 0 }, o || {}));
      const end = m.SCRIPT.length;
      return [at(0), at(3), at(end), at(end, { forgotSeen: true })];""")
    assert got == ["⚠ illumen: 83% full", "⚠ illumen: 2 gone · /illumen-forgot",
                   "⚠ illumen: 17% full", "⚠ illumen: 17% full"]


@needs_node
def test_the_menu_shows_after_a_slash_and_narrows_as_you_type():
    got = run_ts(T, """return [m.matches("/"), m.matches("/il"), m.matches("/illumen-f"), m.matches("fix"), m.matches("/illumen x")]
      .map((l) => l.map((c) => c.name));""")
    assert got == [["/illumen", "/illumen-forgot"], ["/illumen", "/illumen-forgot"], ["/illumen-forgot"], [], []]


@needs_node
def test_forgot_is_the_plainer_output_exactly():
    lines = run_ts(T, "return m.forgotLines();")
    assert lines == [
        "The summary at 11:24 left out 2 things you told Claude:",
        "",
        "1  Never run the migrations against the shared staging database. Use the local docker one.",
        "2  Keep the v2 response shape exactly as it is. The mobile app still pins v2.",
        "",
        "Claude can't see these any more. They weren't found word for word in the summary.",
        "/illumen-forgot restore 1    puts the first one back in your prompt",
    ]
    put = run_ts(T, "return m.forgotLines([0]);")
    assert put[2].endswith("Use the local docker one.  (in your prompt)")
    assert not any("Of 7 messages" in x or "165k" in x for x in lines)


@needs_node
def test_the_script_runs_compact_then_the_request_the_ask_forgot_restore_and_local_migration():
    got = run_ts(T, """const out = [];
      m.SCRIPT.forEach((st, i) => {
        if (st.type !== undefined) out.push("type:" + st.type);
        if (st.send !== undefined) out.push("send:" + st.send);
        if (st.line) out.push(st.line.kind + ":" + (st.line.text || st.line.command || ""));
        if (st.choose !== undefined) out.push("choose:" + st.choose);
      });
      return out;""")
    assert got == [
        "type:/compact",
        "busy:Compacting conversation…",
        "done:",
        "type:run the migrations so I can test the new column",
        "tool:Bash(npm run migrate -- --env staging)",
        "ask:npm run migrate -- --env staging",
        "choose:2",
        "choose:no",
        "type:/illumen-forgot",
        "forgot:",
        "type:/illumen-forgot restore 1",
        "text:Put it back in your prompt. Send it when you're ready.",
        "send:Never run the migrations against the shared staging database. Use the local docker one.",
        "reply:Understood. I'll use the local docker database.",
        "tool:Bash(npm run migrate -- --env local)",
        "reply:Migrations applied to the local docker database. The new column is there.",
        "worked:Worked for 14s",
    ]


@needs_node
def test_the_permission_prompt_moves_to_2_then_collapses_and_the_local_run_applies_3():
    got = run_ts(T, """const asks = (n) => m.sessionAt(n).lines.filter((l) => l.line.kind === "ask").map((l) => l.line.choice);
      const i = m.SCRIPT.findIndex((st) => st.line && st.line.kind === "ask");
      const local = m.sessionAt(m.SCRIPT.length).lines.find((l) => l.line.kind === "tool" && l.line.text.includes("local"));
      return [asks(i + 1), asks(i + 2), asks(i + 3), local.line.result, m.SCRIPT[i + 1].wait];""")
    assert got == [[1], [2], ["no"], "Applied 3 migrations", 1200]


@needs_node
def test_viewing_forgot_in_the_script_drops_2_gone_and_the_session_ends_17_percent_full():
    got = run_ts(T, """const f = m.SCRIPT.findIndex((st) => st.line && st.line.kind === "forgot");
      const at = (n) => m.statusText(m.sessionAt(n), { forgotSeen: false, putBack: 0 });
      return [at(f), at(f + 1), at(m.SCRIPT.length), m.END_PCT];""")
    assert "2 gone" in got[0] and "2 gone" not in got[1]
    assert got[2] == "⚠ illumen: 17% full"
    assert got[3] == 17


@needs_node
def test_restore_with_a_number_that_is_not_there_or_extra_words_uses_the_usage_line():
    got = run_ts(T, """return ["/illumen-forgot restore 9", "/illumen-forgot restore x", "/illumen-forgot now"]
      .map((a) => m.answer(a, { putBack: [] }).text);""")
    assert got == ["Usage: /illumen-forgot, or /illumen-forgot restore [n…]"] * 3


@needs_node
def test_restore_puts_messages_back_in_the_prompt_and_unknowns_are_named():
    got = run_ts(T, """return [m.answer("/illumen-forgot restore 1", { putBack: [] }),
      m.answer("/nope", { putBack: [] }).text, m.answer("hello", { putBack: [] }).text];""")
    assert got[0]["fill"] == ["Never run the migrations against the shared staging database. Use the local docker one."]
    assert got[1] == "Unknown slash command: /nope. This drawing answers /illumen and /illumen-forgot."
    assert got[2] == "This is a drawing, so no model answers here. Type / for the plugin's commands."


@needs_node
def test_enter_runs_the_first_command_a_part_of_a_name_starts():
    got = run_ts(T, """return ["/illumen-f", "/il", "/", "/illumen-f restore 2", "/illumen", "/nope", "hello"].map((a) => m.resolve(a));""")
    assert got == ["/illumen-forgot", "/illumen", "/illumen", "/illumen-forgot restore 2", "/illumen", "/nope", "hello"]


@needs_node
def test_illumen_opens_the_pane_and_closes_it_again():
    got = run_ts(T, """return [m.answer("/illumen", { putBack: [], paneOpen: false }),
      m.answer("/illumen", { putBack: [], paneOpen: true })].map((a) => [a.pane, a.text]);""")
    assert got == [["open", "Opened the illumen pane. /illumen or Esc closes it."],
                   ["close", "Closed the illumen pane. /illumen opens it again."]]


@needs_node
def test_the_pane_says_how_full_and_where_it_stood_before_the_summary():
    got = run_ts(T, """const s = m.sessionAt(m.SCRIPT.length), bar = m.fullBar(s);
      return [m.paneFull(s), m.paneFull(m.sessionAt(3)).big, bar.join("").length, bar.join("").includes("┊"),
        m.PEAK.note, m.roomParts(s).map((p) => p.label), m.roomParts(s).reduce((n, p) => n + p.cells, 0)];""")
    assert got == [{"big": "17% full", "small": "34k of 200k"}, "Summarised", 34, True,
                   "83% before the summary · ┊ where it summarises",
                   ["Instructions", "Tools", "Added", "Summary", "Its work"], 34]


@needs_node
def test_tab_completes_the_highlighted_command_and_the_highlight_wraps():
    got = run_ts(T, """return [m.complete("/il", 0), m.complete("/il", 1), m.complete("/illumen-f", 0),
      m.complete("/il", 9), m.complete("/nope", 0), m.complete("hi", 0), m.complete("/illumen x", 0),
      [m.moveHighlight(0, 2, 1), m.moveHighlight(1, 2, 1), m.moveHighlight(0, 2, -1), m.moveHighlight(1, 2, -1), m.moveHighlight(0, 0, 1)],
      [m.resolve("/il", 1), m.resolve("/il", 0), m.resolve("/illumen-f restore 2", 1), m.resolve("/il x", 1)]];""")
    assert got == ["/illumen", "/illumen-forgot", "/illumen-forgot", "/illumen", None, None, None,
                   [1, 0, 1, 0, 0],
                   ["/illumen-forgot", "/illumen", "/illumen-forgot restore 2", "/illumen x"]]


@needs_node
def test_tab_has_nothing_to_complete_on_a_full_command():
    got = run_ts(T, """return [m.complete("/illumen", 0), m.complete("/illumen-forgot", 0), m.complete("/illumen", 1)];""")
    assert got == [None, None, "/illumen-forgot"]


@needs_node
def test_the_script_puts_item_one_back_and_every_later_view_says_so():
    got = run_ts(T, """const f = m.SCRIPT.findIndex((st) => st.send === m.LEFT_OUT[0]);
      return [m.sessionAt(f).restored, m.sessionAt(f + 1).restored, m.sessionAt(m.SCRIPT.length).restored,
        m.forgotLines([], [0]), m.forgotLines([1], [0]), m.forgotLines([], [0, 1]),
        m.forgotSaid([], [0])];""")
    assert got[0] == [] and got[1] == [0] and got[2] == [0]
    lines = got[3]
    assert lines[2].endswith("Use the local docker one. (back in what it reads)")
    assert lines[3].endswith("still pins v2.") and "(" not in lines[3]
    assert lines[-2] == "Claude still can't see the second one. It wasn't found word for word in the summary."
    assert lines[-1] == "/illumen-forgot restore 2    puts the second one back in your prompt"
    assert "restore 1" not in "\n".join(lines)
    assert got[4][3].endswith("(in your prompt)")
    assert got[5][-1] == "Claude reads both again."
    assert got[6].startswith("The summary at 11:24 left out 2 things you told Claude: 1 Never run")
    assert "(back in what it reads)" in got[6] and "restore 2 puts the second one" in got[6]


@needs_node
def test_the_in_play_forgot_output_is_still_the_specs_before_the_restore():
    got = run_ts(T, "return [m.forgotLines(), m.restoreHint([]), m.restoreHint([0, 1])];")
    assert got[0][-1] == "/illumen-forgot restore 1    puts the first one back in your prompt"
    assert got[1] == got[0][-1] and got[2] is None


def test_the_log_pane_and_seam_read_the_restored_items():
    from pathlib import Path
    src = Path(__file__).resolve().parent.parent / "frontend/src/components/landing"
    log = (src / "TerminalLog.tsx").read_text()
    pane = (src / "TerminalPane.tsx").read_text()
    assert "restored={s.restored}" in (src / "TerminalDrawing.tsx").read_text()
    assert "{sent ? (" in log and "restored={gone.restored}" in log
    # Back in what it reads is ink, not faded, in the receipt and the seam alike.
    assert "restored.includes(i) ? styles.tInk : styles.tFaded" in log
    assert "forgotCaveat(restored)" in log and "{BACK_WORDS}" in log
    assert "put back" in pane and "s.restored.length" in pane


def test_shift_tab_always_moves_focus_back_and_never_completes():
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "frontend/src/components/landing/TerminalDrawing.tsx").read_text()
    assert 'if (e.key === "Tab" && !e.shiftKey) {' in src


def test_the_menu_is_a_combobox_over_a_listbox_of_options():
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "frontend/src/components/landing/TerminalDrawing.tsx").read_text()
    for needle in ('role="combobox"', "aria-expanded={menu}", 'aria-autocomplete="list"',
                   'aria-controls="t-cmds"', 'role="listbox"', 'role="option"', "aria-selected={i === litAt}"):
        assert needle in src, needle
