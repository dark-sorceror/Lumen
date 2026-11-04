"""The story keeps its own word-for-word rule: what it calls gone is nowhere in
the summary, and what it calls kept is quoted in it word for word."""
from __future__ import annotations

import re
from pathlib import Path

from tests.node_ts import needs_node, run_ts

S = "components/landing/story.ts"
SRC = Path(__file__).resolve().parent.parent / "frontend" / "src"


@needs_node
def test_the_five_steps_and_how_full_each_is():
    got = run_ts(S, "return m.STEPS.map((s) => [s.t, s.pct]);")
    assert got == [["You start talking.", 6], ["It fills up.", 88], ["It summarises to keep going.", 11],
                   ["Some of what you typed is gone.", 11], ["illumen shows you which.", 11]]


@needs_node
def test_gone_means_nowhere_in_the_summary_and_kept_means_quoted_in_it():
    # The summary quotes a kept message without its closing full stop.
    got = run_ts(S, """return m.ROWS.filter((r) => r.k === "you")
      .map((r) => [r.t, !!r.gone, m.SUMMARY.includes(r.t.replace(/\\.$/, ""))]);""")
    for text, gone, quoted in got:
        assert gone != quoted, text
    assert m_summary_starts(run_ts(S, "return m.SUMMARY;"))


def m_summary_starts(summary: str) -> bool:
    return summary.startswith("The person asked to “Draft the October issue from my notes”") and summary.endswith("A 280-word draft is done.")


@needs_node
def test_the_guess_is_scored_against_the_gone_rows():
    got = run_ts(S, """
      const gone = m.ROWS.map((r, i) => (r.gone ? i : -1)).filter((i) => i >= 0);
      const kept = m.ROWS.findIndex((r) => r.k === "you" && !r.gone);
      return [gone, m.gonePicks([gone[0], kept])];""")
    gone, scored = got
    assert scored["right"] == [gone[0]]
    assert scored["missed"] == gone[1:]
    assert len(scored["wrong"]) == 1


# Next's compiler drops the space that starts a JSX text running on past its
# line when that text holds an entity (named, decimal or hex): `<b>Tap the
# two</b> you think aren&rsquo;t…` then a line break renders "Tap the twoyou
# think". A space written as {" "} survives.
_LOSES_SPACE = re.compile(r"</[A-Za-z][\w.]*> [^<>{}]*&(?:[A-Za-z]+|#\d+|#x[0-9A-Fa-f]+);[^<>{}]*$")


def test_no_landing_text_loses_the_space_after_a_tag():
    hits = []
    for folder in (SRC / "components" / "landing", SRC / "app" / "(landing)"):
        for path in sorted(folder.glob("*.tsx")):
            for n, line in enumerate(path.read_text().splitlines(), 1):
                if _LOSES_SPACE.search(line):
                    hits.append(f"{path.name}:{n}")
    assert hits == []


def test_the_space_guard_knows_every_kind_of_entity():
    for line in ("<b>Tap the two</b> you think aren&rsquo;t",
                 "<b>Tap the two</b> you think aren&#8217;t",
                 "<b>Tap the two</b> you think aren&#x2019;t"):
        assert _LOSES_SPACE.search(line), line
    assert not _LOSES_SPACE.search('<b>Tap the two</b>{" "}you think aren&rsquo;t')
