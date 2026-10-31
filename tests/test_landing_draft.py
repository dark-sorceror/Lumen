"""The landing's composer hands its words to the chat app's composer."""
from __future__ import annotations

from tests.node_ts import needs_node, run_ts

pytestmark = needs_node

FAKE = """
const data = {};
const store = {
  setItem: (k, v) => { data[k] = v; },
  getItem: (k) => (k in data ? data[k] : null),
  removeItem: (k) => { delete data[k]; },
};
"""


def run(body: str):
    return run_ts("lib/landing.ts", FAKE + body)


def test_round_trip_and_removed_after_taking():
    out = run("""
      m.handOff("  hello  ", store);
      const first = m.takeLandingDraft(store);
      return [first, m.takeLandingDraft(store), m.LANDING_DRAFT_KEY in data];
    """)
    assert out == ["hello", None, False]


def test_expires_after_thirty_minutes():
    out = run("""
      m.handOff("hi", store);
      const at = JSON.parse(data[m.LANDING_DRAFT_KEY]).at;
      const fresh = m.takeLandingDraft(store, at + 30 * 60 * 1000);
      m.handOff("hi", store);
      const old = m.takeLandingDraft(store, at + 30 * 60 * 1000 + 60000);
      return [fresh, old, m.LANDING_DRAFT_KEY in data];
    """)
    assert out == ["hi", None, False]


def test_bad_entries_give_nothing_and_are_removed():
    out = run("""
      const res = [];
      for (const raw of ["{not json", JSON.stringify({text: "", at: Date.now()}),
                         JSON.stringify({text: "  ", at: Date.now()}),
                         JSON.stringify({text: 5, at: Date.now()}), "null"]) {
        data[m.LANDING_DRAFT_KEY] = raw;
        res.push([m.takeLandingDraft(store), m.LANDING_DRAFT_KEY in data]);
      }
      return res;
    """)
    assert out == [[None, False]] * 5


def test_empty_text_leaves_nothing_and_long_text_is_capped():
    out = run("""
      m.handOff("   ", store);
      const empty = m.LANDING_DRAFT_KEY in data;
      m.handOff("x".repeat(25000), store);
      return [empty, m.takeLandingDraft(store).length];
    """)
    assert out == [False, 20000]


def test_a_storage_that_throws_is_survived():
    out = run("""
      const bad = { setItem() { throw new Error("full"); },
                    getItem() { throw new Error("blocked"); },
                    removeItem() { throw new Error("blocked"); } };
      m.handOff("hi", bad);
      return m.takeLandingDraft(bad);
    """)
    assert out is None


def test_only_a_plain_enter_sends():
    out = run("""
      const k = (key, shiftKey = false, isComposing = false) => m.isSendKey({ key, shiftKey, isComposing });
      return [k("Enter"), k("Enter", true), k("Enter", false, true), k("a")];
    """)
    assert out == [True, False, False, False]


def test_starting_a_chat_leaves_the_words_then_goes_to_chat():
    out = run("""
      const went = [];
      m.startChat("  hi there ", (p) => went.push(p), store);
      const first = m.takeLandingDraft(store);
      m.startChat("   ", (p) => went.push(p), store);
      return [first, m.takeLandingDraft(store), went];
    """)
    assert out == ["hi there", None, ["/chat/", "/chat/"]]
