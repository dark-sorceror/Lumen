"""The lower sections: each way in says where your words go, and the plugin
strip's install lines are illumen's."""
from __future__ import annotations

from pathlib import Path

from tests.node_ts import needs_node, run_ts

LANDING = Path(__file__).resolve().parent.parent / "frontend" / "src" / "components" / "landing"


@needs_node
def test_every_way_in_names_where_what_you_send_goes():
    ways = run_ts("components/landing/waysIn.ts", "return m.WAYS;")
    assert [w["id"] for w in ways] == ["cc", "cx", "qw", "an", "oa"]
    for w in ways:
        assert w["leave"].strip(), w["id"]


@needs_node
def test_the_strip_installs_illumen():
    lines = run_ts("lib/landing.ts", "return m.PLUGIN_INSTALL;")
    assert lines[1] == "claude plugin install illumen@illumen"


def test_the_strip_says_what_the_plugin_shows_in_its_own_words():
    strip = (LANDING / "PluginStrip.tsx").read_text()
    for words in ("A line under the prompt", "A row where it summarised", "/illumen-forgot"):
        assert words in strip
