"""In-page links glide to their section and stop just under the sticky nav;
the logo goes to the very top. The hosted site is told apart from a Mac."""
from __future__ import annotations

from tests.node_ts import needs_node, run_ts

GLIDE = "components/landing/glide.ts"
SERVED = "components/landing/served.ts"


@needs_node
def test_a_section_lands_twelve_pixels_under_the_nav():
    assert run_ts(GLIDE, "return m.glideY(400, 1000, 56);") == 1332


@needs_node
def test_the_top_is_zero_and_nothing_goes_above_it():
    assert run_ts(GLIDE, "return [m.glideY(null, 900, 56), m.glideY(10, 0, 56)];") == [0, 0]


@needs_node
def test_only_plain_in_page_links_glide():
    got = run_ts(GLIDE, """return ["#top", "#", "#plugin", "https://x.dev/#a", "/chat/", null].map(m.inPageId);""")
    assert got == ["top", "top", "plugin", None, None, None]


@needs_node
def test_the_hosted_site_is_illumen_lol_or_github_pages():
    got = run_ts(SERVED, """return ["illumen.lol", "www.illumen.lol", "dark-sorceror.github.io",
        "localhost", "127.0.0.1", ""].map(m.isHosted);""")
    assert got == [True, True, True, False, False, False]
