"""The illumen brand's base on the landing: its tokens in both palettes, the
mark, Sora bundled with the page, and the page's name."""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "frontend" / "src"
LANDING = SRC / "app" / "(landing)"
CSS = (LANDING / "globals.css").read_text()

LIGHT = {
    "--paper": "#fbf8f4", "--paper-2": "#f4efe8", "--surface": "#ffffff", "--fill": "#f1ece5", "--fill-2": "#e7e1d8",
    "--ink": "#1f2433", "--ink-2": "#4c5266", "--ink-3": "#6a7085",
    "--line": "rgba(31, 36, 51, 0.09)", "--line-2": "rgba(31, 36, 51, 0.17)",
    "--live": "#ff8f66", "--live-ink": "#b34a1d", "--live-wash": "rgba(255, 143, 102, 0.16)",
    "--gone": "#d0314a", "--gone-ink": "#b42339", "--gone-wash": "rgba(208, 49, 74, 0.08)",
    "--faded": "rgba(31, 36, 51, 0.56)", "--scrim": "rgba(20, 24, 36, 0.28)",
}
DARK = {
    "--paper": "#121110", "--paper-2": "#0d0c0b", "--surface": "#1b1a18", "--fill": "#252321", "--fill-2": "#302e2b",
    "--ink": "#f5f0ea", "--ink-2": "#cbc5bd", "--ink-3": "#a09a92",
    "--line": "rgba(245, 240, 234, 0.09)", "--line-2": "rgba(245, 240, 234, 0.18)",
    "--live": "#ffa07d", "--live-ink": "#ffb595", "--live-wash": "rgba(255, 160, 125, 0.14)",
    "--gone": "#ff5f74", "--gone-ink": "#ff8a99", "--gone-wash": "rgba(255, 95, 116, 0.12)",
    "--faded": "rgba(245, 240, 234, 0.52)", "--scrim": "rgba(0, 0, 0, 0.5)",
}
RIM = "M12.51 21.79A22 22 0 1 1 42.21 51.49A21 21 0 1 0 12.51 21.79Z"
DISC = "M10.07 30.32A18.5 18.5 0 1 1 33.68 53.93A22 22 0 0 1 10.07 30.32Z"


def _block(start: int) -> str:
    """The text inside the braces that open at or after `start`."""
    i = CSS.index("{", start) + 1
    depth, j = 1, i
    while depth:
        depth += {"{": 1, "}": -1}.get(CSS[j], 0)
        j += 1
    return CSS[i:j - 1]


def _has(block: str, tokens: dict[str, str]) -> list[str]:
    return [k for k, v in tokens.items() if not re.search(rf"(?<![\w-]){re.escape(k)}:\s*{re.escape(v)};", block)]


def test_the_light_palette_is_paper_and_ink():
    assert _has(_block(CSS.index(":root {")), LIGHT) == []


def test_the_dark_palette_is_warm_black():
    assert _has(_block(CSS.index("@media (prefers-color-scheme: dark)")), DARK) == []


def test_no_earlier_colour_is_left_in_the_palettes():
    for old in ("#efae00", "#ffc533", "#f7f6f3", "#171716", "rgba(27, 26, 23", "rgba(236, 235, 231"):
        assert old not in CSS, old


def test_the_mark_is_a_disc_with_a_lit_rim():
    mark = (SRC / "components" / "IllumenMark.tsx").read_text()
    assert RIM in mark and DISC in mark
    assert 'viewBox="0 0 64 64"' in mark and "var(--mark-lit)" in mark


def test_the_icon_is_the_mark_on_warm_black():
    icon = (LANDING / "icon.svg").read_text()
    assert RIM in icon and DISC in icon
    assert "#121110" in icon and "#ffa07d" in icon and "#f5f0ea" in icon


def test_sora_is_bundled_not_fetched():
    layout = (LANDING / "layout.tsx").read_text()
    assert "next/font/local" in layout and "next/font/google" not in layout
    assert (LANDING / "fonts" / "Sora-Variable.woff2").stat().st_size > 10_000
    assert re.search(r"--font-display:\s*var\(--font-sora\)", CSS)


def test_the_page_is_named_illumen():
    layout = (LANDING / "layout.tsx").read_text()
    assert 'title: "illumen"' in layout
    assert "Lumen" not in layout


def test_the_wordmark_is_lowercase():
    assert ">illumen<" in (SRC / "components" / "Wordmark.tsx").read_text()
