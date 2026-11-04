"""The story's scroll lock: on a screen tall enough to read beside or under
it, the story's figure (the chat and its meter) pins while its five steps
scroll past, and the step whose words reach a line across the screen is the
one the figure shows. The geometry runs under node (stepLine.ts imports
nothing)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.node_ts import needs_node, run_ts
from tests.test_landing_motion import rules

LINE = "components/landing/stepLine.ts"
LANDING = Path(__file__).resolve().parent.parent / "frontend" / "src" / "components" / "landing"

# The story as it lays out, in px on screen, `p` px into the page's scroll,
# with step `step` (from 0) showing. Beside (a wide screen): the figure pins
# 88 px down, beside five steps of `h` (a share of the screen) that start at
# the story's top. Over (a phone): the figure pins under the 56 px nav, about
# as tall as it measures (its chat frame grows with the screen), `grow` px
# taller at the last step (where Next becomes Start over and the caption
# beside it can wrap once more), and the steps follow it in the page. It
# pins until the story's end: beside, `tail` of the screen past the last
# step; over, right after the last step (`overTail`).
_MODEL = r"""
const BESIDE = [[1280, 900], [1280, 700], [1440, 800], [1024, 768], [1280, 560], [960, 500]];
const OVER = [[375, 548], [360, 560], [360, 640], [390, 844], [400, 800], [320, 568], [768, 1024]];
const o = Object.assign({ wide: 0.45, narrow: 0.4, tail: 0.3, overTail: 0, grow: 17 }, input || {});
function figure(vh) {
  return 184 + Math.min(260, Math.max(120, vh - 440));
}
function wideFigure(vh) {
  return 188 + Math.min(430, Math.max(220, vh - 300));
}
function layout(kind, vh, p, step) {
  const top = vh - p;
  const tops = [], heights = [];
  if (kind === "beside") {
    const H = o.wide * vh;
    for (let i = 0; i < 5; i++) { tops.push(top + i * H); heights.push(H); }
    return { tops, heights, storyTop: top, pinTop: 88, pinnedBottom: null, end: 5 * H + o.tail * vh };
  }
  const H = o.narrow * vh;
  const stage = figure(vh) + (step === 4 ? o.grow : 0);
  for (let i = 0; i < 5; i++) { tops.push(top + stage + i * H); heights.push(H); }
  const bottom = top + stage + 5 * H + o.overTail * vh;
  const at = Math.min(Math.max(56, top), bottom - stage);
  return { tops, heights, storyTop: top, pinTop: 56, pinnedBottom: at + stage, end: stage + 5 * H + o.overTail * vh };
}
// The step shown at `p`, frame after frame from `current`, as the figure
// takes each step's height: and whether it was still changing after two.
function settle(pick, kind, vh, p, current) {
  const seen = [];
  for (let frame = 0; frame < 6; frame++) {
    const l = layout(kind, vh, p, current);
    current = pick({ tops: l.tops, heights: l.heights, vh, pinnedBottom: l.pinnedBottom, current });
    seen.push(current);
  }
  return { step: current, flickers: seen.slice(2).some((s) => s !== current) };
}
// Every px from above the story to past its end, down and back up: the
// steps seen, in the order seen, and the px at which the step didn't settle.
function scrollThrough(pick, kind, vh) {
  const end = Math.round(vh + layout(kind, vh, 0, 0).end + vh);
  const out = { down: [], up: [], flickers: 0 };
  for (const down of [true, false]) {
    let current = null;
    const seen = down ? out.down : out.up;
    for (let k = 0; k <= end; k++) {
      const s = settle(pick, kind, vh, down ? k : end - k, current);
      current = s.step;
      if (s.flickers) out.flickers++;
      if (seen[seen.length - 1] !== current) seen.push(current);
    }
  }
  return out;
}
"""


@needs_node
def test_scrolling_through_the_story_shows_each_step_once_in_order():
    out = run_ts(LINE, _MODEL + """
      const all = [];
      for (const [kind, sizes] of [["beside", BESIDE], ["over", OVER]])
        for (const [w, vh] of sizes) all.push([kind, w, vh, scrollThrough(m.pickStep, kind, vh)]);
      return all;""")
    for kind, w, h, seen in out:
        assert seen["down"] == [0, 1, 2, 3, 4], (kind, w, h, seen["down"])
        assert seen["up"] == [4, 3, 2, 1, 0], (kind, w, h, seen["up"])
        assert seen["flickers"] == 0, (kind, w, h)


# An earlier rule for the line: a third of the way down what the
# pinned figure leaves. When the figure grows, that line moves less far than
# the steps under it do.
_BEFORE = r"""
function before(q) {
  const bottom = Math.min(q.vh, Math.max(0, q.pinnedBottom));
  const line = bottom + (q.vh - bottom) / 3;
  let n = 0;
  q.tops.forEach((top, i) => { if (top <= line) n = i; });
  return n;
}
"""


@needs_node
def test_the_step_under_a_pinned_figure_settles_when_the_figure_grows():
    # Whether Start over wraps the caption once more or twice, the step
    # under the figure settles at every px; the old line flipped at many.
    for grow in (17, 40):
        out = run_ts(LINE, _MODEL + _BEFORE + """
          let now = 0, was = 0;
          for (const [, vh] of OVER) {
            now += scrollThrough(m.pickStep, "over", vh).flickers;
            was += scrollThrough(before, "over", vh).flickers;
          }
          return { now, was };""", {"grow": grow})
        assert out["now"] == 0, grow
        assert out["was"] > 20, (grow, out["was"])


@needs_node
def test_the_line_is_a_fixed_distance_under_a_pinned_figure_and_mid_screen_beside_it():
    got = run_ts(LINE, """return { under: m.UNDER, beside: m.LINE, at: [
      m.readingLine(800, 500), m.readingLine(800, 540), m.readingLine(900, null), m.readingLine(700, null)] };""")
    under, beside, at = got["under"], got["beside"], got["at"]
    # Under a pinned figure: when the figure grows 40 px, the line moves 40.
    assert at[0] == pytest.approx(500 + under * 800)
    assert at[1] - at[0] == pytest.approx(40)
    # Beside it, the line is a share of the screen, the figure aside.
    assert at[2] == pytest.approx(beside * 900) and at[3] == pytest.approx(beside * 700)


@needs_node
def test_a_step_counts_by_its_top_under_the_figure_and_by_its_middle_beside_it():
    got = run_ts(LINE, """
      const vh = 1000, heights = [300, 300, 300];
      const at = (tops, pinnedBottom) => m.pickStep({ tops, heights, vh, pinnedBottom, current: null });
      const beside = m.readingLine(vh, null);
      // A figure whose bottom puts the line 10 px under the second step's top.
      const over = 400 + 10 - m.UNDER * vh;
      return [
        at([beside - 450, beside - 150 - 10, beside + 150], null),
        at([beside - 450, beside - 150 + 10, beside + 150], null),
        at([100, 400, 700], over),
        at([500, 800, 1100], 0),
      ];""")
    # Beside: the second step counts once its middle is past the line,
    # not its top.
    assert got[:2] == [1, 0]
    # Under: its top counts. And before any step reaches the line, the
    # first is the one shown.
    assert got[2:] == [1, 0]


@needs_node
def test_a_step_on_its_line_stays_put_for_a_px_or_two_either_way():
    got = run_ts(LINE, """
      const q = (top, current) => m.pickStep({ tops: [0, top], heights: [0, 0], vh: 1000,
        pinnedBottom: 0, current });
      const line = m.readingLine(1000, 0), h = m.HYSTERESIS;
      return [q(line + h - 1, 1), q(line - h + 1, 0), q(line + h + 1, 1), q(line - h - 1, 0)];""")
    assert got == [1, 0, 0, 1]


@needs_node
def test_the_stepper_lands_each_step_on_its_line():
    # From anywhere on the page, a step's button (or Next) glides to where
    # that step is the one shown; for the first, to where the figure pins,
    # so all of it is in view.
    out = run_ts(LINE, _MODEL + """
      const res = [];
      for (const [kind, sizes] of [["beside", BESIDE], ["over", OVER]]) {
        for (const [w, vh] of sizes) {
          const end = Math.round(vh + layout(kind, vh, 0, 0).end);
          for (const p0 of [0, Math.round(vh * 1.3), Math.round(end / 2), end]) {
            const c0 = settle(m.pickStep, kind, vh, p0, null).step;
            const l = layout(kind, vh, p0, c0);
            for (let i = 0; i < 5; i++) {
              const stage = l.pinnedBottom === null ? null : l.pinTop + (figure(vh) + (c0 === 4 ? o.grow : 0));
              const y = m.landAt({ top: l.tops[i], height: l.heights[i], vh, pinnedBottom: stage,
                storyTop: l.storyTop, pinTop: l.pinTop, scrollTop: p0 });
              // The glide, a frame at a time, and where it comes to rest.
              let current = c0;
              const n = Math.max(1, Math.ceil(Math.abs(y - p0) / 37));
              for (let f = 1; f <= n; f++) current = settle(m.pickStep, kind, vh, p0 + ((y - p0) * f) / n, current).step;
              const at = layout(kind, vh, y, current);
              const pinned = at.pinnedBottom === null
                ? at.storyTop + at.end - wideFigure(vh) >= at.pinTop
                : Math.abs(at.pinnedBottom - (figure(vh) + (current === 4 ? o.grow : 0)) - at.pinTop) < 0.5;
              res.push([kind, w, vh, p0, i, current, Math.round(at.storyTop - at.pinTop), pinned]);
            }
          }
        }
      }
      return res;""")
    for kind, w, h, p0, i, shown, past_pin, pinned in out:
        assert shown == i, (kind, w, h, p0, i, shown)
        assert pinned, (kind, w, h, p0, i)
        if i == 0:
            assert past_pin == 0, (kind, w, h, p0, past_pin)
        else:
            assert past_pin < 0, (kind, w, h, p0, i, past_pin)


# The layout rules, read from the stylesheet as a browser would cascade them
# at real sizes: phones upright (an iPhone SE with Safari's bars shown, a
# 360 x 560 Android, a 390 x 844, a 400 x 800), phones on their side (a Pro
# Max is 932 wide on its side, past the story's 900 px phone width), a
# tablet and laptops, and short windows.
UPRIGHT = [(375, 548), (360, 560), (390, 844), (320, 568), (400, 800)]
SIDEWAYS = [(667, 375), (844, 390), (568, 320), (932, 430)]
WIDE = [(768, 1024), (1024, 600), (1280, 800), (1280, 900)]
SHORT = [(1280, 480), (960, 420)]
EDGES = [(375, 480), (375, 480.5), (375, 481), (1280, 480), (1280, 480.5)]

_FEATURE = re.compile(r"\((min|max)-(width|height):\s*([\d.]+)px\)")
_RANGE = re.compile(r"\((width|height)\s*(<=|>=|<|>)\s*([\d.]+)px\)")


def applies(media: str, w: float, h: float) -> bool:
    """Whether a rule's @media wrappers hold at a w x h screen. Only width
    and height are asked, as min-/max- features or ranges; a rule under any
    other at-rule (@supports, @starting-style, reduced motion) is left out."""
    for part in re.split(r"(?=@)", media):
        part = part.strip()
        if not part:
            continue
        if not part.startswith("@media"):
            return False
        query = part[len("@media"):].strip()
        if "prefers-reduced-motion" in query:
            return False
        for bound, axis, px in _FEATURE.findall(query):
            size = w if axis == "width" else h
            if (bound == "min" and size < float(px)) or (bound == "max" and size > float(px)):
                return False
        for axis, op, px in _RANGE.findall(query):
            size, px = (w if axis == "width" else h), float(px)
            if not {"<": size < px, "<=": size <= px, ">": size > px, ">=": size >= px}[op]:
                return False
    return True


def cascade(css: str, selector: str, prop: str, w: float, h: float) -> str | None:
    """The value a selector's property ends with at a w x h screen: the
    last declaration that applies, as the same selector's rules cascade,
    an !important one over any that isn't."""
    value, important = None, False
    for media, sel, decls in rules(css):
        if selector in [s.strip() for s in sel.split(",")] and applies(media, w, h):
            for name, v in re.findall(r"([\w-]+)\s*:\s*([^;]+)", decls):
                if name == prop and ("!important" in v or not important):
                    important = important or "!important" in v
                    value = v.replace("!important", "").strip()
    return value


def story_css() -> str:
    return (LANDING / "StorySection.module.css").read_text()


def sticky(css: str, w: float, h: float) -> bool:
    return cascade(css, ".stage", "position", w, h) == "sticky"


def test_the_figure_pins_on_every_upright_screen_and_unpins_when_short():
    # An iPhone SE with Safari's bars shown keeps the pinned figure; a phone
    # on its side, or any window too short to read beside or under it, lays
    # the story out as before, with nothing pinned.
    css = story_css()
    for w, h in UPRIGHT + WIDE:
        assert sticky(css, w, h), (w, h)
    for w, h in SIDEWAYS + SHORT:
        assert not sticky(css, w, h), (w, h)


def test_the_hook_reads_the_steps_exactly_where_the_figure_pins():
    # The hook follows the scroll wherever it doesn't match UNPINNED, and
    # reads under the figure where NARROW matches too: at every height, a
    # fractional one included, that is where the stylesheet pins it, and
    # pins it over the steps.
    css = story_css()
    hook = (LANDING / "useActiveStep.ts").read_text()
    narrow = re.search(r'NARROW = "([^"]+)"', hook).group(1)
    unpinned = re.search(r'UNPINNED = "([^"]+)"', hook).group(1)
    for w, h in UPRIGHT + SIDEWAYS + WIDE + SHORT + EDGES:
        follows = not applies(f"@media {unpinned}", w, h)
        under = follows and applies(f"@media {narrow}", w, h)
        assert follows == sticky(css, w, h), (w, h)
        assert under == (sticky(css, w, h) and cascade(css, ".stage", "order", w, h) == "-1"), (w, h)


def test_where_the_figure_pins_over_the_steps_they_move_with_it():
    # Wherever the figure pins over the steps (a phone upright), it comes
    # first, a step a key moves focus to lands under the figure as measured,
    # and no scroll anchoring holds the steps still when the figure grows.
    # Where it doesn't pin over them, none of that applies.
    css = story_css()
    for w, h in UPRIGHT + SIDEWAYS + WIDE + SHORT + EDGES:
        over = w <= 900 and sticky(css, w, h)
        margin = cascade(css, ".stepper button", "scroll-margin-top", w, h) or ""
        anchor = cascade(css, ".story", "overflow-anchor", w, h) or "auto"
        assert ("--stage-h" in margin) == over, (w, h, margin)
        assert (anchor == "none") == over, (w, h, anchor)
        assert (cascade(css, ".stage", "order", w, h) == "-1") == over, (w, h)
    hook = (LANDING / "useActiveStep.ts").read_text()
    # The height is the figure's own, measured as it changes.
    assert "observer.observe(stage)" in hook
    assert 'setProperty("--stage-h", `${stage.offsetHeight}px`)' in hook


def test_the_steps_take_room_to_scroll_only_where_the_figure_pins():
    # Pinned, each step is a stretch of scroll and the stepper is the steps;
    # unpinned, the story is as it was: a short list beside the figure, or
    # a row of pills and the step's words above it on a phone.
    css = story_css()
    for w, h in UPRIGHT + SIDEWAYS + WIDE + SHORT + EDGES:
        pinned = sticky(css, w, h)
        pills = w <= 900 and not pinned
        tall = cascade(css, ".step", "min-height", w, h) or "0"
        assert ("vh" in tall) == pinned, (w, h, tall)
        assert (cascade(css, ".stepper", "flex-direction", w, h) == "row") == pills, (w, h)
        shown = cascade(css, ".stepNow", "display", w, h) or ""
        assert shown.startswith("block") == pills, (w, h, shown)


def test_the_stepper_and_next_glide_the_landing_s_own_scroller():
    # The landing scrolls inside its own root, as its links do
    # (useGlideLinks.ts), at once with reduced motion.
    hook = (LANDING / "useActiveStep.ts").read_text()
    story = (LANDING / "StorySection.tsx").read_text()
    assert 'closest<HTMLElement>("[data-scroller]")' in hook
    assert 'reducedMotion() ? "auto" : "smooth"' in hook
    assert "glideToStep(" in story and "landAt(" in hook


def test_the_figure_over_the_steps_keeps_room_for_a_two_digit_number():
    # As the meter counts from 6% to 88%, the chat's name and model beside it
    # mustn't wrap once more and grow the figure the steps are read under:
    # where it pins over them, the number keeps the room that 88% takes.
    css = story_css()
    for w, h in UPRIGHT + SIDEWAYS + WIDE + SHORT + EDGES:
        over = w <= 900 and sticky(css, w, h)
        room = cascade(css, ".big", "min-width", w, h) or ""
        assert room.endswith("em") == over, (w, h, room)


def test_the_figure_over_the_steps_leaves_with_the_last_step():
    # On a phone nothing follows the last step under the pinned figure: the
    # story ends with that step, and the figure goes on up with the page
    # rather than hanging over blank page.
    css = story_css()
    for w, h in UPRIGHT + SIDEWAYS + WIDE + SHORT + EDGES:
        if w <= 900 and sticky(css, w, h):
            assert cascade(css, ".stepper", "padding-bottom", w, h) in ("0", "0px"), (w, h)


# Phones upright but short: an iPhone SE with Safari's bars shown, Androids
# with Chrome's.
SHORT_UPRIGHT = [(375, 548), (360, 560), (320, 568), (390, 600), (375, 620)]


def frame_px(value: str, h: float) -> float:
    """The chat frame's height at a screen `h` tall, from its clamp()."""
    m = re.fullmatch(r"clamp\((\d+)px,\s*calc\(100svh - (\d+)px\),\s*(\d+)px\)", value)
    assert m, value
    least, less, most = map(float, m.groups())
    return min(most, max(least, h - less))


def test_the_guess_list_keeps_a_swipe_to_itself_only_on_a_touch_phone_mid_guess():
    # On a touch phone, under the pinned figure, a swipe that runs past the
    # end of the unanswered guess's list mustn't carry on into the page: that
    # scrolls the reader on to step 5, which shows the answers, and back
    # again, which used to clear the picks. Nowhere else: a wheel over the list on a
    # wider screen, or any swipe once the answers show, moves the page as it
    # would anywhere on it.
    css = story_css()
    held = [(media, sel) for media, sel, decls in rules(css)
            if re.search(r"overscroll-behavior\s*:\s*contain", decls)]
    assert held == [("@media (max-width: 900px) and (height > 480px) and (pointer: coarse)", ".typed:not(.done)")]
    media = held[0][0]
    for w, h in UPRIGHT + SHORT_UPRIGHT + SIDEWAYS + WIDE + SHORT + EDGES:
        # Its width and height are where the figure pins over the steps.
        assert applies(media, w, h) == (w <= 900 and sticky(css, w, h)), (w, h)
    # The revealed list is the same element, with .done added.
    story = (LANDING / "StorySection.tsx").read_text()
    assert "`${styles.typed}${revealed ? ` ${styles.done}` : \"\"}`" in story


def test_a_short_phone_gives_the_guess_room_and_shows_that_the_list_scrolls():
    # On a phone 620 px tall or less the figure drops its caption and
    # tightens its top row; from an iPhone SE's 548 px up, the chat shows the
    # guess's question (72 px at 375 wide) and two of its rows (38 px each,
    # with their gaps).
    css = story_css()
    for w, h in UPRIGHT + SHORT_UPRIGHT + SIDEWAYS + WIDE + SHORT + EDGES:
        short = w <= 900 and 480 < h <= 620 and sticky(css, w, h)
        caption = cascade(css, ".figFoot > span", "display", w, h)
        assert (caption == "none") == short, (w, h, caption)
        if short and h >= 548:
            assert frame_px(cascade(css, ".figBody", "height", w, h), h) >= 4 + 72 + 4 + 8 + 38 + 8 + 38, (w, h)
    # Where the list runs on below, a fade in the figure's own colour says so.
    assert cascade(css, ".typed::after", "position", 375, 548) == "sticky"
    assert "var(--surface)" in (cascade(css, ".typed::after", "background", 375, 548) or "")


def test_a_row_moved_to_in_the_list_stops_clear_of_its_fade():
    # Tab (or a redraw) scrolls the list to a row; the row stops above the
    # fade at the list's foot, not under it.
    css = story_css()
    fade = cascade(css, ".typed::after", "height", 375, 548)
    for w, h in UPRIGHT + SHORT_UPRIGHT + SIDEWAYS + WIDE + SHORT + EDGES:
        assert cascade(css, ".typed", "scroll-padding-bottom", w, h) == fade, (w, h)


def test_the_figure_over_the_steps_sets_the_list_on_a_whole_pixel():
    # A list sitting on a fractional pixel swallows the first swipe after
    # script has scrolled it to its end (Put back's scrollIntoView, or a focus
    # scroll). Above the list, only the chat's name and model ran on
    # fractional lines (12 px at 1.4); where the figure pins over the steps,
    # their lines are whole pixels, which keeps the list on a whole pixel.
    css = story_css()
    for w, h in UPRIGHT + SHORT_UPRIGHT + SIDEWAYS + WIDE + SHORT + EDGES:
        if w <= 900 and sticky(css, w, h):
            line = cascade(css, ".who", "line-height", w, h) or ""
            assert re.fullmatch(r"\d+px", line), (w, h, line)


def test_reading_back_from_step_5_to_step_4_keeps_the_guess_but_going_below_resets_it():
    story = (LANDING / "StorySection.tsx").read_text()
    go = story[story.index("function go("):story.index("// As the steps scroll")]
    assert "if (next < 3) {" in go
    assert "prev === 4" not in go
    # The reset still clears all three.
    reset = go[go.index("if (next < 3) {"):]
    for call in ("setPicks([])", "setRevealed(false)", "setPutBack([])"):
        assert call in reset
