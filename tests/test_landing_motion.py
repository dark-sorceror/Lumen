"""Reduced motion on the landing: every state still reachable, and none
waiting on motion. Read as text, statically (the frontend has no test
runner), from frontend/src/components/landing and app/(landing):

- every transition or animation that waits before it starts is undone
  under `prefers-reduced-motion: reduce` (globals.css shortens durations
  there but keeps delays, so a delay would still hold a state back);
- every figure that moves on a timer or a frame asks reducedMotion()
  first, and puts its state in place at once when it says so."""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "frontend" / "src"
LANDING = SRC / "components" / "landing"
SHEETS = sorted(LANDING.glob("*.module.css")) + [SRC / "app" / "(landing)" / "page.module.css"]
REDUCE = "(prefers-reduced-motion: reduce)"
_TIME = re.compile(r"(?<![\w-])\d*\.?\d+m?s\b")


def rules(css: str, media: str = "") -> list[tuple[str, str, str]]:
    """(the @media it sits in, its selector, its declarations) for each rule,
    @media blocks opened, comments gone. @supports and @starting-style
    blocks are passed through as their own wrappers."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    out, i = [], 0
    while True:
        brace = css.find("{", i)
        if brace < 0:
            return out
        head = css[i:brace].strip()
        depth, j = 1, brace + 1
        while depth:
            depth += {"{": 1, "}": -1}.get(css[j], 0)
            j += 1
        body = css[brace + 1:j - 1]
        if head.startswith("@keyframes") or head.startswith("@property"):
            pass
        elif head.startswith("@"):
            out += rules(body, f"{media} {head}".strip())
        else:
            out.append((media, " ".join(head.split()), body))
        i = j


def delayed(decls: str, kind: str) -> bool:
    """Whether a rule's transitions (or animations) wait before they start."""
    for name, value in re.findall(r"([\w-]+)\s*:\s*([^;]+)", decls):
        if name == f"{kind}-delay" and not re.fullmatch(r"0m?s|0", value.strip()):
            return True
        if name == kind:
            # A shorthand's second time is its delay, item by item.
            items = re.split(r",(?![^(]*\))", value)
            if any(len(_TIME.findall(item)) >= 2 for item in items):
                return True
    return False


def undone(selector: str, kind: str, sheet: list[tuple[str, str, str]]) -> bool:
    """Whether reduced motion takes the wait away from every selector."""
    reset = (r"transition-delay\s*:\s*0m?s|transition\s*:\s*none" if kind == "transition"
             else r"animation\s*:\s*none|animation-delay\s*:\s*0m?s")
    reduced = [(sel, decls) for media, sel, decls in sheet if REDUCE in media]
    for one in (s.strip() for s in selector.split(",")):
        if not any(one in [x.strip() for x in sel.split(",")] and re.search(reset, decls)
                   for sel, decls in reduced):
            return False
    return True


def test_no_state_waits_on_a_delay_under_reduced_motion():
    left = []
    for path in SHEETS:
        sheet = rules(path.read_text())
        for media, selector, decls in sheet:
            if REDUCE in media or "no-preference" in media:
                continue
            for kind in ("transition", "animation"):
                if delayed(decls, kind) and not undone(selector, kind, sheet):
                    left.append(f"{path.name}: {selector} ({kind})")
    assert left == []


# Files that time things for other reasons than motion: the Copied label,
# the in-view watch, and the toast, whose timer hides a message.
NOT_MOTION = {"CopyButton.tsx", "useInView.ts", "Toast.tsx"}


def test_every_figure_that_moves_on_its_own_asks_about_reduced_motion_first():
    timed = re.compile(r"\bsetTimeout\(|\brequestAnimationFrame\(|\.animate\(|startViewTransition")
    left = [p.name for p in sorted(LANDING.glob("*.ts*"))
            if p.name not in NOT_MOTION and timed.search(p.read_text())
            and "reducedMotion()" not in p.read_text()]
    assert left == []


def test_the_figures_that_play_once_skip_straight_to_their_state():
    # Each of these, with reduced motion, sets its resting state at once
    # instead of travelling to it.
    for name, at_once in {
        "Footer.tsx": "setLit(true)",
        "AppDrawing.tsx": 'setPhase("done")',
        "TerminalDrawing.tsx": "setAt(SCRIPT.length)",
        "StorySection.tsx": "setShownPct(to)",
    }.items():
        text = (LANDING / name).read_text()
        assert "reducedMotion()" in text and at_once in text, name
