// Which of the story's steps the reader is on, and where to scroll to read
// one, as pure arithmetic over where the steps and the figure are on screen
// (useActiveStep measures them). It imports nothing, so a test can run it.

/** Beside the figure (a wide screen), a step switches on when its middle
 * reaches this far down the screen, as a share of its height. */
export const LINE = 0.65;

/** Where the figure pins over the steps (a phone), a step switches on when
 * its top reaches this far below the figure's bottom, as a share of the
 * screen's height. A fixed distance, so that when the figure grows, the line
 * moves exactly as far as the steps under it do, and the step chosen stays
 * chosen. */
export const UNDER = 0.18;

/** A step already reached stays reached this many px past its line, and one
 * not reached yet needs this many px more: rounding can't flip it. */
export const HYSTERESIS = 4;

/** The line across the screen a step's words have to reach. `pinnedBottom`
 * is the bottom of a figure pinned over the steps, or null where it sits
 * beside them. */
export function readingLine(vh: number, pinnedBottom: number | null): number {
  if (pinnedBottom === null) return vh * LINE;
  const bottom = Math.min(vh, Math.max(0, pinnedBottom));
  return Math.min(vh, bottom + UNDER * vh);
}

/** The step on screen, from 0: the last whose words have reached the line.
 * `tops` and `heights` are the steps' boxes on screen; `current` is the
 * step chosen last, or null before the first choice. Under a pinned figure
 * a step's top counts; beside it, a step's middle. */
export function pickStep(o: {
  tops: number[];
  heights: number[];
  vh: number;
  pinnedBottom: number | null;
  current: number | null;
}): number {
  const line = readingLine(o.vh, o.pinnedBottom);
  const byTop = o.pinnedBottom !== null;
  let n = 0;
  o.tops.forEach((top, i) => {
    const slack = o.current !== null && i <= o.current ? HYSTERESIS : -HYSTERESIS;
    if (top + (byTop ? 0 : o.heights[i] / 2) <= line + slack) n = i;
  });
  return n;
}

/** Where to scroll the page so one step is the one read: its words a
 * quarter of a step past the line, so a nudge back doesn't undo it, and
 * never above where the figure pins, so the first step lands with all of
 * the figure in view. `top` and `height` are the step's box on screen now,
 * `scrollTop` the page's scroll now, `storyTop` the story's top on screen
 * now, and `pinTop` how far down the figure pins. `pinnedBottom` is the
 * bottom a figure pinned over the steps will have once pinned (`pinTop`
 * plus its height), or null where it sits beside them. */
export function landAt(o: {
  top: number;
  height: number;
  vh: number;
  pinnedBottom: number | null;
  storyTop: number;
  pinTop: number;
  scrollTop: number;
}): number {
  const line = readingLine(o.vh, o.pinnedBottom);
  const at = o.top + (o.pinnedBottom === null ? o.height / 2 : 0);
  const y = o.scrollTop + at - (line - o.height / 4);
  const pins = o.scrollTop + o.storyTop - o.pinTop;
  return Math.max(0, Math.round(Math.max(y, pins)));
}
