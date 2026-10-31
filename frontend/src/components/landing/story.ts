// The story's content and its one rule: what the chat said, step by step,
// and which of the person's messages the summary dropped. Pure, so a test
// can read it.

export type Step = { t: string; s: string; pct: number };
export type Row = { k: "you" | "work"; t: string; at: number; gone?: boolean };

export const STEPS: Step[] = [
  { t: "You start talking.", s: "It reads its instructions, its tools and what you type.", pct: 6 },
  { t: "It fills up.", s: "Every message, every file it opens and every reply takes room.", pct: 88 },
  { t: "It summarises to keep going.", s: "Near the limit, the agent swaps the chat for a short summary.", pct: 11 },
  { t: "Some of what you typed is gone.", s: "The summary is all it reads now. Can you tell what didn't make it?", pct: 11 },
  { t: "illumen shows you which.", s: "Copy it, or put it back in your next message.", pct: 11 },
];

export const ROWS: Row[] = [
  { k: "you", t: "Draft the October issue from my notes.", at: 0 },
  { k: "work", t: "Read notes.md", at: 0 },
  { k: "you", t: "Sign it from Priya, not “the team”.", at: 0, gone: true },
  { k: "you", t: "Lead with the move to the new office on the 14th.", at: 1 },
  { k: "work", t: "Wrote a 280-word draft", at: 1 },
  { k: "you", t: "Keep it under 300 words.", at: 1 },
  { k: "you", t: "Don't mention the parking changes yet.", at: 1, gone: true },
  { k: "work", t: "Read last month's issue", at: 1 },
  { k: "you", t: "Add the two new hires at the end.", at: 1 },
  { k: "work", t: "Rewrote the draft twice", at: 1 },
];

/** The summary the chat shows, without its bold heading. */
export const SUMMARY =
  "The person asked to “Draft the October issue from my notes”, to “Lead with the move to the new office on the 14th”, to “Keep it under 300 words” and to “Add the two new hires at the end”. A 280-word draft is done.";

/** A guess, as row numbers in ROWS, scored: the picks that are gone, the
 * gone rows not picked, and the picks that aren't gone. Each in row order. */
export function gonePicks(picked: number[]): { right: number[]; missed: number[]; wrong: number[] } {
  const chosen = new Set(picked);
  const right: number[] = [];
  const missed: number[] = [];
  const wrong: number[] = [];
  ROWS.forEach((row, i) => {
    if (row.gone) (chosen.has(i) ? right : missed).push(i);
    else if (chosen.has(i)) wrong.push(i);
  });
  return { right, missed, wrong };
}
