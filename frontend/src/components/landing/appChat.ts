// The hero's app drawing: one made-up chat, "Flaky upload test", after
// Claude Code summarised it. The numbers here are the ones the drawing shows.
export const GONE = [
  { id: "retry", text: "Don't touch the retry logic. The team wants it kept as it is." },
  { id: "clock", text: "Run the full suite ten times before you call it fixed." },
] as const;

export type GoneId = (typeof GONE)[number]["id"];

/** "1 thing you typed is gone", "2 things you typed are gone". */
export const goneWords = (n: number) =>
  n === 1 ? "1 thing you typed is gone" : `${n} things you typed are gone`;

export const FINAL = { pct: 21, used: "42.6k", room: "200k", from: "171k", to: "19.8k" } as const;

// What the newest turn added, row by row, in k; the rows sum to the +16.7k
// it shows. Your message and what you put back are 22 tokens.
export const TURN_ADDED = [
  { label: "Ran the test twice", k: 8.7 },
  { label: "Read clock.ts", k: 3.4 },
  { label: "Read upload.test.ts again", k: 2.8 },
  { label: "Its reply", k: 1.8 },
  { label: "Your message and what you put back", k: 0.022 },
] as const;

/** A size in k as the drawing prints it: "8.7k", or under 0.1k the count
 * itself, "22". */
export const kText = (k: number) => (k < 0.1 ? String(Math.round(k * 1000)) : `${k}k`);
