import { GONE, type GoneId } from "./appChat";

// The hero's app plays its last moments once it's on screen. Claude Code
// summarises the chat, and two things you typed fade. You ask for the fix,
// and it makes the one you told it not to: it changes the retry logic.
// illumen shows what the summary dropped and puts it back, you say "Undo
// that.", and this time it fixes the test. The rhythm: 55 ms a letter,
// 900 ms a work step, 60 ms a word.
export type Phase =
  | "pre"
  | "summarising"
  | "summarised"
  // Turn A: "Now make the fix.", and the mistake.
  | "typingA"
  | "sentA"
  | "workingA"
  | "replyingA"
  | "repliedA"
  // illumen opens the seam and marks what was dropped, then puts it back.
  | "revealing"
  | "puttingBack"
  // Turn B: "Undo that.", with what was put back, and the right fix.
  | "typingB"
  | "sentB"
  | "workingB"
  | "replyingB"
  | "done";

export const TYPED = "Now make the fix.";
export const TYPED_REPLY = "Undo that.";
/** The gone message the play puts back. */
export const PUT_BACK: GoneId = "retry";

export type Turn = {
  /** What you type, and what you put back with it. */
  typed: string;
  putBack: GoneId | null;
  /** The steps shown live while it works, then its work line and what that
   * line opens to. */
  steps: string[];
  work: string;
  did: string[];
  reply: string;
  /** Its receipt: what it drew on, what the turn added in k, and how full
   * it is after it, in %. */
  drew: { file: string; again?: true }[];
  delta: number;
  pct: number;
  /** The phases it's typed, sent, worked on, replied in and done by. */
  at: { typing: Phase; sent: Phase; working: Phase; replying: Phase; done: Phase };
};

// The two turns after the summary. Each delta is what the ring rose by:
// 19.8k + 6.1k = 25.9k (13%), + 16.7k = 42.6k (21%) of 200k.
export const TURNS: Turn[] = [
  {
    typed: TYPED,
    putBack: null,
    steps: ["Reading src/retry.ts", "Editing src/retry.ts", "Running the upload test"],
    work: "Worked for 31s · read 1 file, edited 1, ran the test once",
    did: ["Read src/retry.ts", "Edited src/retry.ts", "Ran npm test -- upload once · passed"],
    reply:
      "Done. I cut the retry delay in src/retry.ts from 200 ms to 50 ms, so the retry lands before the assertion. The upload test passes.",
    drew: [{ file: "retry.ts" }],
    delta: 6.1,
    pct: 13,
    at: { typing: "typingA", sent: "sentA", working: "workingA", replying: "replyingA", done: "repliedA" },
  },
  {
    typed: TYPED_REPLY,
    putBack: PUT_BACK,
    steps: [
      "Reverting src/retry.ts",
      "Reading test/helpers/clock.ts",
      "Editing test/upload.test.ts",
      "Running the test twice",
    ],
    work: "Worked for 48s · reverted 1 file, read 2, edited 1, ran the test twice",
    did: [
      "Reverted src/retry.ts",
      "Read test/upload.test.ts again",
      "Read test/helpers/clock.ts",
      "Edited test/upload.test.ts",
      "Ran npm test -- upload twice · passed",
    ],
    reply:
      "Sorry, I've put src/retry.ts back as it was. The fix is in the test instead: upload.test.ts now moves a fake clock past the retry delay, so it no longer depends on how fast the machine is.",
    drew: [{ file: "upload.test.ts", again: true }, { file: "clock.ts" }],
    delta: 16.7,
    pct: 21,
    at: { typing: "typingB", sent: "sentB", working: "workingB", replying: "replyingB", done: "done" },
  },
];

// How long each phase holds before the next. A phase with `per` holds `ms`
// for each letter it types, step it works or word it replies, then `then`.
export const APP_TIMELINE: { phase: Phase; ms: number; per?: "letter" | "step" | "word"; then?: number }[] = [
  { phase: "pre", ms: 900 },
  { phase: "summarising", ms: 1900 },
  { phase: "summarised", ms: 1700 },
  { phase: "typingA", ms: 55, per: "letter", then: 350 },
  { phase: "sentA", ms: 600 },
  { phase: "workingA", ms: 900, per: "step" },
  { phase: "replyingA", ms: 60, per: "word", then: 300 },
  // A beat on the mistake before illumen shows what caused it.
  { phase: "repliedA", ms: 900 },
  // The first gone message stays marked this long.
  { phase: "revealing", ms: 1200 },
  { phase: "puttingBack", ms: 900 },
  { phase: "typingB", ms: 55, per: "letter", then: 350 },
  { phase: "sentB", ms: 600 },
  { phase: "workingB", ms: 900, per: "step" },
  { phase: "replyingB", ms: 60, per: "word", then: 300 },
  { phase: "done", ms: 0 },
];

/** How long a button is drawn pressed: send, and Put back. */
export const PRESS_MS = 180;

/** One moment of the play: what changes then. */
export type Beat = {
  at: number;
  phase?: Phase;
  /** How many letters of the draft are typed. */
  typed?: number;
  /** Which of the live work steps shows. */
  step?: number;
  /** How many words of the reply are lit. */
  words?: number;
  /** The button drawn pressed, or none. */
  press?: "send" | "putBack" | null;
  /** The seam opens and lights. */
  reveal?: true;
};

/** Whether the play is at `at` or past it. */
export function reached(phase: Phase, at: Phase): boolean {
  const order = APP_TIMELINE.map((s) => s.phase);
  return order.indexOf(phase) >= order.indexOf(at);
}

const turnIn = (phase: Phase) => TURNS.find((t) => Object.values(t.at).includes(phase));

/** The whole play, beat by beat, timed from its start. */
export function appScript(): Beat[] {
  const beats: Beat[] = [];
  let at = 0;
  for (const { phase, ms, per, then = 0 } of APP_TIMELINE) {
    const turn = turnIn(phase);
    if (per && turn) {
      const n =
        per === "letter" ? turn.typed.length : per === "step" ? turn.steps.length : replyWords(turn.reply);
      for (let i = 0; i < n; i++) {
        const beat: Beat = { at: at + i * ms };
        if (i === 0) beat.phase = phase;
        if (per === "letter") beat.typed = i + 1;
        else if (per === "step") beat.step = i;
        else beat.words = i + 1;
        beats.push(beat);
      }
      at += ms * n + then;
      continue;
    }
    const beat: Beat = { at, phase };
    // Sending clears the draft.
    if (turn?.at.sent === phase) {
      beat.typed = 0;
      beat.press = "send";
    }
    if (phase === "revealing") beat.reveal = true;
    if (phase === "puttingBack") beat.press = "putBack";
    beats.push(beat);
    if (beat.press) beats.push({ at: at + PRESS_MS, press: null });
    at += ms;
  }
  return beats;
}

/** Where a turn is: not sent yet, sent, working, replying or done. */
export function turnStage(phase: Phase, i: number): "hidden" | "sent" | "working" | "replying" | "done" {
  const { sent, working, replying, done } = TURNS[i].at;
  if (reached(phase, done)) return "done";
  if (phase === replying) return "replying";
  if (phase === working) return "working";
  if (phase === sent) return "sent";
  return "hidden";
}

/** What the composer shows typed: the first `typed` letters of the message
 * being typed now, if one is. */
export function draftAt(phase: Phase, typed: number): string {
  const turn = TURNS.find((t) => t.at.typing === phase);
  return turn ? turn.typed.slice(0, typed) : "";
}

/** Where the play has a gone message: still gone, put back in the composer,
 * or sent back in with a message. */
export function playedBack(phase: Phase, id: GoneId): "gone" | "queued" | "sent" {
  const turn = TURNS.find((t) => t.putBack === id);
  if (!turn) return "gone";
  if (reached(phase, turn.at.sent)) return "sent";
  return reached(phase, "puttingBack") ? "queued" : "gone";
}

/** How full the ring says it is, and how many things you typed are gone. */
export function ringFor(phase: Phase): { pct: number; gone: number } {
  if (!reached(phase, "summarised")) return { pct: 86, gone: 0 };
  const done = TURNS.filter((t) => reached(phase, t.at.done));
  const back = TURNS.filter((t) => t.putBack && reached(phase, t.at.sent));
  return { pct: done.length ? done[done.length - 1].pct : 10, gone: GONE.length - back.length };
}

export const parts = (text: string) => text.split(/(\s+)/).filter(Boolean);
export const isSpace = (part: string) => /^\s+$/.test(part);
export const wordCount = (text: string) => parts(text).filter((p) => !isSpace(p)).length;

// A file's name (src/retry.ts, upload.test.ts) or the test command.
const CODE = /((?:[\w-]+\/)*[\w-]+(?:\.[\w-]+)*\.ts\b|npm test -- upload)/;

/** A line in pieces, with its file names and the test command set apart,
 * for the drawing to set as code. */
export function spans(text: string): { text: string; code: boolean }[] {
  return text
    .split(CODE)
    .map((piece, i) => ({ text: piece, code: i % 2 === 1 }))
    .filter((piece) => piece.text);
}

/** How many words a reply lights, one by one, its code among them. */
export const replyWords = (reply: string) => spans(reply).reduce((n, piece) => n + wordCount(piece.text), 0);
