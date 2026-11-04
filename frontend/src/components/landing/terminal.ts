// The hero's second view, Claude Code with the illumen plugin, drawn: a
// made-up session in ~/invoice-service that summarises itself, after which the
// plugin speaks up, and a prompt that answers the plugin's two commands. The
// pure parts: what the session shows after each step, what the status line
// and the pane say, which commands match and what the prompt answers. Nothing
// here runs Claude Code or asks the network; every number is a sample.
import { tokensText } from "@/lib/landing";

/** The session's model room, in tokens. */
export const ROOM = 200_000;

/** How long each letter takes when the script types, in ms. */
export const TYPE_MS = 34;

/** One model request: what the cache held and served again, and what the
 * model had to read again. */
export type Request = { reused: number; again: number };

export type Line =
  | { kind: "user"; text: string }
  | { kind: "busy"; text: string }
  | { kind: "done" }
  | { kind: "tool"; text: string; result?: string }
  | { kind: "reply"; text: string }
  /** Claude Code's permission prompt. `choice` is the option lit (1 or 2), or
   * "no" once it has collapsed to the line saying what was chosen. */
  | { kind: "ask"; command: string; about: string; choice: 1 | 2 | "no" }
  /** /illumen-forgot's output, drawn inline. */
  | { kind: "forgot" }
  /** A command's one-line answer, printed plain. */
  | { kind: "text"; text: string }
  /** The grey line Claude Code leaves when a turn is done: how long it took. */
  | { kind: "worked"; text: string };

export type Step = {
  /** How long before it lands, in ms. */
  wait: number;
  /** Typed into the prompt first, a letter at a time; then it is sent. */
  type?: string;
  line?: Line;
  /** Moves the permission prompt's choice (2), or collapses it ("no"). */
  choose?: 2 | "no";
  /** Sent from the prompt, which holds this text while the step waits. */
  send?: string;
  request?: Request;
  /** The context's size after the summary. */
  compacted?: number;
  /** The plugin's note shows from here. */
  toast?: boolean;
  /** /illumen-forgot has been viewed from here. */
  forgot?: boolean;
};

/** The requests before the session on screen: a long chat, nearly full. */
export const EARLIER: Request[] = [
  { reused: 141_000, again: 4_200 },
  { reused: 148_000, again: 3_100 },
  { reused: 154_500, again: 5_800 },
  { reused: 163_000, again: 2_400 },
];

/** The two things you typed that the summary dropped. */
export const LEFT_OUT = [
  "Never run the migrations against the shared staging database. Use the local docker one.",
  "Keep the v2 response shape exactly as it is. The mobile app still pins v2.",
] as const;

export const FORGOT = "/illumen-forgot";

/** What the prompt prints when the first gone message is put back. */
const RESTORE_SAID = "Put it back in your prompt. Send it when you're ready.";

export const SCRIPT: Step[] = [
  { wait: 600, type: "/compact" },
  { wait: 500, line: { kind: "busy", text: "Compacting conversation…" } },
  { wait: 1700, line: { kind: "done" }, compacted: 26_000, toast: true },
  { wait: 1500, type: "run the migrations so I can test the new column" },
  {
    wait: 1100,
    line: { kind: "tool", text: "Bash(npm run migrate -- --env staging)" },
    request: { reused: 26_000, again: 1_200 },
  },
  {
    wait: 700,
    line: {
      kind: "ask",
      command: "npm run migrate -- --env staging",
      about: "Run the database migrations on staging",
      choice: 1,
    },
  },
  { wait: 1200, choose: 2 },
  { wait: 700, choose: "no" },
  { wait: 1400, type: FORGOT },
  { wait: 300, line: { kind: "forgot" }, forgot: true },
  { wait: 2200, type: `${FORGOT} restore 1` },
  { wait: 300, line: { kind: "text", text: RESTORE_SAID } },
  { wait: 900, send: LEFT_OUT[0] },
  {
    wait: 1000,
    line: { kind: "reply", text: "Understood. I'll use the local docker database." },
    request: { reused: 27_200, again: 1_500 },
  },
  {
    wait: 1100,
    line: { kind: "tool", text: "Bash(npm run migrate -- --env local)", result: "Applied 3 migrations" },
    request: { reused: 28_700, again: 2_100 },
  },
  {
    wait: 1300,
    line: { kind: "reply", text: "Migrations applied to the local docker database. The new column is there." },
    request: { reused: 30_800, again: 3_600 },
  },
  { wait: 700, line: { kind: "worked", text: "Worked for 14s" } },
];

/** The summary: from how many tokens to how many, and when. */
export const SUMMARY_FROM = 165_400;
export const SUMMARY_TO = 26_000;
export const SUMMARISED_AT = "11:24";

/** The plugin's commands, as it registers them. */
export const COMMANDS = [
  { name: "/illumen", about: "Open illumen's pane: how full it is, and what it forgot" },
  { name: FORGOT, about: "What the last summary dropped of what you typed, and a way to put it back" },
] as const;

const total = (r: Request) => r.reused + r.again;
const pct = (n: number) => Math.round(n * 100);

/** A token count as the status line spells it: 999, 26k, 165k. */
export const compactTokens = (n: number) => (n >= 1000 ? `${Math.round(n / 1000)}k` : String(Math.round(n)));

const SUMMARY = `${compactTokens(SUMMARY_FROM)} → ${compactTokens(SUMMARY_TO)}`;

export type Session = {
  requests: Request[];
  /** Each keyed by its step, so a line that follows another (Compacted after
   * Compacting) comes in as a line of its own. */
  lines: { key: string; line: Line }[];
  tokens: number;
  fill: number;
  /** Which request came first after the summary, once there was one. */
  compactedAt: number | null;
  toast: boolean;
  /** Whether /illumen-forgot has been viewed by the script. */
  forgotSeen: boolean;
  /** The gone messages the script put back and sent, by index. */
  restored: number[];
  /** Whether a request has reported how full it is since the summary. */
  measured: boolean;
};

/** What the session shows once `n` steps of the script have landed. */
export function sessionAt(n: number): Session {
  const requests = [...EARLIER];
  const lines: Session["lines"] = [];
  let tokens = total(EARLIER[EARLIER.length - 1]);
  let compactedAt: number | null = null;
  let toast = false;
  let forgotSeen = false;
  const restored: number[] = [];
  let measured = true;
  SCRIPT.slice(0, n).forEach((st, i) => {
    if (st.type !== undefined) lines.push({ key: `${i}t`, line: { kind: "user", text: st.type } });
    if (st.send !== undefined) {
      lines.push({ key: `${i}s`, line: { kind: "user", text: st.send } });
      // Sending one of the gone messages puts it back for good: Claude has it again.
      const back = LEFT_OUT.findIndex((t) => t === st.send);
      if (back >= 0) restored.push(back);
    }
    if (st.line) lines.push({ key: `${i}l`, line: st.line });
    if (st.choose !== undefined) {
      const at = lines.map((l) => l.line.kind).lastIndexOf("ask");
      const ask = lines[at]?.line;
      if (ask?.kind === "ask") lines[at] = { key: lines[at].key, line: { ...ask, choice: st.choose } };
    }
    if (st.forgot) forgotSeen = true;
    if (st.compacted !== undefined) {
      tokens = st.compacted;
      compactedAt = requests.length;
      measured = false;
    }
    if (st.request) {
      requests.push(st.request);
      tokens = total(st.request);
      measured = true;
    }
    if (st.toast) toast = true;
  });
  return { requests, lines, tokens, fill: tokens / ROOM, compactedAt, toast, forgotSeen, restored, measured };
}

/** The plugin's line under the prompt: how full, once measured, and "2 gone"
 * until you've looked (/illumen-forgot) or put both back. */
export function statusText(s: Session, { forgotSeen, putBack }: { forgotSeen: boolean; putBack: number }): string {
  const parts: string[] = [];
  if (s.measured) parts.push(`${pct(s.fill)}% full`);
  if (s.compactedAt !== null && !forgotSeen && !s.forgotSeen && putBack < LEFT_OUT.length) parts.push("2 gone", FORGOT);
  return parts.length ? "⚠ illumen: " + parts.join(" · ") : "";
}

/** Whether Claude Code lists commands above the prompt: once you type "/",
 * and only until a space. */
export function showsMenu(ask: string): boolean {
  const said = ask.trim();
  return said.startsWith("/") && !/\s/.test(said);
}

/** The commands the menu lists for what's typed, the one Enter runs first. */
export function matches(ask: string): { name: string; about: string }[] {
  if (!showsMenu(ask)) return [];
  const name = ask.trim();
  return COMMANDS.filter((c) => c.name.startsWith(name));
}

/** What Enter runs: a part of a command's name runs the command the menu
 * lights (the first, unless `highlighted` moved it); anything else as it's typed. */
export function resolve(ask: string, highlighted = 0): string {
  const said = ask.trim();
  const name = said.split(/\s+/)[0];
  if (!said.startsWith("/") || COMMANDS.some((c) => c.name === name)) return said;
  const list = COMMANDS.filter((c) => c.name.startsWith(name));
  const pick = showsMenu(said) ? list[highlighted] : undefined;
  const first = pick ?? list[0];
  return first ? first.name + said.slice(name.length) : said;
}

/** What Tab types: the highlighted match's name, or null when there's nothing
 * to complete (no match, or the prompt already says it), so Tab moves focus. */
export function complete(ask: string, highlighted: number): string | null {
  const list = matches(ask);
  if (!list.length) return null;
  const name = (list[highlighted] ?? list[0]).name;
  return name === ask ? null : name;
}

/** The highlight after an arrow key: one step through `n` matches, wrapping. */
export function moveHighlight(i: number, n: number, dir: 1 | -1): number {
  return n > 0 ? (i + dir + n) % n : 0;
}

export type Answer = {
  /** "forgot" draws /illumen-forgot's receipt; "text" prints `text`. */
  kind: "forgot" | "text";
  text?: string;
  /** What goes into the prompt, to send when you're ready. */
  fill?: string[];
  /** Which gone messages are in your prompt after this, by index. */
  back?: number[];
  /** What happens to the plugin's pane. */
  pane?: "open" | "close";
  /** Whether it was /illumen-forgot: the note goes, the status line drops "2 gone". */
  forgot?: boolean;
};

const USAGE = `Usage: ${FORGOT}, or ${FORGOT} restore [n…]`;

/** What the prompt answers. No model runs here, so anything but the plugin's
 * commands gets an honest answer, never a made-up one. */
export function answer(said: string, opts: { putBack: number[]; paneOpen?: boolean }): Answer {
  const [name, ...args] = said.trim().split(/\s+/);
  if (name === "/illumen") {
    return opts.paneOpen
      ? { kind: "text", text: "Closed the illumen pane. /illumen opens it again.", pane: "close" }
      : { kind: "text", text: "Opened the illumen pane. /illumen or Esc closes it.", pane: "open" };
  }
  if (name === FORGOT) {
    if (args[0] === "restore") {
      const all = LEFT_OUT.map((_, i) => i);
      const asked = args.slice(1).map((a) => parseInt(a, 10) - 1);
      const pick = args.length > 1 ? all.filter((i) => asked.includes(i)) : all;
      if (!pick.length) return { kind: "text", text: USAGE, forgot: true };
      return {
        kind: "text",
        text:
          pick.length === LEFT_OUT.length
            ? "Put both back in your prompt. Send them when you're ready."
            : RESTORE_SAID,
        fill: pick.map((i) => LEFT_OUT[i]),
        back: all.filter((i) => pick.includes(i) || opts.putBack.includes(i)),
        forgot: true,
      };
    }
    if (args.length) return { kind: "text", text: USAGE, forgot: true };
    return { kind: "forgot", forgot: true };
  }
  if (name.startsWith("/"))
    return { kind: "text", text: `Unknown slash command: ${name}. This drawing answers /illumen and ${FORGOT}.` };
  return { kind: "text", text: "This is a drawing, so no model answers here. Type / for the plugin's commands." };
}

/** What the seam row and the receipt say of yours is gone. */
export const GONE_WORDS = "2 things you typed are gone";

/** What the seam and /illumen-forgot say of one the session put back. */
export const BACK_WORDS = "back in what it reads";

/** /illumen-forgot's output, part by part: the drawing styles each. */
export const FORGOT_OUT = {
  before: `The summary at ${SUMMARISED_AT} left out `,
  gone: "2 things you told Claude",
  after: ":",
  caveat: "Claude can't see these any more. They weren't found word for word in the summary.",
  inPrompt: "  (in your prompt)",
  /** After an item the script put back and sent: Claude reads it again. */
  restored: ` (${BACK_WORDS})`,
} as const;

const ORDINALS = ["first", "second"] as const;

/** The line naming the command that puts an item back: for the first one the
 * script hasn't already restored, or null when none is left. */
export function restoreHint(restored: readonly number[] = []): string | null {
  const i = LEFT_OUT.findIndex((_, n) => !restored.includes(n));
  return i < 0 ? null : `${FORGOT} restore ${i + 1}    puts the ${ORDINALS[i]} one back in your prompt`;
}

/** The line under the items: what Claude still can't see, once some are back. */
export function forgotCaveat(restored: readonly number[] = []): string {
  const left = LEFT_OUT.flatMap((_, n) => (restored.includes(n) ? [] : [n]));
  if (left.length === LEFT_OUT.length) return FORGOT_OUT.caveat;
  if (left.length === 0) return "Claude reads both again.";
  return `Claude still can't see the ${ORDINALS[left[0]]} one. It wasn't found word for word in the summary.`;
}

/** /illumen-forgot's output as lines of text; `putBack` marks the ones in
 * your prompt, `restored` the ones the session sent and Claude reads again. */
export function forgotLines(putBack: readonly number[] = [], restored: readonly number[] = []): string[] {
  const hint = restoreHint(restored);
  return [
    FORGOT_OUT.before + FORGOT_OUT.gone + FORGOT_OUT.after,
    "",
    ...LEFT_OUT.map(
      (text, i) =>
        `${i + 1}  ${text}${restored.includes(i) ? FORGOT_OUT.restored : putBack.includes(i) ? FORGOT_OUT.inPrompt : ""}`,
    ),
    "",
    forgotCaveat(restored),
    ...(hint ? [hint] : []),
  ];
}

/** What a screen reader hears for the receipt: the same lines, read through. */
export function forgotSaid(putBack: readonly number[] = [], restored: readonly number[] = []): string {
  return forgotLines(putBack, restored)
    .filter(Boolean)
    .map((l) => l.replace(/\s{2,}/g, " "))
    .join(" ");
}

/** The seam row's words before its "2 things you typed are gone". */
export const SEAM_TEXT = `─ Claude Code summarised the chat · ${SUMMARY} · `;

/** The note the plugin shows once the summary lands, before its link. */
export const NOTE_TEXT = "It summarised the chat. 2 things you typed are gone · ";

// ---------- the pane ----------

/** How many characters wide the pane's bars are. */
export const PANE_CELLS = 34;
const CUT = Math.round(0.92 * PANE_CELLS);

/** How full, as the pane says it. */
export function paneFull(s: Session): { big: string; small: string } {
  return s.measured
    ? { big: `${pct(s.fill)}% full`, small: `${tokensText(s.tokens)} of ${tokensText(ROOM)}` }
    : { big: "Summarised", small: "how full it is shows after the next reply" };
}

/** The pane's bar of block characters, lit to how full it is, with ┊ where
 * Claude Code summarises: [the lit part, the rest]. */
export function fullBar(s: Session): [string, string] {
  const filled = s.measured ? Math.round(s.fill * PANE_CELLS) : 0;
  let bar = "";
  for (let i = 0; i < PANE_CELLS; i++) bar += i === CUT ? "┊" : i < filled ? "█" : "░";
  return [bar.slice(0, filled), bar.slice(filled)];
}

/** Under the bar once it has summarised: a ╵ where it stood before, and what that means. */
export const PEAK = {
  mark: " ".repeat(Math.round((SUMMARY_FROM / ROOM) * PANE_CELLS)) + "╵",
  note: `${pct(SUMMARY_FROM / ROOM)}% before the summary · ┊ where it summarises`,
} as const;

/** What takes the room, each part with its terminal colour and its share of the bar. */
export function roomParts(s: Session): { label: string; tokens: number; hue: string; cells: number }[] {
  const parts = (
    [
      ["Instructions", 9_400, "--tc-instr"],
      ["Tools", 6_800, "--tc-tools"],
      ["Added", 2_400, "--tc-added"],
      ["Summary", 7_400, "--tc-summary"],
      ["Its work", Math.max(0, s.tokens - SUMMARY_TO), "--tc-work"],
    ] as const
  ).filter((p) => p[1] > 0);
  const sum = parts.reduce((n, p) => n + p[1], 0);
  let used = 0;
  return parts.map(([label, tokens, hue], i) => {
    const cells = i === parts.length - 1 ? PANE_CELLS - used : Math.round((tokens / sum) * PANE_CELLS);
    used += cells;
    return { label, tokens, hue: `var(${hue})`, cells: Math.max(0, cells) };
  });
}

/** How full it is once the session ends, in percent: the try row's ring. */
export const END_PCT = pct(sessionAt(SCRIPT.length).fill);
