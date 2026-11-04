"use client";

import { type CSSProperties, type ReactNode, useState } from "react";
import kit from "@/app/(landing)/page.module.css";
import IllumenMark from "@/components/IllumenMark";
import Wordmark from "@/components/Wordmark";
import styles from "./AppDrawing.module.css";

// The small pieces the hero's app drawing is made of: its icons (strokes in
// the text colour, always decorative), the context ring, the room bar with
// its keys, a turn's work line, and the sidebar.

export const cx = (...names: (string | false | null | undefined)[]) => names.filter(Boolean).join(" ");

const GLYPHS = {
  plus: <path d="M8 3v10M3 8h10" />,
  up: <path d="M8 13V3.5M4 7.2 8 3.2l4 4" />,
  down: <path d="m4.5 6.2 3.5 3.5 3.5-3.5" />,
  right: <path d="m6.2 4.5 3.5 3.5-3.5 3.5" />,
  undo: (
    <>
      <path d="M5.2 3.8 2.5 6.5l2.7 2.7" />
      <path d="M2.5 6.5H10a3.2 3.2 0 0 1 0 6.4H7.5" />
    </>
  ),
  close: <path d="m4.5 4.5 7 7M11.5 4.5l-7 7" />,
  sidebar: (
    <>
      <rect x="2" y="3" width="12" height="10" rx="2" />
      <path d="M6.2 3v10" />
    </>
  ),
  compose: (
    <>
      <path d="M13 8.8V12a1.5 1.5 0 0 1-1.5 1.5h-7A1.5 1.5 0 0 1 3 12V4.5A1.5 1.5 0 0 1 4.5 3h3.2" />
      <path d="M11.4 2.6a1.35 1.35 0 0 1 1.9 1.9L8.6 9.2 6.4 9.8l.6-2.2z" />
    </>
  ),
  sliders: (
    <>
      <path d="M2.5 5h6.8M12.6 5h.9M2.5 11h1.1M6.8 11h6.7" />
      <circle cx="11" cy="5" r="1.6" />
      <circle cx="5.2" cy="11" r="1.6" />
    </>
  ),
  file: (
    <>
      <path d="M4.5 2h4.3L12 5.2v7.3a1.5 1.5 0 0 1-1.5 1.5h-6A1.5 1.5 0 0 1 3 12.5v-9A1.5 1.5 0 0 1 4.5 2z" />
      <path d="M8.6 2v3.4H12" />
    </>
  ),
  files: (
    <>
      <rect x="4.8" y="2.5" width="8.7" height="9" rx="1.6" />
      <path d="M2.5 5.2v6.8A1.5 1.5 0 0 0 4 13.5h6.8" />
    </>
  ),
  agents: (
    <>
      <circle cx="4" cy="3.6" r="1.5" />
      <circle cx="4" cy="12.4" r="1.5" />
      <circle cx="12" cy="8" r="1.5" />
      <path d="M4 5.1v5.8M4 6.4c0 1.2.8 1.6 2.2 1.6h4.3" />
    </>
  ),
  terminal: (
    <>
      <rect x="2" y="3" width="12" height="10" rx="2" />
      <path d="m5 6.8 1.9 1.4L5 9.6M8.4 10h2.6" />
    </>
  ),
  laptop: (
    <>
      <rect x="3.5" y="3.5" width="9" height="6.5" rx="1" />
      <path d="M2 12.5h12" />
    </>
  ),
  key: (
    <>
      <circle cx="5.4" cy="10.6" r="2.6" />
      <path d="M7.3 8.7 13 3M11 5l1.5 1.5M9.4 6.6l1 1" />
    </>
  ),
  eyeOff: (
    <path d="M6.2 3.8A6.6 6.6 0 0 1 8 3.5c4.1 0 6.5 4.5 6.5 4.5a11.6 11.6 0 0 1-1.6 2.1M9.8 12.2a6.6 6.6 0 0 1-1.8.3C3.9 12.5 1.5 8 1.5 8a11.6 11.6 0 0 1 2.4-2.9M2.5 2.5l11 11" />
  ),
  putback: <path d="M8 2.5v7.6M4.8 7 8 10.2 11.2 7M3 13.5h10" />,
  book: (
    <path d="M3 3.2A1.2 1.2 0 0 1 4.2 2H13v10H4.2A1.2 1.2 0 0 0 3 13.2zM3 13.2A1.2 1.2 0 0 0 4.2 14.4H13V12" />
  ),
} as const;

export type GlyphName = keyof typeof GLYPHS;

export function AppIcon({ name, small = false }: { name: GlyphName; small?: boolean }) {
  return (
    <svg className={cx(kit.ic, small && kit.icSm)} viewBox="0 0 16 16" aria-hidden="true">
      {GLYPHS[name]}
    </svg>
  );
}

/** How much of the window `pct` percent fills, as the ring's arc length. */
export const dashFor = (pct: number) => Number(((37.7 * pct) / 100).toFixed(1));

/** The context ring: a track and an arc `dash` long, out of 37.7. */
export function Ring({ dash, large = false }: { dash: number; large?: boolean }) {
  return (
    <svg className={cx(styles.ring, large && styles.ringLg)} viewBox="0 0 16 16" aria-hidden="true">
      <circle className={styles.trk} cx="8" cy="8" r="6" />
      <circle className={styles.arc} cx="8" cy="8" r="6" strokeDasharray={`${dash} 37.7`} />
    </svg>
  );
}

export function Mono({ children }: { children: ReactNode }) {
  return <span className={kit.mono}>{children}</span>;
}

// What takes the room now, in k: the island and the Context panel's
// Overview draw the same bar. They sum to the 42.6k it's full with.
const ROOM = [
  { label: "Its work", k: 22.8, hue: "var(--c-work)" },
  { label: "Instructions", k: 9.4, hue: "var(--c-instr)" },
  { label: "Tools", k: 6.8, hue: "var(--c-tools)" },
  { label: "Summary", k: 2.6, hue: "var(--c-summary)" },
  { label: "Added", k: 1, hue: "var(--c-added)" },
];

export function Room() {
  return (
    <>
      <div className={kit.room} aria-hidden="true">
        {ROOM.map((r) => (
          <i key={r.label} style={{ flex: r.k, background: r.hue }} />
        ))}
      </div>
      <div className={styles.keys}>
        {ROOM.map((r) => (
          <span key={r.label} className={kit.key} style={{ "--k": r.hue } as CSSProperties}>
            {r.label} {r.k}k
          </span>
        ))}
      </div>
    </>
  );
}

/** A turn's "Worked for …" line, which opens to its steps. */
export function Work({ summary, children }: { summary: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button type="button" className={styles.work} aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        <AppIcon name="right" small />
        {summary}
      </button>
      <div className={styles.steps} hidden={!open}>
        {children}
      </div>
    </>
  );
}

const elsewhere = (chat: string) => `Only Flaky upload test is drawn here. In the app, this opens ${chat}.`;

/** The app's sidebar: its chats and Settings. Only one chat is drawn, so
 * each row says what it would do. */
export function AppSidebar({ say }: { say: (message: string) => void }) {
  const row = (icon: GlyphName, label: string, toast: string, on = false) => (
    <button
      type="button"
      className={cx(styles.row, on && styles.on)}
      aria-current={on || undefined}
      onClick={() => say(toast)}
    >
      <AppIcon name={icon} />
      {label}
    </button>
  );
  return (
    <div className={styles.side} role="group" aria-label="Chats">
      <div className={styles.sideHead}>
        <IllumenMark size={20} />
        <Wordmark className={styles.sideWord} />
        <button
          type="button"
          className={cx(kit.btn, kit.btnIcon, styles.sideToggle)}
          aria-label="Hide the sidebar"
          onClick={() => say("Hides the sidebar.")}
        >
          <AppIcon name="sidebar" />
        </button>
      </div>
      {row("compose", "New chat", "Opens a new chat, and you pick who answers.")}
      <div className={cx(kit.label, styles.sideLabel)}>Today</div>
      {row("terminal", "Flaky upload test", "This is the chat you're in.", true)}
      {row("key", "Pricing questions", elsewhere("Pricing questions"))}
      <div className={cx(kit.label, styles.sideLabel)}>Yesterday</div>
      {row("key", "October newsletter", elsewhere("October newsletter"))}
      {row("laptop", "Notes from the offsite", elsewhere("Notes from the offsite"))}
      <div className={styles.sideFoot}>
        {row("sliders", "Settings", "Settings: your keys, your CLIs and how illumen looks.")}
      </div>
    </div>
  );
}
