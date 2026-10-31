"use client";

import { type CSSProperties, type KeyboardEvent, type ReactNode, useRef, useState } from "react";
import kit from "@/app/(landing)/page.module.css";
import { FINAL, GONE, type GoneId, kText, TURN_ADDED } from "./appChat";
import { AppIcon, cx, Room } from "./AppParts";
import { TURNS, TYPED_REPLY } from "./appPlay";
import styles from "./AppDrawing.module.css";

export type AsideTab = "overview" | "sees" | "forgotten" | "agents";

const TABS: { id: AsideTab; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "sees", label: "What it sees" },
  { id: "forgotten", label: "Forgotten" },
  { id: "agents", label: "Agents" },
];

// Your newest message, with what you put back, as the newest turn's peek
// counts it: 22 tokens, under the rounding of its work, the two turns'
// 22.8k. With the 19.8k the summary left, they make the 42.6k it's full with.
const YOURS = kText(TURN_ADDED[TURN_ADDED.length - 1].k);
const WORK = TURNS.reduce((n, t) => n + t.delta, 0).toFixed(1);

// What Claude Code reads now, top to bottom, with the share of the widest.
const SEES = [
  { what: "Instructions", about: "its built-in rules", hue: "var(--c-instr)", w: "41%", n: "9.4k" },
  { what: "Tools", about: "what it can use", hue: "var(--c-tools)", w: "30%", n: "6.8k" },
  { what: "Added by Claude Code", about: "CLAUDE.md", hue: "var(--c-added)", w: "5%", n: "1k" },
  { what: "Summary", about: "stands in for 3 turns", hue: "var(--c-summary)", w: "11%", n: "2.6k" },
  { what: "You", about: `"${TYPED_REPLY}" and what you put back`, hue: "var(--c-you)", w: "1%", n: YOURS },
  { what: "Its work", about: "3 files, 3 test runs, its replies", hue: "var(--c-work)", w: "100%", n: `${WORK}k` },
];

function Helper({ got, note, quote }: { got: string; note: string; quote: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <button type="button" className={styles.helper} aria-expanded={open} onClick={() => setOpen((o) => !o)}>
      <span className={styles.hTop}>
        <b>Explore</b>
        <small>{got}</small>
      </span>
      <span className={styles.hNote}>{note}</span>
      <span className={styles.quote} hidden={!open}>
        {quote}
      </span>
    </button>
  );
}

/** The Context panel beside the chat: four tabs (arrow keys move between
 * them), each a view of what Claude Code reads. */
export default function AppAside({
  open,
  forgotten,
  tab,
  onTab,
  onClose,
}: {
  open: boolean;
  /** What you typed that's still not in what it reads. */
  forgotten: GoneId[];
  tab: AsideTab;
  onTab: (tab: AsideTab) => void;
  onClose: () => void;
}) {
  const tabs = useRef<Partial<Record<AsideTab, HTMLButtonElement | null>>>({});

  function onKey(e: KeyboardEvent<HTMLButtonElement>, i: number) {
    const n = TABS.length;
    const to = { ArrowRight: (i + 1) % n, ArrowLeft: (i + n - 1) % n, Home: 0, End: n - 1 }[e.key];
    if (to === undefined) return;
    e.preventDefault();
    const next = TABS[to].id;
    onTab(next);
    tabs.current[next]?.focus();
  }

  const panel = (id: AsideTab) => ({
    className: styles.stBody,
    id: `app-cpanel-${id}`,
    role: "tabpanel",
    "aria-labelledby": `app-ctab-${id}`,
    hidden: tab !== id,
  });

  return (
    <div className={styles.aside} id="app-aside" role="group" aria-label="Context panel" hidden={!open}>
      <div className={styles.asideIn}>
        <div className={styles.stTop}>
          <b className={styles.stTitle}>Context</b>
          <span className={styles.stMeta}>Claude Code · Opus 5.5</span>
          <button
            type="button"
            className={cx(kit.btn, kit.btnIcon, styles.stClose)}
            aria-label="Close Context"
            onClick={onClose}
          >
            <AppIcon name="close" />
          </button>
        </div>
        <div className={styles.tabs} role="tablist" aria-label="Context sections">
          {TABS.map((t, i) => (
            <button
              key={t.id}
              ref={(el) => {
                tabs.current[t.id] = el;
              }}
              type="button"
              role="tab"
              id={`app-ctab-${t.id}`}
              aria-controls={`app-cpanel-${t.id}`}
              aria-selected={tab === t.id}
              tabIndex={tab === t.id ? 0 : -1}
              onClick={() => onTab(t.id)}
              onKeyDown={(e) => onKey(e, i)}
            >
              {t.label}
            </button>
          ))}
        </div>

        <div {...panel("overview")}>
          <div className={styles.big}>
            <b className={kit.num}>{FINAL.pct}%</b>
            <span>
              full · {FINAL.used} of {FINAL.room}
            </span>
          </div>
          <div className={kit.meter} aria-hidden="true">
            <i style={{ width: `${FINAL.pct}%` }} />
          </div>
          <p className={styles.smallPrint}>As Anthropic reported it for the last request.</p>
          <Room />
          <div className={styles.callout}>
            <span>
              3:24 pm · summarised{" "}
              <span className={kit.num}>
                {FINAL.from} → {FINAL.to}
              </span>
              {forgotten.length > 0 && (
                <>
                  {" "}
                  · <span className={kit.goneN}>{forgotten.length} gone</span>
                </>
              )}
            </span>
          </div>
        </div>

        <div {...panel("sees")}>
          <span className={kit.label}>What it reads, top to bottom</span>
          <div className={styles.list}>
            {SEES.map((row) => (
              <div key={row.what} className={styles.li}>
                <i style={{ "--k": row.hue, "--w": row.w } as CSSProperties} />
                <span>
                  {row.what} <small>{row.about}</small>
                </span>
                <span className={kit.num}>{row.n}</span>
              </div>
            ))}
          </div>
        </div>

        <div {...panel("forgotten")}>
          <span className={kit.label}>Typed by you, not in what it reads now</span>
          {GONE.filter((g) => forgotten.includes(g.id)).map((g) => (
            <div key={g.id} className={styles.goneItem}>
              <q>{g.text}</q>
            </div>
          ))}
          <p className={styles.smallPrint}>Your first message made it into the summary.</p>
        </div>

        <div {...panel("agents")}>
          <span className={kit.label}>3 helpers · what each was handed and sent back</span>
          <Helper
            got="got 1.9k · back 420"
            note="Find where the upload retry delay is set."
            quote={
              <>
                Came back: &quot;The delay is <code>RETRY_MS = 200</code> in <code>src/retry.ts</code>. Nothing else
                sets it.&quot;
              </>
            }
          />
          <Helper
            got="got 1.6k · back 880"
            note="List every test that waits on a real timer."
            quote='Came back: "Three: upload, resume and the progress bar test. Only upload waits on the retry."'
          />
          <Helper
            got="got 1.4k · back 310"
            note="Check the history of retry.ts."
            quote='Came back: "Last changed in March, when the delay went from 100 to 200 ms."'
          />
          <p className={styles.smallPrint}>
            Counted from each note, word for word. Most helpers see only their note, not your chat.
          </p>
        </div>
      </div>
    </div>
  );
}
