"use client";

import { useState } from "react";
import kit from "@/app/(landing)/page.module.css";
import styles from "./Sections.module.css";
import { SIcon } from "./sectionIcons";

// Three quiet things: what the model sees, what an answer drew on, and
// what each helper got. Each lives in one quiet place in the app.

const SEES = [
  { l: "Instructions", s: "its built-in rules", n: 9400, k: "--c-instr" },
  { l: "Tools", s: "what it can use", n: 6800, k: "--c-tools" },
  { l: "Your notes", s: "notes.md", n: 2300, k: "--c-added" },
  { l: "Summary", s: "stands in for 6 turns", n: 3100, k: "--c-summary" },
  { l: "You", s: "your message", n: 600, k: "--c-you" },
  { l: "Its work", s: "the draft, rewritten", n: 14000, k: "--c-work" },
];

const HELPERS = [
  { got: "got 1.9k · back 420", note: "Find where the upload retry delay is set.", back: "The delay is ", code: "RETRY_MS = 200", after: " in ", code2: "src/retry.ts", end: "." },
  { got: "got 1.6k · back 880", note: "List every test that waits on a real timer.", back: "Three: upload, resume and the progress bar test." },
  { got: "got 1.4k · back 310", note: "Check the history of retry.ts.", back: "Last changed in March, from 100 to 200 ms." },
];

const fmt = (n: number) => (n >= 1000 ? `${Math.round(n / 100) / 10}k` : String(n));

function Ring({ dash }: { dash: string }) {
  return (
    <svg className={styles.ring} viewBox="0 0 16 16" aria-hidden="true">
      <circle className={styles.trk} cx="8" cy="8" r="6" />
      <circle className={styles.arc} cx="8" cy="8" r="6" strokeDasharray={dash} />
    </svg>
  );
}

export default function Quiet() {
  const [order, setOrder] = useState<"read" | "size">("read");
  const [hot, setHot] = useState<string | null>(null);
  const [open, setOpen] = useState<Set<number>>(new Set([0]));

  const max = Math.max(...SEES.map((x) => x.n));
  const rows = order === "size" ? [...SEES].sort((a, b) => b.n - a.n) : SEES;
  const mark = (src: string, text: string) => (
    <mark className={hot === src ? styles.hot : undefined}>{text}</mark>
  );
  const hl = (src: string) => ({
    className: `${kit.chip} ${hot === src ? kit.hot : ""}`.trim(),
    onPointerEnter: () => setHot(src),
    onPointerLeave: () => setHot(null),
    onFocus: () => setHot(src),
    onBlur: () => setHot(null),
    onClick: () => setHot(src),
  });

  return (
    <section className={kit.sec} id="quiet" aria-labelledby="quiet-title">
      <div className={kit.wrap}>
        <div className={`${kit.head} ${kit.headCenter}`}>
          <span className={kit.kicker}>Always there, out of the way</span>
          <h2 className={kit.h2} id="quiet-title">
            Everything else is one click away.
          </h2>
          <p className={kit.lede}>The chat stays clean. Each of these lives in one quiet place, and opens when you ask.</p>
        </div>
        <div className={styles.three}>
          <article className={styles.thing}>
            <div className={styles.art}>
              <div className={styles.seg} role="group" aria-label="Order">
                <button type="button" aria-pressed={order === "read"} onClick={() => setOrder("read")}>
                  In order
                </button>
                <button type="button" aria-pressed={order === "size"} onClick={() => setOrder("size")}>
                  By size
                </button>
              </div>
              <div className={styles.list}>
                {rows.map((r) => (
                  <div className={styles.li} key={r.l}>
                    <i
                      style={
                        {
                          "--k": `var(${r.k})`,
                          "--w": `${Math.max(3, Math.round((r.n / max) * 100))}%`,
                        } as React.CSSProperties
                      }
                    />
                    <span>
                      {r.l} <small>{r.s}</small>
                    </span>
                    <span className={kit.num}>{fmt(r.n)}</span>
                  </div>
                ))}
              </div>
            </div>
            <h3>What it sees</h3>
            <p>Everything the model reads, in plain words, in the order it reads it.</p>
            <span className={styles.where}>
              <Ring dash="7.9 37.7" />
              Click the ring in the composer
            </span>
          </article>
          <article className={styles.thing}>
            <div className={styles.art}>
              <div className={styles.doc}>
                <span className={styles.file}>
                  <SIcon name="file" />
                  notes.md
                </span>
                {mark("move", "Moving to the new office on the 14th.")} Two new hires: Ana (design) and Joel (support).{" "}
                {mark("hours", "Holiday hours: closed the 24th to the 26th.")}
              </div>
              <p className={styles.miniReply}>
                Here&apos;s the draft. It opens with the move on the 14th and ends with the holiday hours.
              </p>
              <div className={styles.receipt}>
                <span className={styles.drew}>
                  Drew on{" "}
                  <button type="button" {...hl("move")}>
                    <SIcon name="file" />
                    <b>notes.md</b>
                  </button>
                  <button type="button" {...hl("hours")}>
                    <SIcon name="quote" />
                    quotes <b>your notes</b>
                  </button>
                </span>
                <span className={styles.delta}>
                  <Ring dash="0.79 37.7" />
                  <span className={kit.num}>+4.2k</span>
                </span>
              </div>
            </div>
            <h3>What it drew on</h3>
            <p>
              One line under each answer: the files it opened and the words it quoted. Point at one to see it. It never
              guesses a percentage.
            </p>
            <span className={styles.where}>
              <SIcon name="file" />
              Under every reply
            </span>
          </article>
          <article className={styles.thing}>
            <div className={styles.art}>
              {HELPERS.map((h, i) => {
                const expanded = open.has(i);
                return (
                  <button
                    type="button"
                    className={styles.helper}
                    aria-expanded={expanded}
                    key={h.note}
                    onClick={() =>
                      setOpen((was) => {
                        const next = new Set(was);
                        if (!next.delete(i)) next.add(i);
                        return next;
                      })
                    }
                  >
                    <span className={styles.hTop}>
                      <b>Explore</b>
                      <small>{h.got}</small>
                    </span>
                    <span className={styles.hNote}>{h.note}</span>
                    <span className={styles.quote} hidden={!expanded}>
                      Came back: &quot;{h.back}
                      {h.code && <code>{h.code}</code>}
                      {h.after}
                      {h.code2 && <code>{h.code2}</code>}
                      {h.end}&quot;
                    </span>
                  </button>
                );
              })}
            </div>
            <h3>What each helper got</h3>
            <p>
              When Claude Code hands work to a helper, see the note it was given and what came back, counted word for
              word.
            </p>
            <span className={styles.where}>
              <SIcon name="agents" />
              The agents button, when a chat has helpers
            </span>
          </article>
        </div>
      </div>
    </section>
  );
}
