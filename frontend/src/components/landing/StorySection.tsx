"use client";

import { type CSSProperties, useEffect, useLayoutEffect, useRef, useState } from "react";
import kit from "@/app/(landing)/page.module.css";
import { AppIcon } from "./AppParts";
import CopyButton from "./CopyButton";
import { Icon } from "./Icons";
import { reducedMotion } from "./motion";
import { gonePicks, ROWS, STEPS, SUMMARY } from "./story";
import styles from "./StorySection.module.css";
import { glideToStep, useActiveStep } from "./useActiveStep";

// The story: one chat in five steps, from the first message to the moment it
// forgets, with a guess at step 4 (which two of the six things you typed
// aren't in the summary). The server render shows step 1. Where the screen
// is tall enough, the figure pins while the steps scroll past it and the
// scroll chooses the step; the stepper and Next scroll to a step. Elsewhere
// the clicks choose it.

const YOURS = ROWS.map((row, i) => ({ row, i })).filter(({ row }) => row.k === "you");
const COUNT_MS = 700;

export default function StorySection() {
  const [step, setStep] = useState(0);
  // The step the person came from: only rows and the seam that are new animate in.
  const [from, setFrom] = useState(0);
  const [fresh, setFresh] = useState(false);
  const [shownPct, setShownPct] = useState(STEPS[0].pct);
  // The guess, as row numbers in ROWS.
  const [picks, setPicks] = useState<number[]>([]);
  const [revealed, setRevealed] = useState(false);
  const [putBack, setPutBack] = useState<number[]>([]);
  const [over, setOver] = useState(0);

  const tx = useRef<HTMLDivElement>(null);
  const body = useRef<HTMLDivElement>(null);
  const typed = useRef<HTMLDivElement>(null);
  const ask = useRef<HTMLDivElement>(null);
  const counting = useRef(0);
  const shown = useRef(shownPct);
  // The step shown, as go() last set it, for a choice made before a render.
  const now = useRef(0);
  const storyRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const stepRefs = useRef<(HTMLLIElement | null)[]>([]);
  const parts = { story: storyRef, steps: stepRefs, stage: stageRef };
  const refocus = useRef<{ key: string | null; scroll: number | null } | null>(null);

  useEffect(
    () => () => {
      counting.current++;
    },
    [],
  );

  function countTo(to: number, still = false) {
    const run = ++counting.current;
    const start = shown.current;
    if (still || reducedMotion() || start === to) {
      shown.current = to;
      setShownPct(to);
      return;
    }
    const t0 = performance.now();
    const tick = (now: number) => {
      if (run !== counting.current) return;
      const p = Math.min(1, (now - t0) / COUNT_MS);
      const eased = 1 - Math.pow(1 - p, 3);
      shown.current = Math.round(start + (to - start) * eased);
      setShownPct(shown.current);
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  // `still` puts the step in place without travelling there.
  function go(next: number, still = false) {
    const prev = now.current;
    now.current = next;
    setFrom(prev);
    setStep(next);
    setFresh(!still && !reducedMotion());
    if (next === 4) setRevealed(true);
    if (next < 3) {
      setPicks([]);
      setRevealed(false);
      setPutBack([]);
    }
    countTo(STEPS[next].pct, still);
  }

  // As the steps scroll past the pinned figure, it shows the one being read.
  useActiveStep(parts, (next, first) => {
    if (next !== now.current) go(next, first);
  });

  // The stepper and Next: where the figure pins, scroll to the step and let
  // the scroll show it; elsewhere, show it.
  function choose(next: number) {
    if (!glideToStep(next, parts)) go(next);
  }

  // The list redraws as a button becomes a row; put focus back on what had it.
  function redraw(scroll: number | null = null) {
    const active = document.activeElement as HTMLElement | null;
    const key = active && typed.current?.contains(active) ? (active.dataset.fk ?? null) : null;
    refocus.current = { key, scroll };
  }
  useLayoutEffect(() => {
    const saved = refocus.current;
    if (!saved || !typed.current) return;
    refocus.current = null;
    const target =
      (saved.key && typed.current.querySelector<HTMLElement>(`[data-fk="${saved.key}"]`)) || ask.current;
    target?.focus({ preventScroll: true });
    if (saved.scroll !== null)
      typed.current.querySelectorAll(`.${styles.pick}`)[saved.scroll]?.scrollIntoView({ block: "nearest" });
  });

  // Newest at the bottom, as in a chat: once the rows outgrow the frame, slide
  // them up. Again when the frame or the rows change size (the pinned frame
  // is as tall as the screen allows).
  useLayoutEffect(() => {
    const rows = tx.current;
    const frame = body.current;
    if (!rows || !frame || step >= 3) return;
    const measure = () => setOver(Math.max(0, rows.offsetHeight - (frame.clientHeight - 24)));
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    observer.observe(frame);
    observer.observe(rows);
    return () => observer.disconnect();
  }, [step]);

  const list = step >= 3;
  const found = gonePicks(picks).right.length;
  const lead = picks.length === 2 ? (found === 2 ? "You found both. " : `You found ${found} of 2. `) : "";
  const last = step === STEPS.length - 1;
  const rows = ROWS.filter((r) => r.at <= Math.min(step, 1));
  let n = 0;

  return (
    <section className={kit.sec} id="forgets" aria-labelledby="forgets-title">
      <div className={kit.wrap}>
        <div className={`${kit.head} ${kit.headCenter}`}>
          <span className={kit.kicker}>How it works</span>
          <h2 className={kit.h2} id="forgets-title">
            What it sees is lit. <span className={kit.faded}>What it forgot fades.</span>
          </h2>
          <p className={kit.lede}>
            Every chat has a limit. Near it, the agent swaps the chat for a short summary and keeps going. Some of what
            you typed doesn&rsquo;t make it, and nothing tells you. illumen does.
          </p>
        </div>
        <div className={styles.story} ref={storyRef}>
          <ol className={styles.stepper} aria-label="A long chat, step by step">
            {STEPS.map((s, i) => (
              <li
                key={s.t}
                className={styles.step}
                ref={(el) => {
                  stepRefs.current[i] = el;
                }}
              >
                <button type="button" aria-current={i === step ? "step" : undefined} onClick={() => choose(i)}>
                  <span className={styles.n}>{i + 1}</span>
                  <b>{s.t}</b>
                  <span>{s.s}</span>
                </button>
              </li>
            ))}
          </ol>
          <p className={kit.sr} aria-live="polite" id="story-live">
            {`Step ${step + 1} of ${STEPS.length}: ${STEPS[step].t} ${STEPS[step].s}`}
          </p>
          <div className={styles.stage} ref={stageRef}>
            <p className={styles.stepNow} id="step-now">
              {STEPS[step].s}
            </p>
            <div className={styles.fig} role="group" aria-label="The October newsletter chat, drawn">
              <div className={styles.figTop}>
                <svg className={styles.ring} viewBox="0 0 36 36" aria-hidden="true">
                  <circle className={styles.trk} cx="18" cy="18" r="15" />
                  <circle
                    className={styles.arc}
                    cx="18"
                    cy="18"
                    r="15"
                    strokeDasharray={`${(STEPS[step].pct / 100) * 94.2} 94.2`}
                  />
                </svg>
                <div className={styles.big}>
                  <b className={kit.num}>{shownPct}%</b>
                  <span>full</span>
                </div>
                <div className={styles.who}>
                  October newsletter
                  <br />
                  Claude Code · Sonnet 5.5
                </div>
              </div>
              <div className={`${kit.meter} ${styles.meter}`} aria-hidden="true">
                <i style={{ width: `${STEPS[step].pct}%` }} />
                <span className={styles.tick}>
                  <span>summarises near here</span>
                </span>
              </div>
              <div className={over > 0 && !list ? `${styles.figBody} ${styles.over}` : styles.figBody} ref={body}>
                <div
                  className={styles.tx}
                  ref={tx}
                  hidden={list}
                  style={{ transform: over > 0 ? `translateY(${-over}px)` : "none" }}
                >
                  {rows.map((r) => {
                    const isNew = fresh && r.at === step && step !== from && step <= 1;
                    return (
                      <div
                        key={ROWS.indexOf(r)}
                        className={`${r.k === "you" ? styles.txYou : styles.txWork}${isNew ? ` ${styles.isNew}` : ""}`}
                        style={isNew ? ({ "--d": `${n++ * 140}ms` } as CSSProperties) : undefined}
                      >
                        {r.k === "work" ? <AppIcon name="right" small /> : null}
                        {r.t}
                      </div>
                    );
                  })}
                  {step >= 2 ? (
                    <>
                      <div className={`${styles.seamLine}${fresh && from < 2 ? ` ${styles.isNew}` : ""}`}>
                        <span className={styles.seamBtn}>
                          <AppIcon name="book" small />
                          <span>Claude Code summarised the chat to keep going</span>
                          <span className={`${styles.seamFig} ${kit.num}`}>176k → 22k</span>
                        </span>
                      </div>
                      <div
                        className={`${styles.txSum}${fresh && from < 2 ? ` ${styles.isNew}` : ""}`}
                        style={{ "--d": "200ms" } as CSSProperties}
                      >
                        <b>The summary · all it reads of the chat now</b>
                        {SUMMARY}
                      </div>
                    </>
                  ) : null}
                </div>
                <div className={`${styles.typed}${revealed ? ` ${styles.done}` : ""}`} ref={typed} hidden={!list}>
                  <div className={styles.q} ref={ask} tabIndex={-1}>
                    {!revealed ? (
                      <>
                        <span>
                          The six things you typed. <b>Tap the two</b>{" "}you think aren&rsquo;t in the summary.
                        </span>
                        <button
                          type="button"
                          className={`${kit.btn} ${kit.btnLine}`}
                          data-fk="showme"
                          onClick={() => {
                            redraw();
                            setRevealed(true);
                          }}
                        >
                          Just show me
                        </button>
                      </>
                    ) : (
                      <span className={styles.result}>
                        {lead}illumen checks each thing you typed against the summary, word for word.
                      </span>
                    )}
                  </div>
                  {YOURS.map(({ row, i }) => {
                    const mine = picks.includes(i);
                    if (!revealed) {
                      return (
                        <button
                          key={i}
                          type="button"
                          className={styles.pick}
                          data-fk={`pick${i}`}
                          aria-pressed={mine}
                          onClick={() => {
                            redraw();
                            const next = mine ? picks.filter((p) => p !== i) : picks.length < 2 ? [...picks, i] : picks;
                            setPicks(next);
                            if (next.length === 2) setRevealed(true);
                          }}
                        >
                          <span className={styles.t}>{row.t}</span>
                        </button>
                      );
                    }
                    const back = row.gone && putBack.includes(i);
                    const cls = `${styles.pick}${back ? ` ${styles.isBack}` : row.gone ? ` ${styles.isGone}` : ""}`;
                    return (
                      <div key={i} className={cls}>
                        <span className={styles.t}>{row.t}</span>
                        <span className={styles.res}>
                          {back ? (
                            <>
                              <AppIcon name="putback" small />
                              Goes back in with your next message
                            </>
                          ) : row.gone ? (
                            <>
                              <AppIcon name="eyeOff" small />
                              gone
                            </>
                          ) : (
                            <>
                              <Icon name="check" small />
                              in the summary
                            </>
                          )}
                          {mine ? <span className={styles.res}>· you picked it</span> : null}
                        </span>
                        {row.gone && step === 4 && !back ? (
                          <>
                            <CopyButton text={row.t} what={`“${row.t}”`} className={`${kit.btn} ${kit.btnSm}`} />
                            <button
                              type="button"
                              className={`${kit.btn} ${kit.btnSm} ${kit.btnLine}`}
                              data-fk={`pb${i}`}
                              onClick={() => {
                                redraw(YOURS.findIndex((y) => y.i === i));
                                setPutBack([...putBack, i]);
                              }}
                            >
                              <AppIcon name="putback" small />
                              Put back
                            </button>
                          </>
                        ) : null}
                      </div>
                    );
                  })}
                </div>
              </div>
              <div className={styles.figFoot}>
                <span>Fig. 1 · One chat, from the first message to the moment it forgets.</span>
                <button
                  type="button"
                  className={`${kit.btn} ${kit.btnLine}`}
                  onClick={() => choose(last ? 0 : step + 1)}
                >
                  {last ? "Start over" : "Next"}
                  <Icon name="arrow" small />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
