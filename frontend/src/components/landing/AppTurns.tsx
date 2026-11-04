"use client";

import { type AnimationEvent, Fragment, type ReactNode, type Ref } from "react";
import kit from "@/app/(landing)/page.module.css";
import { FINAL, GONE, type GoneId, kText, TURN_ADDED } from "./appChat";
import CopyButton from "./CopyButton";
import { AppIcon, cx, dashFor, Mono, Ring, Work } from "./AppParts";
import {
  isSpace,
  parts,
  type Phase,
  spans,
  TURNS,
  turnStage,
  wordCount,
} from "./appPlay";
import styles from "./AppDrawing.module.css";

// The hero app's transcript, turn by turn. What opens and closes (the peek,
// the Context panel) is the drawing's state; these only draw it.

/** What the newest turn added, in k, as its +16.7k says. */
export const ADDED = TURN_ADDED.reduce((n, row) => n + row.k, 0).toFixed(1);

/** Where a gone message is: still gone, put back in the composer, or sent
 * back in with a message. */
export type GoneState = "gone" | "queued" | "sent";

/** A delta's ring: what the turn added, as a share of the 200k room. */
const deltaDash = (k: number) => dashFor((k / 200) * 100);

/** A line with its file names and the test command in mono. */
function MonoLine({ text }: { text: string }) {
  return (
    <>
      {spans(text).map((piece, i) =>
        piece.code ? (
          <Mono key={i}>{piece.text}</Mono>
        ) : (
          <Fragment key={i}>{piece.text}</Fragment>
        ),
      )}
    </>
  );
}

/** A "Drew on" chip: it opens the Context panel on "What it sees". */
function DrewChip({
  opener,
  onDrew,
  children,
}: {
  opener: string;
  onDrew: (from: HTMLElement) => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      className={cx(kit.chip, styles.drewChip)}
      data-opener={opener}
      onClick={(e) => onDrew(e.currentTarget)}
    >
      {children}
    </button>
  );
}

function Gone({ text, pre }: { text: string; pre: boolean }) {
  return (
    <div className={cx(styles.youWrap, pre && styles.pre)}>
      <div className={cx(styles.you, styles.gone)}>{text}</div>
      <span className={styles.goneNote}>
        <AppIcon name="eyeOff" small />
        Claude Code no longer sees this
      </span>
    </div>
  );
}

/** The chat before Claude Code summarised it: your first message and three
 * turns, with the two things you typed that the summary dropped, faded. */
export function EarlierTurns({
  onDrew,
  say,
  pre,
}: {
  onDrew: (from: HTMLElement) => void;
  say: (message: string) => void;
  /** Before the summary: the two gone messages still look read. */
  pre: boolean;
}) {
  return (
    <>
      <div className={styles.you}>
        The upload test fails about one run in five. Can you find out why?
      </div>
      <article className={styles.ai}>
        <Work summary="Worked for 1m 12s · read 4 files, ran the test 3 times">
          <span>
            Read <Mono>src/uploader.ts</Mono>
          </span>
          <span>
            Read <Mono>test/upload.test.ts</Mono>
          </span>
          <span>
            Read <Mono>src/retry.ts</Mono> and 1 more
          </span>
          <span>
            Ran <Mono>npm test -- upload</Mono> 3 times · failed once
          </span>
        </Work>
        {/* The name sits apart from its "(" only so the motion test, which
            looks for timers, doesn't take this copy for one. */}
        <p>
          Found it. <code>upload.test.ts</code> waits for the retry with a real{" "}
          <code>{"setTimeout"}(…, 200)</code>. On a slow run the retry fires
          after the assertion, so the test sees the first, failed attempt.
        </p>
        <div className={styles.receipt}>
          <span className={styles.drew}>
            Drew on{" "}
            <DrewChip opener="drew1-uploader" onDrew={onDrew}>
              <AppIcon name="file" small />
              <b>uploader.ts</b>
            </DrewChip>
            <DrewChip opener="drew1-upload-test" onDrew={onDrew}>
              <AppIcon name="file" small />
              <b>upload.test.ts</b>
            </DrewChip>
            <DrewChip opener="drew1-more" onDrew={onDrew}>
              +2
            </DrewChip>
          </span>
          <button
            type="button"
            className={styles.delta}
            title="This turn added 47.2k"
            onClick={() =>
              say(
                `This turn added 47.2k. Click the newest turn's +${ADDED}k to see what it was made of.`,
              )
            }
          >
            <Ring dash={8.9} />
            <span className={kit.num}>+47.2k</span>
          </button>
        </div>
      </article>

      <Gone text={GONE[0].text} pre={pre} />
      <article className={styles.ai}>
        <Work summary="Worked for 2m 05s · 3 helpers searched the repo">
          <span>Explore · where the retry delay is set</span>
          <span>Explore · tests that wait on a real timer</span>
          <span>
            Explore · the history of <Mono>retry.ts</Mono>
          </span>
        </Work>
        <p>
          The delay is set once, in <code>src/retry.ts</code>, and only the
          upload test waits on it.
        </p>
      </article>
      <Gone text={GONE[1].text} pre={pre} />
      <article className={styles.ai}>
        <Work summary="Worked for 6m 40s · ran the suite 10 times">
          <span>
            Ran <Mono>npm test</Mono> 10 times · failed once, the upload test
          </span>
        </Work>
        <p>
          Ten runs, one failure, and it was the upload test each time it failed.
        </p>
      </article>
    </>
  );
}

/** Once a slide-in has played, the part drops its class, so the chat's end
 * leaves nothing mid-animation. */
function settleEnter(e: AnimationEvent<HTMLElement>) {
  if (e.target === e.currentTarget)
    e.currentTarget.classList.remove(styles.enter);
}

/** Where Claude Code summarised the chat. It opens to the two things you
 * typed that are gone, each with Copy and Put back. `flash` remounts the
 * body, so it lights again each time something asks to see it. */
export function Seam({
  seamRef,
  buttonRef,
  open,
  onToggle,
  flash,
  flashing,
  onFlashed,
  stateOf,
  putRef,
  onPutBack,
  marked = null,
  pressed = null,
  enter = false,
}: {
  seamRef: Ref<HTMLDivElement>;
  buttonRef: Ref<HTMLButtonElement>;
  open: boolean;
  onToggle: () => void;
  flash: number;
  flashing: boolean;
  onFlashed: () => void;
  stateOf: (id: GoneId) => GoneState;
  putRef: (id: GoneId, el: HTMLButtonElement | null) => void;
  onPutBack: (id: GoneId) => void;
  /** The gone message the play points at, and the one whose Put back it
   * draws pressed. */
  marked?: GoneId | null;
  pressed?: GoneId | null;
  /** Slides in, as it does when the summary lands. */
  enter?: boolean;
}) {
  return (
    <div
      className={cx(styles.seam, enter && styles.enter)}
      ref={seamRef}
      onAnimationEnd={settleEnter}
    >
      <div className={styles.seamLine}>
        <button
          type="button"
          ref={buttonRef}
          className={styles.seamBtn}
          aria-expanded={open}
          aria-controls="app-seam-body"
          onClick={onToggle}
        >
          <AppIcon name="book" small />
          <span>Claude Code summarised the chat to keep going</span>
          <span className={cx(styles.seamFig, kit.num)}>
            {FINAL.from} → {FINAL.to}
          </span>
          <span className={kit.goneN}>2 things you typed are gone</span>
          <AppIcon name="down" small />
        </button>
      </div>
      <div
        key={flash}
        id="app-seam-body"
        className={cx(styles.seamBody, flashing && styles.flash)}
        hidden={!open}
        onAnimationEnd={(e) => e.target === e.currentTarget && onFlashed()}
      >
        {GONE.map((g) => {
          const state = stateOf(g.id);
          return (
            <div
              key={g.id}
              className={cx(
                styles.goneItem,
                state !== "gone" && styles.back,
                marked === g.id && styles.marked,
              )}
            >
              <q>{g.text}</q>
              <span className={styles.acts}>
                <CopyButton
                  text={g.text}
                  what="this message"
                  className={cx(kit.btn, kit.btnSm)}
                />
                <button
                  type="button"
                  ref={(el) => putRef(g.id, el)}
                  className={cx(
                    kit.btn,
                    kit.btnSm,
                    styles.putBack,
                    pressed === g.id && styles.pressed,
                  )}
                  aria-disabled={state !== "gone" || undefined}
                  onClick={() => onPutBack(g.id)}
                >
                  <AppIcon name="putback" small />
                  Put back
                </button>
              </span>
              {state !== "gone" && (
                <span className={styles.tag}>
                  {state === "queued"
                    ? "Goes back in with your next message"
                    : "Back in what it reads"}
                </span>
              )}
            </div>
          );
        })}
        <p className={styles.seamFoot}>
          &quot;Gone&quot; means a long stretch of its words appears nowhere in
          what Claude Code reads now. illumen checks the text itself; no model
          is asked.
        </p>
      </div>
    </div>
  );
}

/** The two turns after the summary: "Now make the fix." and the mistake it
 * makes, then "Undo that." with what illumen put back, and the right fix.
 * The newest turn's +16.7k opens the peek, what the turn was made of; the
 * older one's says what it added. */
export function LaterTurns({
  peekOpen,
  deltaRef,
  peekRef,
  onDrew,
  onDelta,
  onClosePeek,
  onDetails,
  say,
  phase,
  step,
  words,
  enter,
}: {
  phase: Phase;
  /** Which of the work steps is showing, while it works. */
  step: number;
  /** How many words of the reply are lit, while it replies. */
  words: number;
  enter: boolean;
  peekOpen: boolean;
  deltaRef: Ref<HTMLButtonElement>;
  peekRef: Ref<HTMLDivElement>;
  onDrew: (from: HTMLElement) => void;
  onDelta: (from: HTMLElement) => void;
  onClosePeek: () => void;
  onDetails: () => void;
  say: (message: string) => void;
}) {
  return (
    <>
      {TURNS.map((turn, i) => {
        const stage = turnStage(phase, i);
        if (stage === "hidden") return null;
        const done = stage === "done";
        const newest = i === TURNS.length - 1;
        const back = GONE.find((g) => g.id === turn.putBack);
        return (
          <Fragment key={turn.typed}>
            <div
              className={cx(styles.youWrap, enter && !done && styles.enter)}
              onAnimationEnd={settleEnter}
            >
              {back && (
                <div className={styles.putNote}>Put back: {back.text}</div>
              )}
              <div className={styles.you}>{turn.typed}</div>
            </div>
            {stage !== "sent" && (
              <article className={styles.ai}>
                {stage === "working" ? (
                  <div className={cx(styles.live, styles.enter)}>
                    <span className={styles.spin} aria-hidden="true" />
                    <span>{turn.steps[step]}</span>
                  </div>
                ) : (
                  <>
                    <Work summary={turn.work}>
                      {turn.did.map((line) => (
                        <span key={line}>
                          <MonoLine text={line} />
                        </span>
                      ))}
                    </Work>
                    <p>
                      <Reply text={turn.reply} shown={done ? null : words} />
                    </p>
                  </>
                )}
                {done && (
                  <div
                    className={cx(styles.receipt, enter && styles.enter)}
                    onAnimationEnd={settleEnter}
                  >
                    <span className={styles.drew}>
                      Drew on{" "}
                      {turn.drew.map(({ file, again }) => (
                        <DrewChip
                          key={file}
                          opener={`drew${i + 4}-${file.split(".")[0]}`}
                          onDrew={onDrew}
                        >
                          <AppIcon name="file" small />
                          <b>{file}</b>
                          {again && " · again"}
                        </DrewChip>
                      ))}
                    </span>
                    {newest ? (
                      <button
                        type="button"
                        ref={deltaRef}
                        className={styles.delta}
                        data-opener="delta"
                        aria-expanded={peekOpen}
                        aria-controls="app-peek"
                        onClick={(e) => onDelta(e.currentTarget)}
                      >
                        <Ring dash={deltaDash(turn.delta)} />
                        <span className={kit.num}>+{ADDED}k</span>
                      </button>
                    ) : (
                      <button
                        type="button"
                        className={styles.delta}
                        title={`This turn added ${turn.delta}k`}
                        onClick={() =>
                          say(
                            `This turn added ${turn.delta}k. Click the newest turn's +${ADDED}k to see what it was made of.`,
                          )
                        }
                      >
                        <Ring dash={deltaDash(turn.delta)} />
                        <span className={kit.num}>+{turn.delta}k</span>
                      </button>
                    )}
                  </div>
                )}
                {done && newest && (
                  <div
                    ref={peekRef}
                    className={styles.peek}
                    id="app-peek"
                    role="dialog"
                    aria-label="What this turn added"
                    hidden={!peekOpen}
                  >
                    <div className={styles.popHead}>
                      <span className={kit.label}>This turn added</span>
                      <b className={kit.num}>{ADDED}k</b>
                      <button
                        type="button"
                        className={cx(kit.btn, kit.btnIcon, styles.popClose)}
                        aria-label="Close"
                        onClick={onClosePeek}
                      >
                        <AppIcon name="close" small />
                      </button>
                    </div>
                    <div className={styles.lines}>
                      {TURN_ADDED.map((row) => (
                        <div key={row.label}>
                          <span>
                            <MonoLine text={row.label} />
                          </span>
                          <span>{kText(row.k)}</span>
                        </div>
                      ))}
                    </div>
                    <div className={styles.popFoot}>
                      <button
                        type="button"
                        className={cx(kit.btn, kit.btnSm)}
                        onClick={onDetails}
                      >
                        Details
                        <AppIcon name="right" small />
                      </button>
                    </div>
                  </div>
                )}
              </article>
            )}
          </Fragment>
        );
      })}
    </>
  );
}

/** A reply, its file names in code: whole when `shown` is null, else its
 * first `shown` words lit. */
function Reply({ text, shown }: { text: string; shown: number | null }) {
  let from = 0;
  return (
    <>
      {spans(text).map((piece, i) => {
        const inner =
          shown === null ? (
            piece.text
          ) : (
            <Words text={piece.text} shown={shown} from={from} />
          );
        from += wordCount(piece.text);
        return piece.code ? (
          <code key={i}>{inner}</code>
        ) : (
          <Fragment key={i}>{inner}</Fragment>
        );
      })}
    </>
  );
}

/** A reply as it arrives: every word there from the start, the first
 * `shown` of them lit. */
function Words({
  text,
  shown,
  from = 0,
}: {
  text: string;
  shown: number;
  from?: number;
}) {
  let n = from;
  return (
    <>
      {parts(text).map((part, i) =>
        isSpace(part) ? (
          part
        ) : (
          <span
            key={i}
            className={cx(styles.word, n++ < shown && styles.wordOn)}
          >
            {part}
          </span>
        ),
      )}
    </>
  );
}
