import { Fragment, type Ref } from "react";
import { cx } from "./AppParts";
import { BACK_WORDS, FORGOT_OUT, forgotCaveat, GONE_WORDS, LEFT_OUT, type Line, restoreHint, SEAM_TEXT, type Session } from "./terminal";
import styles from "./TerminalDrawing.module.css";

// The terminal drawing's transcript: Claude Code's start-up banner, the
// session's lines (the seam row where it summarised), and what each command
// you ran printed. Drawn from what it's handed; TerminalDrawing keeps the state.

/** A command you ran and what it printed. */
export type Output = { id: number; ask: string; kind: "forgot" | "text"; text?: string };

/** Claude Code's start-up banner: its mascot, drawn as the pixels its three
 * rows of block characters make (" ▐▛███▜▌", "▝▜█████▛▘", "  ▘▘ ▝▝"; each
 * pixel half a column wide and half a line tall, the eyes left unlit), then
 * its version, the model and the folder. */
function Banner() {
  return (
    <div className={styles.tHello}>
      <span className={styles.tClawd}>
        <svg viewBox="0 0 18 6" preserveAspectRatio="none" shapeRendering="crispEdges" aria-hidden="true">
          <path
            fill="var(--claude)"
            d="M3 0h12v1H3zM3 1h2v1H3zM6 1h6v1H6zM13 1h2v1h-2zM1 2h16v1H1zM3 3h12v1H3zM4 4h1v1H4zM6 4h1v1H6zM11 4h1v1h-1zM13 4h1v1h-1z"
          />
        </svg>
      </span>
      <div>
        <p>
          <b>Claude Code</b>
          <span className={styles.tDim}> v2.1.316</span>
        </p>
        <p className={styles.tDim}>Opus 5.5 with high effort · Claude Max</p>
        <p className={styles.tDim}>~/invoice-service</p>
      </div>
    </div>
  );
}

/** Claude Code's permission prompt, drawn inline as a thin box (not a control);
 * once declined it is the one dim line saying so. */
function Ask({ line }: { line: Extract<Line, { kind: "ask" }> }) {
  if (line.choice === "no")
    return <p className={styles.tDim}>{"  ⎿  No, and tell Claude what to do differently"}</p>;
  const option = (n: 1 | 2, text: string) => (
    <p className={line.choice === n ? styles.tAsked : undefined}>
      {line.choice === n ? "❯ " : "  "}
      {n}. {text}
    </p>
  );
  return (
    <div className={styles.tAsk} role="group" aria-label="Claude Code asks permission">
      <p className={styles.tDim}>Bash command</p>
      <p>{`  ${line.command}`}</p>
      <p className={styles.tDim}>{`  ${line.about}`}</p>
      <p>Do you want to proceed?</p>
      {option(1, "Yes")}
      {option(2, "No, and tell Claude what to do differently")}
    </div>
  );
}

function SessionLine({ line }: { line: Exclude<Line, { kind: "done" }> }) {
  if (line.kind === "ask") return <Ask line={line} />;
  if (line.kind === "forgot")
    return (
      <div className={styles.tOut}>
        <Forgot putBack={[]} restored={[]} />
      </div>
    );
  if (line.kind === "text")
    return (
      <div className={styles.tOut}>
        <div className={styles.body}>
          <pre>{line.text}</pre>
        </div>
      </div>
    );
  if (line.kind === "user")
    return (
      <p className={cx(styles.tYou, styles.tHang)}>
        <span className={styles.tDim}>&gt; </span>
        {line.text}
      </p>
    );
  if (line.kind === "tool")
    return (
      <div>
        <p className={styles.tHang}>
          <span className={styles.tOk}>⏺</span> {line.text}
        </p>
        {line.result !== undefined && <p className={styles.tDim}>{`  ⎿ ${line.result}`}</p>}
      </div>
    );
  if (line.kind === "reply") return <p className={styles.tHang}>⏺ {line.text}</p>;
  if (line.kind === "worked") return <p className={styles.tDim}>✻ {line.text}</p>;
  return (
    <p>
      <span className={styles.tClaude}>✻</span> {line.text}
    </p>
  );
}

type GoneProps = {
  putBack: readonly number[];
  /** The ones the script put back and sent. */
  restored: readonly number[];
  /** Which one says "[ Copied ]" now. */
  copied: number | null;
  onCopy: (i: number) => void;
  onPutBack: (i: number) => void;
};

/** Where Claude Code summarised: one line that opens to the two things you
 * typed that are gone, each to copy or put back in your prompt. */
function Seam({ open, onToggle, putBack, restored, copied, onCopy, onPutBack }: GoneProps & { open: boolean; onToggle: () => void }) {
  return (
    <div>
      <button type="button" className={styles.tSeam} aria-expanded={open} onClick={onToggle}>
        {SEAM_TEXT}
        <span className={styles.tRed}>{GONE_WORDS}</span>
        {open ? " ▾" : " ▸"}
      </button>
      {open && (
        <div className={styles.tSeamBody}>
          {LEFT_OUT.map((text, i) => {
            const sent = restored.includes(i);
            const back = sent || putBack.includes(i);
            return (
              <div key={text} className={cx(styles.tGone, back && styles.back)}>
                <q>{text}</q>
                <span className={styles.tActs}>
                  <button type="button" className={styles.tBtn} onClick={() => onCopy(i)}>
                    {copied === i ? "[ Copied ]" : "[ Copy ]"}
                  </button>
                  {sent ? (
                    <span className={styles.tDim}>{BACK_WORDS}</span>
                  ) : (
                    <button type="button" className={styles.tBtn} disabled={back} onClick={() => onPutBack(i)}>
                      {back ? "in your prompt" : "[ Put back ]"}
                    </button>
                  )}
                </span>
              </div>
            );
          })}
          <p className={styles.tDim}>illumen looks for each one word for word in the summary.</p>
        </div>
      )}
    </div>
  );
}

/** /illumen-forgot's output: the two things the summary left out, and how to put one back. */
function Forgot({ putBack, restored }: { putBack: readonly number[]; restored: readonly number[] }) {
  const hint = restoreHint(restored);
  return (
    <div className={styles.body}>
      <p>
        {FORGOT_OUT.before}
        <span className={styles.tRed}>{FORGOT_OUT.gone}</span>
        {FORGOT_OUT.after}
      </p>
      <div className={styles.gap} />
      {LEFT_OUT.map((text, i) => {
        const back = putBack.includes(i);
        return (
          <div key={text} className={styles.item}>
            <span className={styles.tDim}>{i + 1}</span>
            <span className={restored.includes(i) ? styles.tInk : styles.tFaded}>
              {text}
              {restored.includes(i) ? (
                <span className={styles.tDim}>{FORGOT_OUT.restored}</span>
              ) : (
                back && <span className={styles.tDim}>{FORGOT_OUT.inPrompt}</span>
              )}
            </span>
          </div>
        );
      })}
      <div className={styles.gap} />
      <p className={styles.tDim}>{forgotCaveat(restored)}</p>
      {hint && <pre className={styles.tDim}>{hint}</pre>}
    </div>
  );
}

export default function TerminalLog({
  logRef,
  s,
  waiting,
  outputs,
  seamOpen,
  onSeam,
  ...gone
}: GoneProps & {
  logRef: Ref<HTMLDivElement>;
  s: Session;
  /** Whether Claude Code is between a line and the next: "Working…". */
  waiting: boolean;
  outputs: readonly Output[];
  seamOpen: boolean;
  onSeam: () => void;
}) {
  const lastAt = s.lines.length - 1;
  return (
    <div ref={logRef} className={styles.tlog} aria-live="off">
      <Banner />
      {s.lines.map(({ key, line }, i) => {
        // "Compacting…" shows only until the next line.
        if (line.kind === "busy" && i !== lastAt) return null;
        if (line.kind !== "done") return <SessionLine key={key} line={line} />;
        return (
          <Fragment key={key}>
            <p>
              <span className={styles.tDim}>✻ Conversation compacted</span>
            </p>
            <Seam open={seamOpen} onToggle={onSeam} {...gone} />
          </Fragment>
        );
      })}
      {waiting && (
        <p className={styles.tBusy}>
          <span className={styles.tClaude}>✻</span> Working…
        </p>
      )}
      {outputs.map((o) => (
        <div key={o.id} className={styles.tOut}>
          <p className={cx(styles.tYou, styles.tHang)}>
            <span className={styles.tDim}>&gt; </span>
            {o.ask}
          </p>
          {o.kind === "forgot" ? (
            <Forgot putBack={gone.putBack} restored={gone.restored} />
          ) : (
            <div className={styles.body}>
              <pre>{o.text}</pre>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
