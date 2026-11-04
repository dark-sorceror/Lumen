import { Fragment, type Ref } from "react";
import { tokensText } from "@/lib/landing";
import { cx } from "./AppParts";
import { fullBar, paneFull, PEAK, roomParts, type Session, SUMMARISED_AT } from "./terminal";
import styles from "./TerminalDrawing.module.css";

// The plugin's small pane beside the transcript: how full it is, where it
// stood before the summary, what takes the room, and a way to what it forgot.
// Drawn in block characters, as a terminal pane is. TerminalDrawing opens it.

export default function TerminalPane({
  paneRef,
  s,
  open,
  flash,
  onForgot,
}: {
  paneRef: Ref<HTMLDivElement>;
  s: Session;
  open: boolean;
  /** Counts each time a command opened (or reopened) it: it lights a moment. */
  flash: number;
  onForgot: () => void;
}) {
  const full = paneFull(s);
  const [lit, rest] = fullBar(s);
  const parts = roomParts(s);
  const summarised = s.compactedAt !== null;
  return (
    <div ref={paneRef} className={styles.tpane} role="group" aria-label="The plugin's pane" hidden={!open}>
      {flash > 0 && <span key={flash} className={styles.flash} aria-hidden="true" />}
      <div className={styles.tpaneIn}>
        <div className={styles.ttl}>
          <b>illumen</b>
          <span className={styles.tDim}>esc to close</span>
        </div>
        <div>
          <p className={styles.big}>{full.big}</p>
          <p className={styles.tDim}>{full.small}</p>
          <div className={styles.tbars} aria-hidden="true">
            <span className={styles.tInk}>{lit}</span>
            <span className={styles.tDim}>{rest}</span>
          </div>
          {summarised && (
            <>
              <div className={cx(styles.tbars, styles.tFaded)} aria-hidden="true">
                {PEAK.mark}
              </div>
              <p className={styles.tFaded}>{PEAK.note}</p>
            </>
          )}
        </div>
        <div>
          <p className={styles.lbl}>What takes the room</p>
          <div className={styles.tbars} aria-hidden="true">
            {parts.map((p) => (
              <span key={p.label} style={{ color: p.hue }}>
                {"█".repeat(p.cells)}
              </span>
            ))}
          </div>
          <p className={styles.tDim}>
            {parts.map((p, i) => (
              <Fragment key={p.label}>
                <span style={{ color: p.hue }} aria-hidden="true">
                  ■{" "}
                </span>
                {`${p.label} ${tokensText(p.tokens)}${i < parts.length - 1 ? "  " : ""}`}
              </Fragment>
            ))}
          </p>
        </div>
        {summarised && (
          <button type="button" className={styles.rowLink} onClick={onForgot}>
            <span>
              Forgot <span className={styles.tRed}>2</span> things you typed{s.restored.length > 0 ? ` · ${s.restored.length} put back` : ""} · {SUMMARISED_AT}
            </span>
            <span aria-hidden="true">›</span>
          </button>
        )}
      </div>
    </div>
  );
}
