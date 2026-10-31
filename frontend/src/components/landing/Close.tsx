"use client";

import { useRef, useState } from "react";
import kit from "@/app/(landing)/page.module.css";
import { estimateTokens, tokensText } from "@/lib/landing";
import styles from "./Sections.module.css";

const MAX_HEIGHT_PX = 200;
const CIRCLE = 37.7;
const WINDOW_TOKENS = 200000;

/** The closing composer: type a first message and see about how much of the
 * context it would take. Starting the chat is coming soon. */
export default function Close() {
  const [text, setText] = useState("");
  const field = useRef<HTMLTextAreaElement>(null);
  const n = estimateTokens(text);

  function grow() {
    const el = field.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_HEIGHT_PX)}px`;
  }

  return (
    <section className={styles.close} id="start" aria-labelledby="close-title">
      <div className={kit.wrap}>
        <h2 className={`${kit.h2} ${styles.closeTitle}`} id="close-title">
          Start a chat and look inside.
        </h2>
        <div className={styles.closeBox}>
          <div className={styles.composer}>
            <label className={kit.sr} htmlFor="close-field">
              Your first message
            </label>
            <textarea
              id="close-field"
              ref={field}
              rows={2}
              placeholder="Ask anything. You'll pick who answers next."
              value={text}
              onChange={(e) => {
                setText(e.target.value);
                grow();
              }}
            />
            <div className={styles.cRow}>
              <span className={styles.estimate}>
                <svg className={`${styles.ring} ${styles.ringLg}`} viewBox="0 0 16 16" aria-hidden="true">
                  <circle className={styles.trk} cx="8" cy="8" r="6" />
                  <circle
                    className={styles.arc}
                    cx="8"
                    cy="8"
                    r="6"
                    strokeDasharray={`${Math.min(CIRCLE, (n / WINDOW_TOKENS) * CIRCLE)} ${CIRCLE}`}
                  />
                </svg>
                <span className={kit.num}>{n ? `≈ +${tokensText(n)} tokens` : "Context"}</span>
              </span>
              <span className={styles.grow} />
              <span className={`${kit.btn} ${kit.btnSolid} ${kit.btnSm} ${kit.soon}`}>Coming soon</span>
            </div>
          </div>
        </div>
        <p className={styles.fine}>Free. No account. Runs on your Mac.</p>
      </div>
    </section>
  );
}
