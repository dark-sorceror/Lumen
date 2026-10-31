"use client";

import { useRef, useState } from "react";
import kit from "@/app/(landing)/page.module.css";
import { estimateTokens, isSendKey, tokensText } from "@/lib/landing";
import styles from "./Sections.module.css";
import { SIconLg } from "./sectionIcons";

const MAX_HEIGHT_PX = 200;
const CIRCLE = 37.7;
const WINDOW_TOKENS = 200000;

/** The closing composer: type a first message, and it starts the chat
 * (the page's start behaviour) with the words. */
export default function Close({ onStart }: { onStart: (words: string, opener: HTMLElement) => void }) {
  const [text, setText] = useState("");
  const field = useRef<HTMLTextAreaElement>(null);
  const send = useRef<HTMLButtonElement>(null);
  const n = estimateTokens(text);

  function grow() {
    const el = field.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_HEIGHT_PX)}px`;
  }

  function submit() {
    if (send.current) onStart(text.trim(), send.current);
  }

  return (
    <section className={styles.close} id="start" aria-labelledby="close-title">
      <div className={kit.wrap}>
        <h2 className={`${kit.h2} ${styles.closeTitle}`} id="close-title">
          Start a chat and look inside.
        </h2>
        <form
          action="#start"
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
        >
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
              onKeyDown={(e) => {
                if (!isSendKey({ key: e.key, shiftKey: e.shiftKey, isComposing: e.nativeEvent.isComposing })) return;
                e.preventDefault();
                submit();
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
              <button type="submit" ref={send} className={styles.send} aria-label="Start the chat">
                <SIconLg name="up" />
              </button>
            </div>
          </div>
        </form>
        <p className={styles.fine}>Free. No account. Your words wait in the new chat while you pick who answers.</p>
      </div>
    </section>
  );
}
