"use client";

import { useState } from "react";
import kit from "@/app/(landing)/page.module.css";
import styles from "./Sections.module.css";
import { SIcon } from "./sectionIcons";
import { WAYS } from "./waysIn";

/** Ways in: pick where the chat runs and read where what you send goes. */
export default function WaysIn() {
  const [picked, setPicked] = useState(WAYS[0].id);
  const way = WAYS.find((w) => w.id === picked) ?? WAYS[0];

  return (
    <section className={kit.sec} id="ways-in" aria-labelledby="ways-title">
      <div className={kit.wrap}>
        <div className={`${kit.head} ${kit.headCenter}`}>
          <span className={kit.kicker}>Ways in</span>
          <h2 className={kit.h2} id="ways-title">
            Use the AI you already have.
          </h2>
          <p className={kit.lede}>
            Your Claude Code or Codex login, your own API key, or a model that runs on your Mac. Whatever runs the
            chat, illumen shows the same things.
          </p>
        </div>
        <div className={styles.ways}>
          <div className={styles.agents} role="group" aria-label="Where the chat runs">
            {WAYS.map((w) => (
              <button
                key={w.id}
                type="button"
                className={styles.agent}
                aria-pressed={w.id === picked}
                onClick={() => setPicked(w.id)}
              >
                <SIcon name={w.icon} />
                {w.name}
                {w.dot && (
                  <>
                    <span
                      className={`${kit.dot} ${w.dot === "ok" ? kit.dotOk : `${kit.dotWait} ${styles.waitDot}`}`}
                      aria-hidden="true"
                    />
                    <span className={kit.sr}>{w.dot === "ok" ? ", ready" : ", not loaded yet"}</span>
                  </>
                )}
              </button>
            ))}
          </div>
          <p className={styles.wayNote} aria-live="polite">
            {way.text}
            <b>{way.leave}</b>
          </p>
          <ul className={styles.keeps}>
            <li>
              <SIcon name="check" />
              Chats are saved on your Mac.
            </li>
            <li>
              <SIcon name="check" />
              Your keys are stored only on your Mac.
            </li>
            <li>
              <SIcon name="check" />
              No account. No sign-in.
            </li>
          </ul>
          <p className={styles.smallPrint}>
            A hosted model, Claude Code or Codex still gets what you send it; that&apos;s how it answers.
          </p>
        </div>
      </div>
    </section>
  );
}
