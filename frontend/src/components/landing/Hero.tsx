"use client";

import { type CSSProperties, type KeyboardEvent, type ReactNode } from "react";
import kit from "@/app/(landing)/page.module.css";
import { ClaudeMark, Icon } from "./Icons";
import styles from "./Hero.module.css";

export type HeroView = "app" | "plugin";

const SWITCH_NOTE: Record<HeroView, string> = {
  app: "A chat app for your Mac, with Claude Code, Codex, your keys or a local model.",
  plugin: "A plugin for the Claude Code you already run in your terminal.",
};

/** The page's top: what illumen is, the way in, the app/plugin switch, and
 * the window under it. `window` fills the window (the app or the terminal),
 * `replay` sits in its title bar, `tryRow` goes under it, and `title` is the
 * bar's centred title. */
export default function Hero({
  window,
  tryRow,
  view,
  onView,
  title,
  replay,
}: {
  window: ReactNode;
  tryRow: ReactNode;
  view: HeroView;
  onView: (view: HeroView, focus?: boolean) => void;
  title: string;
  replay?: ReactNode;
}) {
  const plug = view === "plugin";

  function onKey(e: KeyboardEvent<HTMLButtonElement>, mine: HeroView) {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    e.preventDefault();
    onView(mine === "app" ? "plugin" : "app", true);
  }

  return (
    <section className={styles.hero} aria-labelledby="hero-title">
      <div className={kit.wrap}>
        <span className={styles.eyebrow}>
          <svg className={styles.ring} viewBox="0 0 16 16" aria-hidden="true">
            <circle className={styles.trk} cx="8" cy="8" r="6" />
            <circle className={styles.arc} cx="8" cy="8" r="6" strokeDasharray="7.9 37.7" />
          </svg>
          A glass box for your AI&rsquo;s context
        </span>
        <h1 className={styles.h1} id="hero-title">
          <span className={styles.lit}>See what your AI sees.</span>
        </h1>
        <p className={styles.sub}>
          illumen shows how full your AI&rsquo;s context is, what it forgot, and what each answer drew on. Chat in the
          illumen app on your Mac, or bring it into Claude Code.
        </p>
        <div className={styles.cta}>
          <span className={`${kit.btn} ${kit.btnSolid} ${kit.btnLg} ${kit.soon}`}>Coming soon</span>
          <span className={`${kit.btn} ${kit.btnLine} ${kit.btnLg} ${kit.soon}`}>
            <ClaudeMark />
            Plugin coming soon
          </span>
        </div>
        <p className={styles.fine}>Free. No account. Runs on your Mac.</p>

        <div className={styles.switch} role="tablist" aria-label="Two ways to use illumen">
          <button
            type="button"
            role="tab"
            id="tab-app"
            aria-selected={!plug}
            aria-controls="app"
            tabIndex={plug ? -1 : 0}
            onClick={() => onView("app")}
            onKeyDown={(e) => onKey(e, "app")}
          >
            <Icon name="laptop" />
            The illumen app
          </button>
          <button
            type="button"
            role="tab"
            id="tab-plugin"
            aria-selected={plug}
            aria-controls="view-plugin"
            tabIndex={plug ? 0 : -1}
            onClick={() => onView("plugin")}
            onKeyDown={(e) => onKey(e, "plugin")}
          >
            <ClaudeMark />
            In Claude Code
          </button>
        </div>
        <p className={styles.switchNote} id="switch-note">
          {SWITCH_NOTE[view]}
        </p>
        <figure className={styles.shot} aria-label="illumen, drawn and playable. Try the buttons under it.">
          <div className={plug ? `${styles.win} ${styles.isTerm}` : styles.win} id="win">
            <div className={styles.winBar}>
              <i style={{ "--c": "var(--tl-r)" } as CSSProperties} aria-hidden="true" />
              <i style={{ "--c": "var(--tl-y)" } as CSSProperties} aria-hidden="true" />
              <i style={{ "--c": "var(--tl-g)" } as CSSProperties} aria-hidden="true" />
              <span aria-hidden="true">{title}</span>
              {replay}
            </div>
            {window}
          </div>
        </figure>
        {tryRow}
      </div>
    </section>
  );
}
