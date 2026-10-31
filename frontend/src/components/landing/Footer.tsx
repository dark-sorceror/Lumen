"use client";

import { useEffect, useRef, useState } from "react";
import kit from "@/app/(landing)/page.module.css";
import IllumenMark from "@/components/IllumenMark";
import Wordmark from "@/components/Wordmark";
import { PLUGIN_INSTALL } from "@/lib/landing";
import { Icon } from "./Icons";
import { reducedMotion } from "./motion";

const INSTALL = PLUGIN_INSTALL[1];

/** The brand and its links, the plugin's install line (coming soon), then
 * the name drawn large, lit from the left as it comes into view. */
export default function Footer() {
  const giant = useRef<HTMLDivElement>(null);
  const [lit, setLit] = useState(false);

  useEffect(() => {
    const el = giant.current;
    if (!el) return;
    if (reducedMotion() || typeof IntersectionObserver === "undefined") {
      setLit(true);
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[entries.length - 1].isIntersecting) {
          setLit(true);
          observer.disconnect();
        }
      },
      { threshold: 0.4 },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <footer className={kit.footBar} aria-label="illumen">
      <div className={`${kit.wrap} ${kit.footGrid}`}>
        <div className={kit.footBrand}>
          <a className={kit.brand} href="#top" aria-label="illumen, back to the top">
            <IllumenMark size={30} />
            <Wordmark className={kit.footWord} />
          </a>
          <p>See what your AI sees: how full its context is, what it forgot, and what each answer drew on.</p>
          <div className={`${kit.install} ${kit.footCmd}`}>
            <div className={kit.cmd}>
              <span className={kit.cmdP}>$</span>
              <code>{INSTALL}</code>
              <span className={`${kit.btn} ${kit.btnSm} ${kit.soon}`}>Coming soon</span>
            </div>
          </div>
        </div>
        <nav className={kit.footCols} aria-label="Footer">
          <div className={kit.footCol}>
            <p className={kit.footH}>Product</p>
            <a href="#forgets">How it works</a>
            <a href="#quiet">What it shows</a>
            <a href="#ways-in">Ways in</a>
            <span className={kit.footSoon}>Start a chat · coming soon</span>
          </div>
          <div className={kit.footCol}>
            <p className={kit.footH}>Claude Code</p>
            <a href="#plugin">The plugin</a>
            <a href="#win" data-show="plugin">
              Try it in the terminal
            </a>
            <a href="#win" data-show="app">
              Try the app
            </a>
          </div>
          <div className={kit.footCol}>
            <p className={kit.footH}>Project</p>
            <a href="https://github.com/dark-sorceror/Lumen" target="_blank" rel="noopener">
              GitHub
              <Icon name="arrow" small />
            </a>
            <a href="https://github.com/dark-sorceror/Lumen/issues" target="_blank" rel="noopener">
              Report a problem
              <Icon name="arrow" small />
            </a>
          </div>
        </nav>
      </div>
      <div ref={giant} className={lit ? `${kit.footGiant} ${kit.lit}` : kit.footGiant} aria-hidden="true">
        <span>illumen</span>
      </div>
      <div className={`${kit.wrap} ${kit.footBase}`}>
        <span>© 2026 illumen</span>
        <span>Runs on your Mac. What you send goes only to the model you choose.</span>
        <a href="#top" className={kit.footUp}>
          Back to top
          <Icon name="up" small />
        </a>
      </div>
    </footer>
  );
}
