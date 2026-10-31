"use client";

import { useEffect, useRef } from "react";
import kit from "@/app/(landing)/page.module.css";
import CopyButton from "./CopyButton";

const CLONE = "git clone https://github.com/dark-sorceror/Lumen && cd Lumen && uv sync";
const START = "uv run python -m workbench.dev";

function Cmd({ text, what }: { text: string; what: string }) {
  return (
    <div className={kit.install}>
      <div className={kit.cmd}>
        <span className={kit.cmdP}>$</span>
        <code>{text}</code>
        <CopyButton text={text} what={what} className={`${kit.btn} ${kit.btnSm}`} />
      </div>
    </div>
  );
}

/** What the hosted site says instead of opening a chat: illumen runs on your
 * Mac, and how to put it there. While open, everything marked
 * `data-behind-sheet` is inert, Esc closes it, and focus goes back to
 * `opener`. `copied` says the person's typed words are on the clipboard. */
export default function InstallSheet({
  open,
  onClose,
  copied,
  opener,
}: {
  open: boolean;
  onClose: () => void;
  copied: boolean;
  opener?: HTMLElement | null;
}) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const close = useRef(onClose);
  useEffect(() => {
    close.current = onClose;
  });

  useEffect(() => {
    if (!open) return;
    const behind = [...document.querySelectorAll<HTMLElement>("[data-behind-sheet]")];
    behind.forEach((el) => {
      el.inert = true;
    });
    closeRef.current?.focus();
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") close.current();
    }
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      behind.forEach((el) => {
        el.inert = false;
      });
      opener?.focus({ preventScroll: true });
    };
  }, [open, opener]);

  if (!open) return null;
  return (
    <>
      <div className={kit.scrim} onClick={onClose} />
      <div className={kit.sheet} role="dialog" aria-modal="true" aria-labelledby="sheet-title">
        <h3 id="sheet-title">illumen runs on your Mac</h3>
        <p className={kit.sheetLede}>
          This site is the front page. illumen itself runs on your Mac: install it once, then chat at{" "}
          <span className={kit.mono}>localhost:3000</span>.
        </p>
        <ol>
          <li>
            Get the code and its parts.
            <Cmd text={CLONE} what="the install command" />
          </li>
          <li>
            Start it.
            <Cmd text={START} what="the start command" />
          </li>
          <li>
            Open{" "}
            <a className={kit.mono} href="http://localhost:3000/">
              localhost:3000
            </a>
            .{copied && " Your message is copied; paste it there."}
          </li>
        </ol>
        <div className={kit.acts}>
          <button type="button" ref={closeRef} className={`${kit.btn} ${kit.btnLine}`} onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </>
  );
}
