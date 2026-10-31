"use client";

import { useEffect, useRef, useState } from "react";
import { Icon } from "./Icons";
import styles from "./CopyButton.module.css";

/** Puts `text` on the clipboard: the Clipboard API where the page may use
 * it, else a hidden field and the browser's own copy command. Whether it
 * worked. Focus goes back where it was. */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // Refused (no permission, an unfocused document): try the old way.
  }
  const before = document.activeElement as HTMLElement | null;
  const field = document.createElement("textarea");
  field.value = text;
  field.setAttribute("readonly", "");
  field.style.position = "fixed";
  field.style.inset = "0 auto auto 0";
  field.style.opacity = "0";
  document.body.appendChild(field);
  try {
    field.select();
    return document.execCommand("copy");
  } catch {
    return false;
  } finally {
    field.remove();
    before?.focus({ preventScroll: true });
  }
}

const SHOWN_MS = 1400;

/** A Copy button (styled by `className`) that says "Copied" for a moment after, and tells a screen
 * reader too. `what` names the thing copied, for its label. */
export default function CopyButton({ text, what, className }: { text: string; what: string; className?: string }) {
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle");
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined);
  useEffect(() => () => clearTimeout(timer.current), []);

  async function copy() {
    const ok = await copyText(text);
    setState(ok ? "copied" : "failed");
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setState("idle"), SHOWN_MS);
  }

  return (
    <>
      <button type="button" className={className} onClick={copy} aria-label={`Copy ${what}`}>
        <Icon name={state === "copied" ? "check" : "copy"} small />
        {state === "copied" ? "Copied" : state === "failed" ? "Couldn't copy" : "Copy"}
      </button>
      <span role="status" className={styles.srOnly}>
        {state === "copied" ? "Copied" : state === "failed" ? "Couldn't copy" : ""}
      </span>
    </>
  );
}
