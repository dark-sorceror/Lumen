"use client";

import {
  type FormEvent,
  type KeyboardEvent as ReactKeyEvent,
  type Ref,
  type RefObject,
  useCallback,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import kit from "@/app/(landing)/page.module.css";
import { AppIcon, cx, dashFor, Ring } from "./AppParts";
import { copyText } from "./CopyButton";
import { reducedMotion } from "./motion";
import {
  answer,
  complete,
  END_PCT,
  FORGOT,
  forgotSaid,
  LEFT_OUT,
  matches,
  moveHighlight,
  NOTE_TEXT,
  resolve,
  SCRIPT,
  sessionAt,
  showsMenu,
  statusText,
  TYPE_MS,
} from "./terminal";
import TerminalLog, { type Output } from "./TerminalLog";
import TerminalPane from "./TerminalPane";
import { useToast } from "./Toast";
import TryRow from "./TryRow";
import { useInView } from "./useInView";
import styles from "./TerminalDrawing.module.css";

// The hero's second view: Claude Code in a pitch-black terminal with the
// illumen plugin, one made-up session in ~/invoice-service. It plays once on
// screen: /compact, the summary, the plugin's seam row, its note and its
// status line; then a message, three tools and the reply. Its prompt answers
// the plugin's two commands (a menu lists them once you type "/"), and its
// small pane says how full it is and what it forgot. No model runs here.

/** What the window bar and the try row can ask of the drawing. */
export type TerminalDrawingHandle = {
  /** Jumps the session to its end if it's playing. */
  skip(): void;
  /** Plays the session again from the start, dropping what you ran. */
  replay(): void;
  /** Runs a command at the prompt, as if typed and sent. */
  run(command: string): void;
  /** Opens the plugin's pane (lights it if it's open); Esc closes it and
   * focus goes back to `opener`. */
  openPane(opener?: HTMLElement | null): void;
};

/** How long the plugin's note stays, in ms. */
const NOTE_MS = 6000;
/** How long "[ Copied ]" stays, in ms. */
const COPIED_MS = 1400;
/** The pane opens by itself only on a terminal this wide; narrower, it would
 * cover the transcript. */
const PANE_MIN = 700;

export default function TerminalDrawing({
  ref,
  hidden = false,
  onPlayingChange,
}: {
  ref?: Ref<TerminalDrawingHandle>;
  hidden?: boolean;
  /** Called when the session starts or stops playing, for the window bar's Skip / Replay. */
  onPlayingChange?: (playing: boolean) => void;
}) {
  const say = useToast();
  const rootRef = useRef<HTMLDivElement>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const paneRef = useRef<HTMLDivElement>(null);
  const fieldRef = useRef<HTMLInputElement>(null);
  const probeRef = useRef<HTMLElement>(null);
  // One column, in px, measured over a hundred of them (one alone is
  // rounded to 1/64 px, which drifts a pixel every hundred or so columns).
  const chPx = useRef(0);
  // Where focus goes back when Esc closes the pane.
  const opener = useRef<HTMLElement | null>(null);
  const asked = useRef(0);
  // Something was put in the prompt: the caret goes to its end.
  const caretToEnd = useRef(false);
  const copyTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  // The frame that says a command's answer, after the status line is emptied.
  const sayFrame = useRef(0);
  const onPlaying = useRef(onPlayingChange);
  useEffect(() => {
    onPlaying.current = onPlayingChange;
  });
  const seen = useInView(rootRef, { threshold: 0.3 });

  // How far the script is. The server and the first client render draw the
  // finished session (as a thumbnail or a no-JS reader sees it); on the
  // client, with motion allowed, it goes back to the start and plays the
  // first time it's seen.
  const [at, setAt] = useState(SCRIPT.length);
  // How many letters of the step's typing are in the prompt.
  const [typed, setTyped] = useState(0);
  const [started, setStarted] = useState(false);
  const [draft, setDraft] = useState("");
  // Which match in the menu is lit (Enter runs it, Tab completes it).
  const [lit, setLit] = useState(0);
  const [outputs, setOutputs] = useState<Output[]>([]);
  // What a screen reader hears after a command.
  const [said, setSaid] = useState("");
  const [toastGone, setToastGone] = useState(false);
  const [seamOpen, setSeamOpen] = useState(false);
  // The gone messages you put back in your prompt, by index.
  const [putBack, setPutBack] = useState<number[]>([]);
  const [forgotSeen, setForgotSeen] = useState(false);
  const [paneOpen, setPaneOpen] = useState(false);
  const [flash, setFlash] = useState(0);
  const [copied, setCopied] = useState<number | null>(null);
  // The block cursor: the caret's column, how far the field has scrolled,
  // whether the field has focus (hollow when it hasn't), and whether the
  // caret's column is in the field's visible width.
  const [cursor, setCursor] = useState({ pos: 0, scroll: 0, focused: false, inView: true });

  const done = at >= SCRIPT.length;
  const s = sessionAt(at);
  const step = done ? null : SCRIPT[at];
  const typing = step?.type !== undefined && typed > 0 ? step.type.slice(0, typed) : null;
  const last = s.lines.length ? s.lines[s.lines.length - 1].line : null;
  // A step that sends text holds it in the prompt while it waits.
  const sending = step?.send ?? null;
  const waiting =
    step !== null &&
    step.type === undefined &&
    step.send === undefined &&
    step.line !== undefined &&
    step.line.kind !== "forgot" &&
    step.line.kind !== "text" &&
    last !== null &&
    last.kind !== "done" &&
    last.kind !== "busy" &&
    last.kind !== "ask";
  const noteShown = s.toast && !toastGone && !s.forgotSeen;
  const placeholder = done ? `Try ${FORGOT}` : "";
  const shown = sending ?? draft;
  const menu = showsMenu(shown);
  const found = matches(shown);
  const litAt = Math.min(lit, Math.max(found.length - 1, 0));
  const wide = () => (rootRef.current?.clientWidth ?? 0) >= PANE_MIN;

  // With motion allowed the session goes back to its start, to play when
  // seen; with reduced motion it just stays at its end.
  useLayoutEffect(() => {
    if (reducedMotion()) setAt(SCRIPT.length);
    else setAt(0);
  }, []);

  // Seen for the first time, in its view: it plays, once.
  useEffect(() => {
    if (seen && !hidden) setStarted(true);
  }, [seen, hidden]);

  // One step at a time, a letter at a time while it types, and only while
  // it's on screen in its view.
  useEffect(() => {
    if (!started || !seen || hidden || done) return;
    const st = SCRIPT[at];
    if (st.type !== undefined && typed < st.type.length) {
      const t = setTimeout(() => setTyped((n) => n + 1), typed === 0 ? st.wait : TYPE_MS);
      return () => clearTimeout(t);
    }
    const t = setTimeout(
      () => {
        setAt(at + 1);
        setTyped(0);
      },
      st.type !== undefined ? 350 : st.wait,
    );
    return () => clearTimeout(t);
  }, [started, seen, hidden, done, at, typed]);

  // The note goes after a few seconds on screen, as a toast does; the line
  // under the prompt keeps what it said.
  useEffect(() => {
    if (!noteShown || !started || hidden) return;
    const t = setTimeout(() => setToastGone(true), NOTE_MS);
    return () => clearTimeout(t);
  }, [noteShown, started, hidden]);

  useEffect(() => {
    onPlaying.current?.(!done);
  }, [done]);

  // Each time the view shows, the pane opens on a wide terminal, measured
  // now that it's laid out, and unlit: only a command lights it. It closes
  // when the view goes.
  useLayoutEffect(() => {
    opener.current = null;
    setFlash(0);
    setPaneOpen(!hidden && (rootRef.current?.clientWidth ?? 0) >= PANE_MIN);
  }, [hidden]);

  // The newest line in view, inside the terminal only (never the page).
  useLayoutEffect(() => {
    const log = logRef.current;
    if (log) log.scrollTop = log.scrollHeight;
  }, [at, outputs, seamOpen, hidden]);

  const placeCursor = useCallback(() => {
    const f = fieldRef.current;
    if (!f) return;
    const hundred = probeRef.current?.getBoundingClientRect().width ?? 0;
    if (hundred > 0) chPx.current = hundred / 100;
    const pos = f.selectionStart ?? f.value.length;
    // Chrome scrolls a field back to its start when it loses focus, so the
    // caret's column can lie past either edge: the block hides there rather
    // than sit on the prompt's border or be cut. A field scrolled to its end
    // puts the caret up to a pixel past its width, where Chrome draws its own.
    const x = pos * chPx.current - f.scrollLeft;
    const next = {
      pos,
      scroll: f.scrollLeft,
      focused: document.activeElement === f,
      inView: chPx.current === 0 || (x > -1 && x < f.clientWidth + 1),
    };
    setCursor((c) =>
      c.pos === next.pos && c.scroll === next.scroll && c.focused === next.focused && c.inView === next.inView
        ? c
        : next,
    );
  }, []);
  // Now, and again once the browser has moved the caret and scrolled the field.
  const followCaret = useCallback(() => {
    placeCursor();
    requestAnimationFrame(placeCursor);
  }, [placeCursor]);

  // The prompt changed without a key (a command ran, something was put back).
  useLayoutEffect(() => {
    const f = fieldRef.current;
    if (f && caretToEnd.current) {
      caretToEnd.current = false;
      f.focus({ preventScroll: true });
      f.setSelectionRange(f.value.length, f.value.length);
      // Chrome doesn't scroll a field to a caret set from script (and focus()
      // does nothing to a field that has focus), so the end is brought in view.
      f.scrollLeft = f.scrollWidth;
    }
    followCaret();
  }, [draft, placeholder, followCaret]);

  // The field changed width (the pane slid in or out, the window changed):
  // Chrome clamped its scroll and moves no caret, so the caret is kept in
  // view while the field has focus, and the block follows it.
  useEffect(() => {
    const f = fieldRef.current;
    if (!f || typeof ResizeObserver === "undefined") return;
    const watch = new ResizeObserver(() => {
      if (document.activeElement === f) {
        const pos = f.selectionStart ?? f.value.length;
        const ch = chPx.current;
        const x = pos * ch - f.scrollLeft;
        if (pos === f.value.length) f.scrollLeft = f.scrollWidth;
        else if (ch && x < 0) f.scrollLeft = pos * ch;
        else if (ch && x + ch > f.clientWidth) f.scrollLeft = pos * ch + ch - f.clientWidth;
      }
      followCaret();
    });
    watch.observe(f);
    return () => watch.disconnect();
  }, [followCaret]);

  const closePane = useCallback((refocus: boolean) => {
    const back = opener.current;
    const inside = paneRef.current?.contains(document.activeElement) ?? false;
    opener.current = null;
    setPaneOpen(false);
    setFlash(0);
    if (!refocus) return;
    if (back?.isConnected) back.focus({ preventScroll: true });
    else if (inside) fieldRef.current?.focus({ preventScroll: true });
  }, []);

  // Esc closes the pane while you're in the hero (or nowhere), and focus
  // goes back to what opened it.
  useEffect(() => {
    if (hidden || !paneOpen) return;
    function onKey(e: KeyboardEvent) {
      const root = rootRef.current;
      if (e.key !== "Escape" || !root || root.closest("[inert]")) return;
      const now = document.activeElement;
      const hero = root.closest("section") ?? root;
      if (!(now === document.body || (now && hero.contains(now)))) return;
      e.preventDefault();
      closePane(true);
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [hidden, paneOpen, closePane]);

  useEffect(
    () => () => {
      clearTimeout(copyTimer.current);
      cancelAnimationFrame(sayFrame.current);
    },
    [],
  );

  /** Jumps to the session's end, if it's still playing (or waiting to). */
  function skip() {
    if (done) return;
    setAt(SCRIPT.length);
    setTyped(0);
    setToastGone(true);
  }

  /** Plays from the start; what you ran and put back is taken out. */
  function replay() {
    setOutputs([]);
    cancelAnimationFrame(sayFrame.current);
    setSaid("");
    setToastGone(false);
    setSeamOpen(false);
    setPutBack([]);
    setForgotSeen(false);
    setFlash(0);
    setCopied(null);
    setDraft("");
    setTyped(0);
    // Nothing travels with reduced motion: the session is simply at its end.
    if (reducedMotion()) setAt(SCRIPT.length);
    else setAt(0);
    setStarted(seen);
    opener.current = null;
    setPaneOpen(wide());
  }

  function openPane(from: HTMLElement | null) {
    if (!paneOpen) {
      opener.current = from;
      setPaneOpen(true);
    }
    setFlash((n) => n + 1);
  }

  /** Puts `texts` in the prompt after what's in it (`before`), to send when ready. */
  function fill(texts: readonly string[], before: string) {
    setDraft([before.trim(), ...texts].filter(Boolean).join(" "));
    caretToEnd.current = true;
  }

  /** Says `text` to a screen reader: the status line is emptied first, then
   * filled a frame later, so the same answer twice is heard twice. */
  function announce(text: string) {
    setSaid("");
    cancelAnimationFrame(sayFrame.current);
    sayFrame.current = requestAnimationFrame(() => setSaid(text));
  }

  function run(command: string) {
    const ask = command.trim();
    if (!ask) return;
    skip();
    const out = answer(ask, { putBack, paneOpen });
    const id = ++asked.current;
    setOutputs((all) => [...all, { id, ask, kind: out.kind, text: out.text }].slice(-3));
    announce(out.text ?? forgotSaid(putBack, sessionAt(SCRIPT.length).restored));
    if (out.forgot) {
      setToastGone(true);
      setForgotSeen(true);
    }
    if (out.back) setPutBack(out.back);
    if (out.pane === "open") openPane(null);
    if (out.pane === "close") closePane(false);
    if (out.fill) fill(out.fill, "");
    else setDraft("");
  }

  function putBackOne(i: number) {
    if (putBack.includes(i)) return;
    setPutBack((all) => [...all, i].sort((a, b) => a - b));
    fill([LEFT_OUT[i]], draft);
  }

  async function copy(i: number) {
    if (!(await copyText(LEFT_OUT[i]))) {
      say("Couldn't copy here.");
      return;
    }
    setCopied(i);
    clearTimeout(copyTimer.current);
    copyTimer.current = setTimeout(() => setCopied(null), COPIED_MS);
  }

  function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    run(resolve(draft, litAt));
  }

  /** Tab completes the lit command and the arrows move the light, while the
   * menu has matches; with no menu the keys are left alone (Tab moves focus). */
  function menuKey(e: ReactKeyEvent<HTMLInputElement>) {
    if (!found.length || e.altKey || e.ctrlKey || e.metaKey) return;
    if (e.key === "Tab" && !e.shiftKey) {
      const name = complete(draft, litAt);
      if (name === null) return;
      e.preventDefault();
      setLit(0);
      setDraft(name);
      caretToEnd.current = true;
    } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      setLit(moveHighlight(litAt, found.length, e.key === "ArrowDown" ? 1 : -1));
    }
  }

  useImperativeHandle(ref, () => ({
    skip,
    replay,
    run,
    openPane: (el) => {
      skip();
      openPane(el ?? null);
    },
  }));

  const field = () => fieldRef.current?.focus();
  const under = shown ? shown[cursor.pos] || " " : placeholder[0] || " ";

  return (
    <div
      ref={rootRef}
      className={cx(styles.tv, paneOpen && styles.paneOpen)}
      id="view-plugin"
      role="tabpanel"
      aria-labelledby="tab-plugin"
      hidden={hidden}
    >
      <div className={styles.ts}>
        <TerminalLog
          logRef={logRef}
          s={s}
          waiting={waiting}
          outputs={outputs}
          seamOpen={seamOpen}
          onSeam={() => setSeamOpen((o) => !o)}
          putBack={putBack}
          restored={s.restored}
          copied={copied}
          onCopy={copy}
          onPutBack={putBackOne}
        />

        {/* Claude Code lists commands above the prompt once you type "/", and only until a space. */}
        <ul className={styles.tCmds} id="t-cmds"
          role="listbox"
          aria-label="Commands that match what you typed"
          hidden={!menu}
        >
          {menu && !found.length && <li role="option" aria-selected="false" aria-disabled="true" className={styles.tDim}>No command starts with {draft.trim()}</li>}
          {found.map((c, i) => (
            <li key={c.name} role="presentation">
              <button
                type="button"
                id={`t-cmd-${i}`}
                role="option"
                aria-selected={i === litAt}
                tabIndex={-1}
                className={cx(i === litAt && styles.on)}
                onClick={() => {
                  run(c.name);
                  field();
                }}
              >
                <span className={styles.tInk}>{c.name}</span>
                <span className={styles.about}>{c.about}</span>
              </button>
            </li>
          ))}
        </ul>

        <form className={styles.tPrompt} autoComplete="off" onSubmit={submit}>
          <span className={styles.tDim} aria-hidden="true">
            &gt;
          </span>
          <span className={styles.tTyped} aria-hidden="true" hidden={typing === null}>
            {typing}
            <i className={styles.tCaret} />
          </span>
          <span className={styles.tIn}>
            <label className={kit.sr} htmlFor="t-field">
              Type at Claude Code&apos;s prompt. Type / for commands.
            </label>
            <input
              ref={fieldRef}
              id="t-field"
              spellCheck={false}
              placeholder={placeholder}
              value={shown}
              onChange={(e) => {
                setDraft(e.target.value);
                setLit(0);
                followCaret();
              }}
              role="combobox"
              aria-expanded={menu}
              aria-autocomplete="list"
              aria-controls="t-cmds"
              aria-activedescendant={found.length ? `t-cmd-${litAt}` : undefined}
              onKeyDown={(e) => {
                menuKey(e);
                followCaret();
              }}
              onKeyUp={followCaret}
              onClick={followCaret}
              onSelect={followCaret}
              onFocus={() => {
                skip();
                followCaret();
              }}
              onBlur={followCaret}
            />
            {/* A steady block over the character at the caret, which it shows inverted. */}
            <i
              className={cx(styles.tCursor, !cursor.focused && styles.idle)}
              aria-hidden="true"
              hidden={typing !== null || sending !== null || !cursor.inView}
              style={{ left: `calc(${cursor.pos}ch - ${cursor.scroll}px)` }}
            >
              {under}
            </i>
            <span className={styles.tMeasure} aria-hidden="true">
              <i ref={probeRef} />
            </span>
          </span>
        </form>

        {/* The plugin's note: a box at the top right, or on a narrow
            transcript one line under the prompt (Claude Code's narrow
            renderer prints it there). One element under a key that never
            changes, outside the prompt: typing never redraws it or plays its
            entrance again, and the width only restyles it. */}
        {noteShown && (
          <div key="note" className={styles.tToast}>
            <span className={styles.noteName}>illumen</span>{" "}
            <span className={styles.noteWords}>
              {NOTE_TEXT}
              <button
                type="button"
                className={styles.tLink}
                onClick={() => {
                  run(FORGOT);
                  field();
                }}
              >
                {FORGOT}
              </button>
            </span>
            <button
              type="button"
              className={styles.x}
              aria-label="Dismiss the note"
              onClick={() => {
                setToastGone(true);
                field();
              }}
            >
              ×
            </button>
          </div>
        )}

        <p className={styles.tStatus}>{statusText(s, { forgotSeen, putBack: putBack.length })}</p>
        <p className={kit.sr} role="status">
          {said}
        </p>
      </div>

      <TerminalPane paneRef={paneRef} s={s} open={paneOpen} flash={flash} onForgot={() => run(FORGOT)} />
    </div>
  );
}

/** The row under the window in this view: the plugin's two commands, and the session again. */
export function TerminalTryRow({ term }: { term: RefObject<TerminalDrawingHandle | null> }) {
  return (
    <TryRow id="try-plugin" label="Try it in the terminal" lead="Try it, or type at the prompt">
      <button type="button" data-tcmd="forgot" onClick={() => term.current?.run(FORGOT)}>
        <AppIcon name="eyeOff" small />
        {FORGOT}
      </button>
      <button type="button" data-tcmd="pane" onClick={(e) => term.current?.openPane(e.currentTarget)}>
        <Ring dash={dashFor(END_PCT)} />
        /illumen
      </button>
      <button type="button" data-tcmd="replay" onClick={() => term.current?.replay()}>
        <AppIcon name="undo" small />
        Play it again
      </button>
    </TryRow>
  );
}
