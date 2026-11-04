"use client";

import {
  type CSSProperties,
  type FormEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
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
import AppAside, { type AsideTab } from "./AppAside";
import { AppIcon, AppSidebar, cx, dashFor, Ring, Room } from "./AppParts";
import { EarlierTurns, type GoneState, LaterTurns, Seam } from "./AppTurns";
import {
  FINAL,
  GONE,
  type GoneId,
  goneWords,
} from "./appChat";
import { closeLayer, type Layer, openLayer, topLayer } from "./layers";
import {
  appScript,
  type Beat,
  draftAt,
  type Phase,
  PUT_BACK,
  playedBack,
  reached,
  ringFor,
  TURNS,
} from "./appPlay";
import { reducedMotion } from "./motion";
import { useToast } from "./Toast";
import TryRow from "./TryRow";
import { useInView } from "./useInView";
import styles from "./AppDrawing.module.css";

// The hero's window, the illumen app: one made-up chat, "Flaky upload test",
// as Claude Code summarises it. It plays once: the summary drops two things
// you typed, Claude Code makes the fix you'd told it not to, illumen shows
// what was dropped and puts it back, and "Undo that." gets the right fix.
// Everything in it answers: the sidebar and pills say what they do, the
// ring opens its island, the seam shows the two things you typed that are
// gone (and puts them back), the newest turn's +16.7k opens what it was made
// of, and the Context panel slides in beside the chat. The composer takes a
// click like the app's, but only the play types into it.

export type { AsideTab };

/** What the try row (and the page) can ask of the drawing. An opener is
 * where focus goes back when the layer it opened closes. */
export type AppDrawingHandle = {
  openIsland(opener?: HTMLElement | null): void;
  showSeam(): void;
  openPeek(opener?: HTMLElement | null): void;
  openAside(tab: AsideTab, opener?: HTMLElement | null): void;
  /** Jumps the chat to its end if it's playing. */
  skip(): void;
  /** Plays the chat again from before the summary, dropping what you added. */
  replay(): void;
};

type LayerId = "island" | "peek" | "aside";
const GONE_TEXT = Object.fromEntries(GONE.map((g) => [g.id, g.text])) as Record<
  GoneId,
  string
>;
// The phases the play spends on the seam: it stays in view, not the end.
const AT_SEAM: Phase[] = ["revealing", "puttingBack", "typingB"];

const GLOW_MS = 1800;

// The ribbon down the chat's left: how full it was as each message went.
const RIBBON = [
  {
    w: "7px",
    title: "Your 1st message: 9% full when it went",
    sr: "Message 1, 9% full",
  },
  {
    w: "26px",
    title: "Your 2nd message: 33% full when it went",
    sr: "Message 2, 33% full",
  },
  {
    w: "44px",
    title: "Your 3rd message: 55% full when it went",
    sr: "Message 3, 55% full",
  },
];

export default function AppDrawing({
  ref,
  hidden = false,
  onLayerClose,
  onPlayingChange,
}: {
  ref?: Ref<AppDrawingHandle>;
  hidden?: boolean;
  /** Called each time a layer closes, so the try row can let go. */
  onLayerClose?: () => void;
  /** Called when the chat starts or stops playing, for the window bar's Skip / Replay. */
  onPlayingChange?: (playing: boolean) => void;
}) {
  const say = useToast();
  const rootRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const seamRef = useRef<HTMLDivElement>(null);
  const seamBtnRef = useRef<HTMLButtonElement>(null);
  const islandRef = useRef<HTMLDivElement>(null);
  const ringRef = useRef<HTMLButtonElement>(null);
  const peekRef = useRef<HTMLDivElement>(null);
  const deltaRef = useRef<HTMLButtonElement>(null);
  const fieldRef = useRef<HTMLTextAreaElement>(null);
  const putRefs = useRef<Partial<Record<GoneId, HTMLButtonElement | null>>>({});

  // The open layers, newest last (state draws them; the ref is the same
  // stack for handlers that run before the next render), and each opener's
  // element by its key.
  const [layers, setLayerState] = useState<Layer[]>([]);
  const layersRef = useRef<Layer[]>([]);
  const openers = useRef(new Map<string, HTMLElement>());
  const onClosed = useRef(onLayerClose);
  useEffect(() => {
    onClosed.current = onLayerClose;
  });

  // Where the chat's last moments are. The server and the first client render
  // draw the finished chat (as a thumbnail or no-JS reader sees it); on the
  // client, with motion allowed, it goes back to "pre" and plays when seen.
  const [phase, setPhaseState] = useState<Phase>("done");
  const phaseRef = useRef<Phase>("done");
  const setPhase = (next: Phase) => {
    phaseRef.current = next;
    setPhaseState(next);
  };
  const [typed, setTyped] = useState(0);
  const [step, setStep] = useState(0);
  const [words, setWords] = useState(0);
  const [press, setPress] = useState<Beat["press"]>(null);
  // Whether things slide in as they arrive (not after a skip).
  const [animated, setAnimated] = useState(false);
  const started = useRef(false);
  const onPlaying = useRef(onPlayingChange);
  useEffect(() => {
    onPlaying.current = onPlayingChange;
  });
  const seen = useInView(rootRef, { threshold: 0.35 });

  const [tab, setTab] = useState<AsideTab>("overview");
  // The play opens the seam, so the finished chat has it open.
  const [seamOpen, setSeamOpen] = useState(true);
  const [flash, setFlash] = useState(0);
  const [flashing, setFlashing] = useState(false);
  // What you put back yourself: it waits in the composer, since nothing
  // sends here. What the play put back follows the phase.
  const [queued, setQueued] = useState<GoneId[]>([]);
  const [glow, setGlow] = useState(false);
  const [follow, setFollow] = useState(0);
  const timers = useRef(new Set<ReturnType<typeof setTimeout>>());
  const glowTimer = useRef<ReturnType<typeof setTimeout>>(undefined);

  const setLayers = useCallback((next: Layer[]) => {
    layersRef.current = next;
    setLayerState(next);
  }, []);

  /** Keeps `el` as an opener, under its data-opener (or id), and gives the key. */
  const remember = useCallback((el?: HTMLElement | null): string | null => {
    if (!el) return null;
    const key = el.dataset.opener || el.id || "opener";
    openers.current.set(key, el);
    return key;
  }, []);

  const open = useCallback(
    (id: LayerId, opener: string | null) =>
      setLayers(openLayer(layersRef.current, id, opener)),
    [setLayers],
  );

  const close = useCallback(
    (id: LayerId, refocus: boolean) => {
      const layer = layersRef.current.find((l) => l.id === id);
      if (!layer) return;
      setLayers(closeLayer(layersRef.current, id));
      onClosed.current?.();
      if (refocus && layer.opener)
        openers.current.get(layer.opener)?.focus({ preventScroll: true });
    },
    [setLayers],
  );

  const openerOf = (id: LayerId) =>
    layersRef.current.find((l) => l.id === id)?.opener ?? null;
  const isOpen = (id: LayerId) => layers.some((l) => l.id === id);

  function toBottom() {
    const s = scrollRef.current;
    s?.scrollTo({
      top: s.scrollHeight,
      behavior: reducedMotion() ? "auto" : "smooth",
    });
  }

  // A layer is hidden until the render that opens it commits, so focus
  // moves in from an effect, to the layer's first control.
  const focusPending = useRef<"island" | "peek" | null>(null);
  useEffect(() => {
    const id = focusPending.current;
    if (!id || !layers.some((l) => l.id === id)) return;
    focusPending.current = null;
    (id === "island" ? islandRef : peekRef).current
      ?.querySelector<HTMLElement>("button")
      ?.focus({ preventScroll: true });
  }, [layers]);

  function openIsland(opener: string | null) {
    close("peek", false);
    open("island", opener);
    focusPending.current = "island";
  }

  function openPeek(opener: string | null) {
    close("island", false);
    open("peek", opener);
    focusPending.current = "peek";
    toBottom();
  }

  function openAside(next: AsideTab, opener: string | null) {
    close("island", false);
    close("peek", false);
    setTab(next);
    open("aside", opener);
  }

  function showSeam() {
    close("island", false);
    close("peek", false);
    setSeamOpen(true);
    setFlash((n) => n + 1);
    setFlashing(true);
  }

  /** Jumps to the chat's end, if it's still playing (or waiting to). */
  function skip() {
    if (phaseRef.current === "done") return;
    started.current = true;
    timers.current.forEach(clearTimeout);
    timers.current.clear();
    setAnimated(false);
    setTyped(0);
    setWords(0);
    setPress(null);
    setSeamOpen(true);
    setPhase("done");
    setFollow((n) => n + 1);
  }

  /** Plays from before the summary; what you added is taken out. */
  function replay() {
    started.current = true;
    timers.current.forEach(clearTimeout);
    timers.current.clear();
    clearTimeout(glowTimer.current);
    // Through close(), so the try row lets go of its pressed button.
    for (const layer of [...layersRef.current]) close(layer.id as LayerId, false);
    setGlow(false);
    setFlashing(false);
    setQueued([]);
    setTyped(0);
    setWords(0);
    setPress(null);
    if (reducedMotion()) {
      // Nothing travels: the chat is simply at its end, minus what you added.
      setAnimated(false);
      setSeamOpen(true);
      setPhase("done");
      return;
    }
    setAnimated(true);
    setSeamOpen(false);
    setStep(0);
    setPhase("pre");
    for (const beat of appScript()) later(() => play(beat), beat.at);
  }

  /** One beat of the play: what changes at that moment. */
  function play(beat: Beat) {
    if (beat.typed !== undefined) setTyped(beat.typed);
    if (beat.step !== undefined) setStep(beat.step);
    if (beat.words !== undefined) setWords(beat.words);
    if (beat.press !== undefined) setPress(beat.press);
    if (beat.reveal) showSeam();
    if (beat.phase) setPhase(beat.phase);
  }

  useImperativeHandle(ref, () => ({
    openIsland: (el) => openIsland(remember(el) ?? remember(ringRef.current)),
    showSeam,
    openPeek: (el) => openPeek(remember(el) ?? remember(deltaRef.current)),
    openAside: (next, el) => openAside(next, remember(el)),
    skip,
    replay,
  }));

  // With motion allowed the chat goes back to before its summary, to play
  // when seen; with reduced motion it just stays at its end.
  useLayoutEffect(() => {
    if (reducedMotion()) {
      started.current = true;
      setPhase("done");
    } else {
      setPhase("pre");
    }
  }, []);

  // Seen for the first time: a beat, then it plays, once.
  useEffect(() => {
    if (!seen || started.current || phaseRef.current !== "pre") return;
    started.current = true;
    later(() => {
      if (rootRef.current?.hidden) started.current = false;
      else replay();
    }, 700);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seen]);

  useEffect(() => {
    onPlaying.current?.(phase !== "done");
  }, [phase]);

  // Each new moment of the play stays in view; while illumen shows what was
  // dropped, that's the seam.
  useEffect(() => {
    if (animated && phase !== "pre" && !AT_SEAM.includes(phase)) toBottom();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [animated, phase, step]);

  // The chat opens at its end, as one you come back to does.
  useEffect(() => {
    const s = scrollRef.current;
    if (s) s.scrollTop = s.scrollHeight;
  }, []);

  // Esc closes the top layer only, while you're in the hero (or nowhere),
  // and focus goes back to what opened it. A press outside the island or the
  // peek closes it; its own button and the try row toggle it instead.
  useEffect(() => {
    if (hidden) return;
    function onKey(e: KeyboardEvent) {
      const root = rootRef.current;
      if (e.key !== "Escape" || !root || root.closest("[inert]")) return;
      const top = topLayer(layersRef.current);
      const at = document.activeElement;
      const hero = root.closest("section") ?? root;
      if (!top || !(at === document.body || (at && hero.contains(at)))) return;
      e.preventDefault();
      close(top.id as LayerId, true);
    }
    function onDown(e: PointerEvent) {
      const t = e.target;
      if (!(t instanceof Element) || t.closest("[data-try]")) return;
      if (!islandRef.current?.contains(t) && !ringRef.current?.contains(t))
        close("island", false);
      if (!peekRef.current?.contains(t) && !deltaRef.current?.contains(t))
        close("peek", false);
    }
    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onDown);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("pointerdown", onDown);
    };
  }, [hidden, close]);

  // "See which" and the try row bring the seam up near the top, lit a moment.
  useEffect(() => {
    const s = scrollRef.current;
    const seam = seamRef.current;
    if (!flash || !s || !seam) return;
    s.scrollTo({
      top: seam.offsetTop - 60,
      behavior: reducedMotion() ? "auto" : "smooth",
    });
  }, [flash]);

  // Skipping the playback brings the chat's end into view.
  useEffect(() => {
    if (follow) toBottom();
  }, [follow]);

  // Only the playback types here; you can't (it's a drawing).
  const shownDraft = draftAt(phase, typed);

  // The field grows with what you type, up to five lines.
  useLayoutEffect(() => {
    const f = fieldRef.current;
    if (!f) return;
    f.style.height = "auto";
    f.style.height = `${Math.min(f.scrollHeight, 120)}px`;
  }, [shownDraft]);

  useEffect(() => {
    const pending = timers.current;
    return () => {
      pending.forEach(clearTimeout);
      clearTimeout(glowTimer.current);
    };
  }, []);

  /** Where a gone message is: what the play did with it (its Put back lands
   * when the press lets go), else what you did. */
  function goneState(id: GoneId): GoneState {
    const played = playedBack(phase, id);
    if (played === "sent") return "sent";
    if (played === "queued" && press !== "putBack") return "queued";
    return queued.includes(id) ? "queued" : "gone";
  }
  const ids = GONE.map((g) => g.id);
  const chips = ids.filter((id) => goneState(id) === "queued");
  const forgotten = ids.filter((id) => goneState(id) !== "sent");

  function putBack(id: GoneId) {
    if (goneState(id) !== "gone") return;
    setQueued((q) => [...q, id]);
    if (reducedMotion()) return;
    setGlow(true);
    clearTimeout(glowTimer.current);
    glowTimer.current = setTimeout(() => setGlow(false), GLOW_MS);
  }

  function takeOut(id: GoneId) {
    setQueued((q) => q.filter((x) => x !== id));
    const button = putRefs.current[id];
    if (button?.offsetParent) button.focus({ preventScroll: true });
    else fieldRef.current?.focus();
  }

  function later(fn: () => void, ms: number) {
    const t = setTimeout(() => {
      timers.current.delete(t);
      fn();
    }, ms);
    timers.current.add(t);
  }

  // The composer is read-only: a click focuses it, but nothing typed lands
  // and send sends nothing.
  function sendOff(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
  }

  const islandOpen = isOpen("island");
  const peekOpen = isOpen("peek");
  const asideOpen = isOpen("aside");
  const working =
    glow ||
    phase === "summarising" ||
    phase === "puttingBack" ||
    TURNS.some((t) => t.at.working === phase);
  const ring = ringFor(phase);
  const playing = phase !== "done";
  // Touching the app while it plays takes it to the end first. A press on
  // the seam's button goes on to open it, as it would a closed seam.
  const interrupt = (e: ReactPointerEvent) => {
    skip();
    if (e.target instanceof Node && seamBtnRef.current?.contains(e.target))
      setSeamOpen(false);
  };
  const toSees = (el: HTMLElement) => openAside("sees", remember(el));

  return (
    <div
      ref={rootRef}
      className={cx(styles.app, asideOpen && styles.asideOpen)}
      id="app"
      role="tabpanel"
      aria-labelledby="tab-app"
      hidden={hidden}
      onPointerDownCapture={playing ? interrupt : undefined}
      onFocusCapture={playing ? skip : undefined}
    >
      <AppSidebar say={say} />

      <div className={styles.main} role="group" aria-label="Chat">
        <div className={styles.top}>
          <button
            type="button"
            className={kit.btn}
            onClick={() => say("This chat: its folder, rename and delete.")}
          >
            <span className={styles.title}>Flaky upload test</span>
            <AppIcon name="down" small />
          </button>
          <span className={styles.grow} />
          <button
            type="button"
            className={kit.btn}
            data-opener="files"
            onClick={(e) => toSees(e.currentTarget)}
          >
            <AppIcon name="files" />
            Files <span className={kit.num}>5</span>
          </button>
        </div>

        <div
          className={styles.ribbon}
          role="group"
          aria-label="Context, message by message"
        >
          <span className={cx(kit.label, styles.ribbonLabel)}>Context</span>
          {RIBBON.map((bar) => (
            <span
              key={bar.sr}
              className={styles.rb}
              style={{ "--w": bar.w } as CSSProperties}
              title={bar.title}
            >
              <span className={kit.sr}>{bar.sr}</span>
            </span>
          ))}
          {reached(phase, "summarised") && (
            <div
              className={styles.rbCut}
              title="Claude Code summarised here, at 86% full"
            >
              <i />
              <span>summarised</span>
            </div>
          )}
          {TURNS.map((turn, i) => {
            if (!reached(phase, turn.at.sent)) return null;
            // How full it was as this one went, and whether one went since.
            const pct = ringFor(turn.at.sent).pct;
            const next = TURNS[i + 1];
            const newest = !next || !reached(phase, next.at.sent);
            const n = RIBBON.length + i + 1;
            return (
              <span
                key={turn.typed}
                className={cx(styles.rb, newest && styles.lit)}
                style={
                  { "--w": `${Math.round(pct * 0.8)}px` } as CSSProperties
                }
                title={`Your ${newest ? "newest" : `${n}th`} message: ${pct}% full when it went`}
              >
                <span className={kit.sr}>
                  {newest ? "Newest message" : `Message ${n}`}, {pct}% full
                </span>
              </span>
            );
          })}
        </div>

        <div className={styles.scroll} ref={scrollRef}>
          <div className={styles.col}>
            <EarlierTurns
              onDrew={toSees}
              say={say}
              pre={!reached(phase, "summarised")}
            />

            {phase === "summarising" && (
              <div className={cx(styles.live, styles.liveSum, styles.enter)}>
                <span className={styles.spin} aria-hidden="true" />
                <span>Claude Code is summarising the chat to keep going…</span>
              </div>
            )}

            {reached(phase, "summarised") && (
              <Seam
                enter={animated && phase !== "done"}
                seamRef={seamRef}
                buttonRef={seamBtnRef}
                open={seamOpen}
                onToggle={() => setSeamOpen((o) => !o)}
                flash={flash}
                flashing={flashing}
                onFlashed={() => setFlashing(false)}
                stateOf={goneState}
                putRef={(id, el) => {
                  putRefs.current[id] = el;
                }}
                onPutBack={putBack}
                marked={phase === "revealing" ? PUT_BACK : null}
                pressed={press === "putBack" ? PUT_BACK : null}
              />
            )}

            <LaterTurns
              peekOpen={peekOpen}
              deltaRef={deltaRef}
              peekRef={peekRef}
              onDrew={toSees}
              onDelta={(el) =>
                peekOpen ? close("peek", false) : openPeek(remember(el))
              }
              onClosePeek={() => close("peek", true)}
              onDetails={() =>
                openAside("sees", openerOf("peek") ?? remember(ringRef.current))
              }
              say={say}
              phase={phase}
              step={step}
              words={words}
              enter={animated}
            />
          </div>
        </div>

        <div className={styles.foot}>
          <div className={styles.cwrap}>
            <div
              ref={islandRef}
              className={styles.island}
              id="app-island"
              role="dialog"
              aria-label="Context"
              hidden={!islandOpen}
            >
              <div className={styles.popHead}>
                <span className={kit.label}>Context · Opus 5.5</span>
                <button
                  type="button"
                  className={cx(kit.btn, kit.btnIcon, styles.popClose)}
                  aria-label="Close"
                  onClick={() => close("island", true)}
                >
                  <AppIcon name="close" small />
                </button>
              </div>
              <div className={styles.big}>
                <b className={kit.num}>{FINAL.pct}%</b>
                <span>
                  full · {FINAL.used} of {FINAL.room}
                </span>
              </div>
              <Room />
              <div className={styles.callout}>
                <span className={kit.goneN}>{goneWords(ring.gone)}</span>
                <button
                  type="button"
                  className={cx(kit.btn, kit.btnSm, styles.calloutBtn)}
                  onClick={() => {
                    showSeam();
                    seamBtnRef.current?.focus();
                  }}
                >
                  See which
                </button>
              </div>
              <div className={styles.popFoot}>
                <button
                  type="button"
                  className={cx(kit.btn, kit.btnSm)}
                  onClick={() =>
                    openAside(
                      "overview",
                      openerOf("island") ?? remember(ringRef.current),
                    )
                  }
                >
                  Open Context
                  <AppIcon name="right" small />
                </button>
              </div>
            </div>

            <form
              className={cx(styles.composer, working && styles.working)}
              autoComplete="off"
              onSubmit={sendOff}
            >
              <div className={styles.cChips}>
                {chips.map((id) => (
                  <span key={id} className={cx(kit.chip, styles.putChip)}>
                    <AppIcon name="putback" small />
                    <span className={styles.chipText}>{GONE_TEXT[id]}</span>
                    <button
                      type="button"
                      className={styles.chipX}
                      aria-label="Take it out again"
                      onClick={() => takeOut(id)}
                    >
                      <AppIcon name="close" small />
                    </button>
                  </span>
                ))}
              </div>
              <label className={kit.sr} htmlFor="app-field">
                Message Claude Code
              </label>
              <textarea
                ref={fieldRef}
                className={styles.field}
                id="app-field"
                rows={1}
                placeholder="Message Claude Code"
                value={shownDraft}
                readOnly
                aria-readonly="true"
              />
              <div className={styles.cRow}>
                <button
                  type="button"
                  className={cx(kit.btn, kit.btnIcon)}
                  aria-label="Attach"
                  onClick={() =>
                    say(
                      "Attach a file or an image. It goes into the chat's folder for Claude Code to open.",
                    )
                  }
                >
                  <AppIcon name="plus" />
                </button>
                <button
                  type="button"
                  ref={ringRef}
                  className={cx(kit.pill, styles.inst)}
                  data-opener="ring"
                  aria-expanded={islandOpen}
                  aria-controls="app-island"
                  aria-label={`Context: ${ring.pct}% full${ring.gone ? `, ${goneWords(ring.gone)}` : ""}`}
                  onClick={(e) =>
                    islandOpen
                      ? close("island", false)
                      : openIsland(remember(e.currentTarget))
                  }
                >
                  <Ring dash={dashFor(ring.pct)} large />
                  <span className={kit.num}>{ring.pct}%</span>
                  <span className={styles.full}>full</span>
                  {ring.gone > 0 && (
                    <>
                      <span className={styles.sep} aria-hidden="true" />
                      <span className={kit.goneN}>{ring.gone} gone</span>
                    </>
                  )}
                </button>
                <button
                  type="button"
                  className={kit.pill}
                  data-opener="agents"
                  aria-label="3 helpers in this chat"
                  onClick={(e) =>
                    openAside("agents", remember(e.currentTarget))
                  }
                >
                  <AppIcon name="agents" small />
                  <span className={kit.num}>3</span>
                </button>
                <span className={styles.grow} />
                <button
                  type="button"
                  className={kit.pill}
                  onClick={() =>
                    say(
                      "Pick any model Claude Code offers, and how hard it thinks.",
                    )
                  }
                >
                  Opus 5.5<span className={styles.long}>&nbsp;· High</span>
                  <AppIcon name="down" small />
                </button>
                <button
                  type="submit"
                  className={cx(styles.send, press === "send" && styles.press)}
                  aria-label="Send"
                  aria-disabled="true"
                >
                  <AppIcon name="up" />
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>

      <AppAside
        open={asideOpen}
        forgotten={forgotten}
        tab={tab}
        onTab={setTab}
        onClose={() => close("aside", true)}
      />
    </div>
  );
}

export type AppTry = "full" | "forgot" | "drew" | "helpers";

/** The row under the window: four ways into the drawing. Each one presses
 * while the layer it opened is open (`pressed`, which the page lets go of
 * when a layer closes). */
export function AppTryRow({
  app,
  pressed,
  onPress,
}: {
  app: RefObject<AppDrawingHandle | null>;
  pressed: AppTry | null;
  onPress: (tried: AppTry | null) => void;
}) {
  function go(tried: AppTry, from: HTMLElement) {
    const drawing = app.current;
    if (!drawing) return;
    drawing.skip();
    if (tried === "full") drawing.openIsland(from);
    else if (tried === "forgot") drawing.showSeam();
    else if (tried === "drew") drawing.openPeek(from);
    else drawing.openAside("agents", from);
    onPress(tried === "forgot" ? null : tried);
  }

  const button = (tried: AppTry, icon: ReactNode, label: string) => (
    <button
      type="button"
      data-try={tried}
      data-opener={`try-${tried}`}
      aria-pressed={pressed === tried}
      onClick={(e) => go(tried, e.currentTarget)}
    >
      {icon}
      {label}
    </button>
  );

  return (
    <TryRow id="try-app" label="Try it in the app" lead="Try it in the window">
      {button("full", <Ring dash={dashFor(FINAL.pct)} />, "How full it is")}
      {button("forgot", <AppIcon name="eyeOff" small />, "What it forgot")}
      {button("drew", <AppIcon name="file" small />, "What it drew on")}
      {button(
        "helpers",
        <AppIcon name="agents" small />,
        "What each helper got",
      )}
    </TryRow>
  );
}
