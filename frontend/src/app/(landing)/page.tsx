"use client";

import { useCallback, useRef, useState } from "react";
import AppDrawing, {
  type AppDrawingHandle,
  type AppTry,
  AppTryRow,
} from "@/components/landing/AppDrawing";
import Close from "@/components/landing/Close";
import Footer from "@/components/landing/Footer";
import Hero, { type HeroView } from "@/components/landing/Hero";
import InstallSheet from "@/components/landing/InstallSheet";
import Nav from "@/components/landing/Nav";
import PluginStrip from "@/components/landing/PluginStrip";
import Quiet from "@/components/landing/Quiet";
import StorySection from "@/components/landing/StorySection";
import WaysIn from "@/components/landing/WaysInSection";
import { AppIcon } from "@/components/landing/AppParts";
import { copyText } from "@/components/landing/CopyButton";
import { isHosted } from "@/components/landing/served";
import TerminalDrawing, {
  type TerminalDrawingHandle,
  TerminalTryRow,
} from "@/components/landing/TerminalDrawing";
import { ToastProvider } from "@/components/landing/Toast";
import { useGlideLinks } from "@/components/landing/useGlideLinks";
import { FORWARD_SCRIPT, startChat } from "@/lib/landing";
import styles from "./page.module.css";

// The landing page at /: what illumen is, the window it draws, and the ways
// in. It opens no socket; nothing here imports the chat app. "Get started"
// does one of two things: served by the app on your Mac it opens a new chat
// at /chat/; served as the public site it tells you how to install.

const TITLES: Record<HeroView, string> = {
  app: "illumen",
  plugin: "invoice-service — claude",
};

export default function Landing() {
  const [view, setView] = useState<HeroView>("app");
  const [sheet, setSheet] = useState({ open: false, copied: false });
  const opener = useRef<HTMLElement | null>(null);
  const app = useRef<AppDrawingHandle>(null);
  // The try row's button that's pressed: the one whose layer is open.
  const [tried, setTried] = useState<AppTry | null>(null);
  // Whether the app drawing is still playing: its window-bar button skips, then replays.
  const [playing, setPlaying] = useState(false);
  // The same for the terminal, in the plugin view.
  const term = useRef<TerminalDrawingHandle>(null);
  const [termPlaying, setTermPlaying] = useState(false);

  const showView = useCallback((next: HeroView, focus = false) => {
    setView(next);
    if (focus)
      requestAnimationFrame(() =>
        document.getElementById(`tab-${next}`)?.focus(),
      );
  }, []);
  useGlideLinks("header", showView);

  // `words` are what was typed where the person started; the hosted site
  // can't carry them to a chat, so it copies them for pasting.
  const start = useCallback(async (from: HTMLElement | null, words = "") => {
    if (!isHosted(location.hostname)) {
      startChat(words, (path) => location.assign(path));
      return;
    }
    opener.current = from;
    const copied = words.trim() ? await copyText(words) : false;
    setSheet({ open: true, copied });
  }, []);

  return (
    <div className={styles.page} data-scroller>
      {/* First, before anything is drawn: an old /#c=<id> goes on to /chat/. */}
      <script dangerouslySetInnerHTML={{ __html: FORWARD_SCRIPT }} />
      <ToastProvider>
        <Nav onStart={(from) => start(from)} />
        <main id="top" data-behind-sheet>
          <Hero
            view={view}
            onView={showView}
            title={TITLES[view]}
            onStart={(from) => start(from)}
            window={
              <>
                <AppDrawing
                  ref={app}
                  hidden={view !== "app"}
                  onLayerClose={() => setTried(null)}
                  onPlayingChange={setPlaying}
                />
                <TerminalDrawing
                  ref={term}
                  hidden={view !== "plugin"}
                  onPlayingChange={setTermPlaying}
                />
              </>
            }
            replay={
              view === "app" ? (
                <button
                  type="button"
                  aria-label={playing ? "Skip to the end" : "Play it again"}
                  onClick={() =>
                    playing ? app.current?.skip() : app.current?.replay()
                  }
                >
                  <AppIcon name={playing ? "right" : "undo"} small />
                  <span>{playing ? "Skip" : "Replay"}</span>
                </button>
              ) : (
                <button
                  type="button"
                  aria-label={
                    termPlaying
                      ? "Skip to the end of the session"
                      : "Replay the session"
                  }
                  onClick={() =>
                    termPlaying ? term.current?.skip() : term.current?.replay()
                  }
                >
                  <AppIcon name={termPlaying ? "right" : "undo"} small />
                  <span>{termPlaying ? "Skip" : "Replay"}</span>
                </button>
              )
            }
            tryRow={
              view === "app" ? (
                <AppTryRow app={app} pressed={tried} onPress={setTried} />
              ) : (
                <TerminalTryRow term={term} />
              )
            }
          />
          <StorySection />
          <Quiet />
          <WaysIn />
          <PluginStrip />
          <Close onStart={(words, from) => start(from, words)} />
        </main>
        <Footer onStart={(from) => start(from)} />
        <InstallSheet
          open={sheet.open}
          copied={sheet.copied}
          opener={opener.current}
          onClose={() => setSheet({ open: false, copied: false })}
        />
      </ToastProvider>
    </div>
  );
}
