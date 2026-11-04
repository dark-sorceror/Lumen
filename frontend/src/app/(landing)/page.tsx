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
import Nav from "@/components/landing/Nav";
import PluginStrip from "@/components/landing/PluginStrip";
import Quiet from "@/components/landing/Quiet";
import StorySection from "@/components/landing/StorySection";
import WaysIn from "@/components/landing/WaysInSection";
import { AppIcon } from "@/components/landing/AppParts";
import TerminalDrawing, {
  type TerminalDrawingHandle,
  TerminalTryRow,
} from "@/components/landing/TerminalDrawing";
import { ToastProvider } from "@/components/landing/Toast";
import { useGlideLinks } from "@/components/landing/useGlideLinks";
import { FORWARD_SCRIPT } from "@/lib/landing";
import styles from "./page.module.css";

// The landing page at /: what illumen is, the window it draws, and the ways
// in. It opens no socket; nothing here imports the chat app. Until the app
// and the plugin are out, every way into them says "Coming soon".

const TITLES: Record<HeroView, string> = {
  app: "illumen",
  plugin: "invoice-service — claude",
};

export default function Landing() {
  const [view, setView] = useState<HeroView>("app");
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

  return (
    <div className={styles.page} data-scroller>
      {/* First, before anything is drawn: an old /#c=<id> goes on to /chat/. */}
      <script dangerouslySetInnerHTML={{ __html: FORWARD_SCRIPT }} />
      <ToastProvider>
        <Nav />
        <main id="top">
          <Hero
            view={view}
            onView={showView}
            title={TITLES[view]}
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
          <Close />
        </main>
        <Footer />
      </ToastProvider>
    </div>
  );
}
