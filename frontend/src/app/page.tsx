"use client";

import { useEffect, useRef, useState } from "react";
import { useChatSocket } from "@/hooks/useChatSocket";
import ConnectionDot from "@/components/ConnectionDot";
import MessageList from "@/components/MessageList";
import Timeline from "@/components/Timeline";
import Composer from "@/components/Composer";
import ContextPanel from "@/components/ContextPanel";
import Sidebar from "@/components/Sidebar";
import DisconnectedBanner from "@/components/DisconnectedBanner";
import styles from "./page.module.css";

export default function Home() {
  const {
    messages,
    streaming,
    connected,
    send,
    stop,
    context,
    cacheImpact,
    editError,
    getContext,
    previewEdit,
    applyEdit,
    stats,
  } = useChatSocket();
  const isEmpty = messages.length === 0;
  const [contextOpen, setContextOpen] = useState(false);

  // Ref to the actual scrolling element (.scroll). Owned here rather than
  // inside MessageList so the "jump to bottom" button -- which lives in
  // page.tsx alongside .scrollArea -- can drive the same element that
  // MessageList's stick-to-bottom logic reads from.
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [showJumpButton, setShowJumpButton] = useState(false);
  // Threshold is intentionally more generous than MessageList's own
  // "still sticking" threshold: the button should only appear once the user
  // has scrolled up a meaningful distance, not the moment they nudge off
  // the very bottom.
  const JUMP_BUTTON_THRESHOLD_PX = 200;

  useEffect(() => {
    const container = scrollRef.current;
    if (!container) return;
    const handleScroll = () => {
      const distance = container.scrollHeight - container.scrollTop - container.clientHeight;
      setShowJumpButton(distance > JUMP_BUTTON_THRESHOLD_PX);
    };
    handleScroll();
    container.addEventListener("scroll", handleScroll, { passive: true });
    return () => container.removeEventListener("scroll", handleScroll);
  }, []);

  // Nothing to jump to (and no scroll position to speak of) once the thread
  // is cleared back to empty.
  useEffect(() => {
    if (isEmpty) setShowJumpButton(false);
  }, [isEmpty]);

  const scrollToBottom = () => {
    const container = scrollRef.current;
    if (!container) return;
    container.scrollTo({ top: container.scrollHeight, behavior: "smooth" });
  };

  // Defensive: messages always start empty on mount today (no persistence
  // layer), so this only matters if that ever changes. If the app somehow
  // mounts with messages already present, skip the empty->bottom animation
  // entirely and start pinned at the bottom immediately (no snap-then-slide).
  const skipInitialAnim = useRef(!isEmpty);
  const [animReady, setAnimReady] = useState(!skipInitialAnim.current);
  useEffect(() => {
    if (!animReady) {
      const id = requestAnimationFrame(() => setAnimReady(true));
      return () => cancelAnimationFrame(id);
    }
  }, [animReady]);

  const statusText = !connected ? "disconnected" : streaming ? "generating…" : "connected";

  return (
    // .shell is a 2-row grid: the navbar spans the full width on
    // row 1, and row 2 holds the (layout-only, non-functional) sidebar next
    // to the chat column -- see Sidebar.tsx for the scaffolding
    // note. The header lives as a direct grid child (not nested inside
    // .page) so it can span both columns above the sidebar.
    <div className={styles.shell}>
      <header className={styles.headerBar}>
        <div className={styles.header}>
          <span className={styles.title}>Lumen</span>
          <div className={styles.navRight}>
            {/* White-box stat readout: input -> output (KV-cache
                reused) tokens plus live tokens/sec for the current/last
                response. Nothing renders until the first gen_stats of a
                session arrives (i.e. idle/no generation yet). */}
            {stats && (
              <span
                className={styles.stats}
                title="Prompt tokens → output tokens, and generation speed"
              >
                {stats.promptTokens.toLocaleString()} &rarr;{" "}
                {stats.outputTokens.toLocaleString()} (
                <span title="KV-cache reused tokens">
                  {stats.cachedTokens.toLocaleString()} cached
                </span>
                ) &middot; {Math.round(stats.tokensPerSec)} tok/s
              </span>
            )}
            <div className={styles.status}>
              <ConnectionDot connected={connected} />
              <span className={styles.statusText}>{statusText}</span>
            </div>
          </div>
        </div>
      </header>
      <Sidebar />
      <div className={styles.page}>
        {/* Floating "server not reachable" toast -- position:
            fixed, so it never participates in .page's flex flow. Rendered
            here purely for tree locality; it paints relative to the
            viewport regardless of where it sits in the DOM. */}
        <DisconnectedBanner connected={connected} />
        <div className={styles.main}>
          <div
            className={`${styles.scrollArea} ${isEmpty ? styles.scrollAreaEmpty : ""} ${!animReady ? styles.noAnim : ""}`}
          >
            {/* Timeline is a SIBLING of .scroll, not a child of it -- it's
                absolutely positioned against .scrollArea (which never
                scrolls), so it stays fixed in place while .scroll's content
                scrolls beneath it. It also overlays the empty gutter to the
                left of MessageList's centered column instead of consuming
                horizontal flow space, so the message/composer column keeps
                its original (pre-Timeline) width. */}
            {!isEmpty && <Timeline messages={messages} />}
            <div className={styles.scroll} ref={scrollRef}>
              {!isEmpty && (
                <MessageList
                  messages={messages}
                  streaming={streaming}
                  scrollRef={scrollRef}
                />
              )}
            </div>
            {showJumpButton && (
              <button
                type="button"
                className={styles.jumpButton}
                onClick={scrollToBottom}
                aria-label="Jump to latest message"
                title="Jump to latest message"
              >
                <svg
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <path d="M6 9l6 6 6-6" />
                </svg>
              </button>
            )}
          </div>
          <div
            className={`${styles.stage} ${isEmpty ? styles.stageEmpty : ""} ${!animReady ? styles.noAnim : ""}`}
          >
            {isEmpty && (
              <div className={styles.greeting}>
                <h2 className={styles.greetingTitle}>What are we exploring?</h2>
                <p className={styles.greetingSubtitle}>
                  Chatting with Qwen3-8B through your own inference engine.
                </p>
              </div>
            )}
            <div className={styles.composerDock}>
              <Composer
                streaming={streaming}
                disabled={!connected}
                onSend={send}
                onStop={stop}
                contextOpen={contextOpen}
                onToggleContext={() => setContextOpen((v) => !v)}
              />
            </div>
          </div>
        </div>
        <ContextPanel
          open={contextOpen}
          onClose={() => setContextOpen(false)}
          segments={context}
          streaming={streaming}
          cacheImpact={cacheImpact}
          editError={editError}
          onRequestContext={getContext}
          onPreviewEdit={previewEdit}
          onApplyEdit={applyEdit}
        />
      </div>
    </div>
  );
}
