"use client";

import { useEffect, useLayoutEffect, useRef, type RefObject } from "react";
import type { ChatMessage as ChatMessageType, MessageAttachment } from "@/hooks/useChatSocket";
import ChatMessage from "./ChatMessage";
import styles from "./MessageList.module.css";

// Gap left above a new turn's user message when it's pinned to the top of the
// chat area.
const TOP_GAP_PX = 20;
// How close to the end of the real (non-spacer) content the user must scroll,
// under their OWN steam, for follow to re-engage.
const AT_BOTTOM_PX = 8;
// Extra scroll range kept below the pinned position, so the pin never lands
// flush against the end of the scroll range.
const PIN_SLACK_PX = 24;
// A scroll we started ourselves keeps firing `scroll` events as it animates.
// Ignore them for this long so our own scrolling can't be mistaken for the
// user's intent -- the feedback loop at the heart of the old bug.
const SELF_SCROLL_MS = 800;

// What the view should do as content streams in.
//   "pinned"    - a new turn's user message is parked near the top; the reply
//                 grows into the space below it and the view holds still.
//   "following" - stick to the newest content.
//   "free"      - the user is reading something older; never move the view.
type FollowMode = "pinned" | "following" | "free";

type Props = {
  messages: ChatMessageType[];
  streaming: boolean;
  scrollRef: RefObject<HTMLDivElement | null>;
  onOpenAttachment: (attachment: MessageAttachment) => void;
};

function lastUserIndex(messages: ChatMessageType[]): number {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user") return i;
  }
  return -1;
}

export default function MessageList({ messages, streaming, scrollRef, onOpenAttachment }: Props) {
  const bottomRef = useRef<HTMLDivElement | null>(null);
  // Self-sizing tail spacer (imperatively sized, no state -> no re-render
  // races): sits AFTER `bottomRef` so it never counts as message content, and
  // grows just enough that the newest turn's user message can be scrolled all
  // the way to the top even when the reply is short. Shrinks back to 0 as the
  // reply grows tall enough to fill the viewport on its own.
  const tailRef = useRef<HTMLDivElement | null>(null);
  // Last spacer height we applied, so the natural (spacer-less) content height
  // can be derived by subtraction WITHOUT resetting the spacer to 0 to measure
  // -- that reset would transiently shrink the scroll range and clamp
  // scrollTop, making the pinned message drift on every streamed token.
  const tailHRef = useRef(0);
  const modeRef = useRef<FollowMode>("following");
  const prevLastUserRef = useRef(-1);
  // Timestamp until which scroll events are assumed to be ours, not the user's.
  const selfScrollUntilRef = useRef(0);

  // Intent tracking. Deliberately NOT position-derived: the tail spacer means
  // "near the bottom of the scroll range" says nothing about whether the user
  // is looking at the newest content, so only a real gesture (wheel, touch
  // drag, paging key) may release follow. Position is consulted for one thing
  // only -- the user scrolling themselves back to the true bottom, which
  // re-engages it.
  useEffect(() => {
    const container = scrollRef.current;
    if (!container) return;

    const isSelfScroll = () => performance.now() < selfScrollUntilRef.current;
    const release = () => {
      modeRef.current = "free";
      // A user gesture cancels an in-flight smooth scroll in the browser, so
      // drop the guard immediately rather than swallowing their next events.
      selfScrollUntilRef.current = 0;
    };

    const handleScroll = () => {
      if (isSelfScroll()) return;
      // Only a released view re-engages by position. While "pinned" the view
      // is deliberately parked with reserved space below it, so a position
      // test would instantly undo the pin -- that mode ends via a user gesture
      // or via the reply outgrowing the viewport, never from a scroll event.
      if (modeRef.current !== "free") return;
      // Measure against the end of REAL content, not the end of the scroll
      // range: the tail spacer means the raw bottom is blank reserved space,
      // and the user should never have to scroll into it to resume following.
      const anchors = container.querySelectorAll('[id^="msg-"]');
      const last = anchors[anchors.length - 1];
      const distance = last
        ? last.getBoundingClientRect().bottom - container.getBoundingClientRect().bottom
        : container.scrollHeight - container.scrollTop - container.clientHeight;
      if (distance <= AT_BOTTOM_PX) modeRef.current = "following";
    };
    const onWheel = (e: WheelEvent) => {
      if (e.deltaY < 0) release();
    };
    let touchY = 0;
    const onTouchStart = (e: TouchEvent) => {
      touchY = e.touches[0]?.clientY ?? 0;
    };
    const onTouchMove = (e: TouchEvent) => {
      const y = e.touches[0]?.clientY ?? 0;
      // Finger moving down the screen == content moving down == scrolling up.
      if (y > touchY + 2) release();
      touchY = y;
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "PageUp" || e.key === "Home" || e.key === "ArrowUp") release();
    };

    handleScroll();
    container.addEventListener("scroll", handleScroll, { passive: true });
    container.addEventListener("wheel", onWheel, { passive: true });
    container.addEventListener("touchstart", onTouchStart, { passive: true });
    container.addEventListener("touchmove", onTouchMove, { passive: true });
    container.addEventListener("keydown", onKeyDown);
    return () => {
      container.removeEventListener("scroll", handleScroll);
      container.removeEventListener("wheel", onWheel);
      container.removeEventListener("touchstart", onTouchStart);
      container.removeEventListener("touchmove", onTouchMove);
      container.removeEventListener("keydown", onKeyDown);
    };
  }, [scrollRef]);

  // Size the tail spacer + drive the scroll, in a layout effect so both land
  // before paint (no flicker). Runs on every message/streaming change -- i.e.
  // every streamed token -- so the spacer continuously shrinks as the reply
  // grows.
  useLayoutEffect(() => {
    const container = scrollRef.current;
    const tail = tailRef.current;
    if (!container || !tail) return;

    const markSelfScroll = () => {
      selfScrollUntilRef.current = performance.now() + SELF_SCROLL_MS;
    };

    const userIdx = lastUserIndex(messages);
    // Natural (spacer-less) content height, derived by subtracting the spacer
    // we last applied (never by mutating the DOM to remeasure -- see tailHRef).
    const natural = container.scrollHeight - tailHRef.current;

    let target = 0;
    let haveTarget = false;
    let need = 0;
    if (userIdx !== -1) {
      const userEl = document.getElementById(`msg-${userIdx}`);
      if (userEl) {
        const cTop = container.getBoundingClientRect().top;
        const uTop = userEl.getBoundingClientRect().top;
        // The user message's offset from the top of the scrollable content.
        // Adding scrollTop cancels the scroll out of the rect, so this stays
        // correct even while a smooth scroll is mid-flight.
        const userTopInContent = uTop - cTop + container.scrollTop;
        target = Math.max(0, userTopInContent - TOP_GAP_PX);
        haveTarget = true;
        // Reserve up to a screenful below the newest user message so the reply
        // has somewhere to stream into while the message sits at the top.
        const belowUser = natural - userTopInContent;
        // ...and, independently, guarantee the scroll range actually REACHES
        // `target` with PIN_SLACK_PX to spare. The old code only did the
        // screenful calculation, which could come up a few px short -- landing
        // the pin flush against the end of the scroll range, where the
        // bottom-detection mistook it for "the user is at the bottom" and
        // yanked the view down on the next token.
        const forTarget = target + PIN_SLACK_PX + container.clientHeight - natural;
        need = Math.max(0, container.clientHeight - belowUser, forTarget);
      }
    }
    tail.style.height = `${need}px`;
    tailHRef.current = need;

    const isNewUserTurn = userIdx !== -1 && userIdx !== prevLastUserRef.current;
    prevLastUserRef.current = userIdx;

    if (isNewUserTurn && haveTarget) {
      // A brand-new turn always pins, regardless of what the view was doing
      // before -- this is the "send a message, read it from the top" default.
      modeRef.current = "pinned";
      markSelfScroll();
      container.scrollTo({ top: target, behavior: "smooth" });
      return;
    }

    if (modeRef.current === "pinned") {
      // Hold still while the reply fills the reserved space. Once it outgrows
      // the viewport there is nothing left to reveal by holding, so hand over
      // to normal bottom-following.
      const lastEl = document.getElementById(`msg-${messages.length - 1}`);
      const overflowed =
        !!lastEl &&
        lastEl.getBoundingClientRect().bottom > container.getBoundingClientRect().bottom;
      if (!overflowed) return;
      modeRef.current = "following";
    }

    if (modeRef.current === "following") {
      // Scroll the end-of-messages marker into view -- NOT the tail spacer --
      // so following never reveals the reserved empty space.
      markSelfScroll();
      bottomRef.current?.scrollIntoView({ block: "end" });
    }
  }, [messages, streaming, scrollRef]);

  const activeUserIdx = lastUserIndex(messages);

  return (
    <div className={styles.list}>
      {messages.map((message, i) => (
        // `id` gives the Timeline rail (and the scroll logic above) a stable
        // target.
        <div key={i} id={`msg-${i}`} className={styles.messageAnchor}>
          <ChatMessage
            message={message}
            isStreaming={streaming && i === messages.length - 1 && message.role === "assistant"}
            onOpenAttachment={onOpenAttachment}
            // Auto-minimize a long user message while its reply is generating
            // (see ChatMessage). Only the message whose reply is streaming.
            autoCollapse={streaming && message.role === "user" && i === activeUserIdx}
          />
        </div>
      ))}
      <div ref={bottomRef} />
      <div ref={tailRef} aria-hidden="true" />
    </div>
  );
}
