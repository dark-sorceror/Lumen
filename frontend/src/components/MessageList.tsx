"use client";

import { useEffect, useLayoutEffect, useRef, type RefObject } from "react";
import type { ChatMessage as ChatMessageType, MessageAttachment } from "@/hooks/useChatSocket";
import ChatMessage from "./ChatMessage";
import styles from "./MessageList.module.css";

// Within this many px of the bottom, auto-follow RE-ENGAGES. Deliberately
// tiny (a few px, absorbing only sub-pixel rounding at the true bottom): the
// user has to actually return to the bottom for streaming to resume pinning
// -- being merely "near" it is not enough. A large value here is exactly
// what caused the old scroll "jag" (every streamed token re-stuck the view
// while the user was still trying to read a little above the bottom).
const RE_ENGAGE_PX = 4;
// Past this, the user is clearly reading earlier content -- follow stays off.
// This is only a backstop for scroll sources the wheel/touch listeners don't
// see (keyboard PageUp, the scrollbar thumb); the wheel/touch handlers are
// what disengage instantly on a deliberate drag. Between RE_ENGAGE_PX and
// this, `stickRef` keeps whatever the user's last explicit intent set it to.
const DISENGAGE_PX = 120;
// Gap left above a new turn's user message when it's scrolled to the top of
// the chat area (the "margin gap" -- see the scroll-to-top effect below).
const TOP_GAP_PX = 20;

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
  // races): sits AFTER `bottomRef` so it never counts as message content for
  // the follow logic, and grows just enough that the newest turn's user
  // message can be scrolled all the way to the top of the chat area even when
  // the reply is short. Shrinks to 0 as the reply grows tall enough to fill
  // the viewport on its own.
  const tailRef = useRef<HTMLDivElement | null>(null);
  // Last spacer height we applied, so natural (spacer-less) content height can
  // be derived by subtraction WITHOUT resetting the spacer to 0 to measure --
  // that reset would transiently shrink the scroll range and clamp scrollTop,
  // making the pinned message drift down a few px on every streamed token.
  const tailHRef = useRef(0);
  // Whether streaming content should keep pinning the view to the bottom. A
  // ref, not state: read imperatively, must never trigger a re-render. Starts
  // true so the very first load lands pinned.
  const stickRef = useRef(true);
  const prevLastUserRef = useRef(-1);

  // Track stick/unstick from the user's actual interaction (see the two-bug
  // fix note): scroll position re-engages only at the true bottom and
  // disengages when far; a deliberate wheel-up / touch-drag disengages
  // instantly so a streamed token can't snap the view back down.
  useEffect(() => {
    const container = scrollRef.current;
    if (!container) return;
    const distanceFromBottom = () =>
      container.scrollHeight - container.scrollTop - container.clientHeight;
    const handleScroll = () => {
      const distance = distanceFromBottom();
      if (distance <= RE_ENGAGE_PX) stickRef.current = true;
      else if (distance > DISENGAGE_PX) stickRef.current = false;
    };
    const onWheel = (e: WheelEvent) => {
      if (e.deltaY < 0) stickRef.current = false;
    };
    const onTouchMove = () => {
      if (distanceFromBottom() > RE_ENGAGE_PX) stickRef.current = false;
    };
    handleScroll();
    container.addEventListener("scroll", handleScroll, { passive: true });
    container.addEventListener("wheel", onWheel, { passive: true });
    container.addEventListener("touchmove", onTouchMove, { passive: true });
    return () => {
      container.removeEventListener("scroll", handleScroll);
      container.removeEventListener("wheel", onWheel);
      container.removeEventListener("touchmove", onTouchMove);
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
        const userTopInContent = uTop - cTop + container.scrollTop;
        target = Math.max(0, userTopInContent - TOP_GAP_PX);
        haveTarget = true;
        // Reserve up to a SCREENFUL of room below the newest user message, so
        // it can sit at the top with the reply streaming into the space
        // beneath it. Crucially this also means pinning the message to the top
        // does NOT simultaneously land at the scroll bottom -- if it did
        // (spacer sized to *exactly* reach the top), the bottom-follow logic
        // would instantly re-arm and drag the message back down, which is the
        // bug where the message wouldn't stay at the top. The spacer shrinks
        // to 0 once the reply is tall enough to fill a screen on its own.
        const belowUser = natural - userTopInContent;
        need = Math.max(0, container.clientHeight - belowUser);
      }
    }
    tail.style.height = `${need}px`;
    tailHRef.current = need;

    // A brand-new user turn: pin that message to the top of the chat area and
    // let the reply stream into the space below it. Disengage bottom-follow so
    // streamed tokens don't pull it back down; it stays put until the user
    // themselves scrolls back to the bottom (which re-arms follow).
    const isNewUserTurn = userIdx !== -1 && userIdx !== prevLastUserRef.current;
    prevLastUserRef.current = userIdx;

    if (isNewUserTurn && haveTarget) {
      stickRef.current = false;
      container.scrollTo({ top: target, behavior: "smooth" });
    } else if (stickRef.current) {
      // Following the stream at the bottom (only when the user is parked
      // there): scroll the end-of-messages marker into view -- NOT the tail
      // spacer -- so following never reveals the reserved empty space.
      bottomRef.current?.scrollIntoView({ block: "end" });
    }
  }, [messages, streaming, scrollRef]);

  const activeUserIdx = lastUserIndex(messages);

  return (
    <div className={styles.list}>
      {messages.map((message, i) => (
        // `id` gives the Timeline rail (and the scroll-to-top logic above) a
        // stable target.
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
