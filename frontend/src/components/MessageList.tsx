"use client";

import { useEffect, useRef, type RefObject } from "react";
import type { ChatMessage as ChatMessageType } from "@/hooks/useChatSocket";
import ChatMessage from "./ChatMessage";
import styles from "./MessageList.module.css";

type Props = {
  messages: ChatMessageType[];
  streaming: boolean;
  scrollRef: RefObject<HTMLDivElement | null>;
};

function lastUserIndex(messages: ChatMessageType[]): number {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user") return i;
  }
  return -1;
}

export default function MessageList({ messages, streaming, scrollRef }: Props) {
  const bottomRef = useRef<HTMLDivElement | null>(null);

  // Simple stick-to-bottom: keep the newest content in view as the reply
  // streams in. `scrollRef` is the scrolling container (owned by page.tsx).
  useEffect(() => {
    void scrollRef;
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [messages, streaming, scrollRef]);

  const activeUserIdx = lastUserIndex(messages);

  return (
    <div className={styles.list}>
      {messages.map((message, i) => (
        // `id` gives the Timeline rail a stable target.
        <div key={i} id={`msg-${i}`} className={styles.messageAnchor}>
          <ChatMessage
            message={message}
            isStreaming={streaming && i === messages.length - 1 && message.role === "assistant"}
            // Auto-minimize a long user message while its reply is generating
            // (see ChatMessage). Only the message whose reply is streaming.
            autoCollapse={streaming && message.role === "user" && i === activeUserIdx}
          />
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
