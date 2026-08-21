"use client";

import { useState } from "react";
import type { ChatMessage } from "@/hooks/useChatSocket";
import Markdown from "./Markdown";
import styles from "./Timeline.module.css";

type Props = {
  messages: ChatMessage[];
};

type Tick = {
  message: ChatMessage;
  // Original index into the full `messages` array (not position among user
  // messages), so a click can find the matching `#msg-${index}` anchor
  // MessageList renders for every message, user or assistant.
  index: number;
  // Brief of the assistant reply that answered this user message (the
  // message at index+1, if it's an assistant turn with answer text yet).
  // null while still generating / no reply exists.
  reply: string | null;
};

type TickGroup = {
  label: string;
  ticks: Tick[];
};

const USER_MAX_CHARS = 70;
const REPLY_MAX_CHARS = 90;
const MS_PER_DAY = 24 * 60 * 60 * 1000;

// --- Discrete magnification levels (v3.2) ---
// The hovered tick is the tallest (level 3); each neighbour one ordinal step
// away drops one level; ticks 3+ steps away stay at base. Four widths total:
// base + {0,1,2,3} * STEP.
const BASE_WIDTH = 12;
const STEP = 8;

function widthForDistance(d: number): number {
  if (d === 0) return BASE_WIDTH + 3 * STEP; // hovered
  if (d === 1) return BASE_WIDTH + 2 * STEP; // immediate neighbour
  if (d === 2) return BASE_WIDTH + 1 * STEP; // next-out neighbour
  return BASE_WIDTH; // everything else: default
}

function truncate(text: string, max: number): string {
  const collapsed = text.trim().replace(/\s+/g, " ");
  if (collapsed.length <= max) return collapsed;
  return `${collapsed.slice(0, max).trimEnd()}…`;
}

// Defensive strip of a `<think>…</think>` block from assistant text before
// briefing it -- useChatSocket already splits thinking out of `message.text`,
// but if a raw/unsplit string ever reaches here we still want only the answer.
function stripThink(raw: string): string {
  const close = raw.indexOf("</think>");
  if (close !== -1) return raw.slice(close + "</think>".length);
  // An open-but-unclosed think block means the model is still thinking and
  // hasn't produced answer text yet -- treat as empty.
  if (raw.includes("<think>")) return "";
  return raw;
}

function startOfDay(ms: number): number {
  const d = new Date(ms);
  d.setHours(0, 0, 0, 0);
  return d.getTime();
}

/** "Today" / "Yesterday" / "Jul 17" relative to `now`, for a real timestamp. */
function realDayLabel(timestampMs: number, nowMs: number): string {
  const diffDays = Math.round((startOfDay(nowMs) - startOfDay(timestampMs)) / MS_PER_DAY);
  if (diffDays === 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  return new Date(timestampMs).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

/**
 * Groups user-message ticks into date-labeled clusters for the rail, oldest
 * group first (the rail renders top = oldest, bottom = newest, matching
 * Timeline's tick order). Consecutive ticks on the same calendar day (from
 * each message's `timestamp`) share a group; a new day starts a new group,
 * labeled "Today" / "Yesterday" / "Mon D".
 */
export function groupTicks(userTicks: Tick[]): TickGroup[] {
  if (userTicks.length === 0) return [];

  const now = Date.now();

  // Bucket consecutive ticks by calendar day, derived from each message's
  // timestamp. Same-day messages share a group; a new day starts a new one.
  const groups: TickGroup[] = [];
  for (const tick of userTicks) {
    const label = realDayLabel(tick.message.timestamp, now);
    const current = groups[groups.length - 1];
    if (current && current.label === label) {
      current.ticks.push(tick);
    } else {
      groups.push({ label, ticks: [tick] });
    }
  }

  return groups;
}

/**
 * Left-edge conversation navigator, ChatGPT-style: one tick per USER
 * message (assistant replies aren't ticked -- this is "your chat", not a
 * full transcript index), top = oldest, bottom = newest, clustered into
 * date-labeled groups (see groupTicks). Clicking a tick scroll-to's the
 * matching message via its `id={`msg-${index}`}` anchor (rendered by
 * MessageList); hovering shows a floating popout with a glimpse of that
 * user message AND a brief of the assistant reply that answered it.
 *
 * The rail itself is rendered as a sibling of the scrolling message list
 * (see page.tsx), absolutely positioned against a non-scrolling ancestor --
 * so it stays visually fixed in place while messages scroll beneath it.
 *
 * Magnet magnification (macOS-dock style): a window-level mousemove measures
 * the pointer's Y against each tick's center; the nearest tick lengthens
 * most and neighbors taper off via a Gaussian falloff. Using a window
 * listener (rather than an onMouseMove on the rail) means the effect works
 * without giving the rail `pointer-events: auto`, so clicks/wheel still pass
 * through the rail's empty gutter to the message column beneath it.
 */
export default function Timeline({ messages }: Props) {
  // The message-index of the tick currently hovered (or keyboard-focused).
  // Drives both the white "active" color and the discrete magnification: the
  // hovered tick is level 3, its ordinal neighbours step down one level each.
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  // Pair each USER message with the assistant reply that follows it (user at
  // index i -> assistant at i+1). Ticks are still user messages only; the
  // paired reply is just briefed in the tooltip. Consume the FULL messages
  // list (not a user-filtered one) so the i+1 lookup is meaningful.
  const userTicks: Tick[] = messages
    .map((message, index) => {
      const next = messages[index + 1];
      const replyRaw = next && next.role === "assistant" ? stripThink(next.text) : "";
      const reply = replyRaw.trim() ? replyRaw : null;
      return { message, index, reply };
    })
    .filter(({ message }) => message.role === "user");

  if (userTicks.length === 0) return null;

  const groups = groupTicks(userTicks);

  // Per-tick (group index, position-within-group), so the magnet distance is
  // measured WITHIN a day-group only -- the magnification never bleeds across
  // a group boundary into a different day's ticks.
  const posOf = new Map<number, { g: number; p: number }>();
  groups.forEach((group, g) =>
    group.ticks.forEach((t, p) => posOf.set(t.index, { g, p })),
  );
  const hovered = hoverIndex != null ? posOf.get(hoverIndex) ?? null : null;

  const widthFor = (index: number): number => {
    if (!hovered) return BASE_WIDTH;
    const cur = posOf.get(index);
    // Different group (or unknown) -> always base: magnet stays within the day.
    if (!cur || cur.g !== hovered.g) return BASE_WIDTH;
    return widthForDistance(Math.abs(cur.p - hovered.p));
  };

  const handleClick = (index: number) => {
    // Align the jumped-to message to the TOP of the chat viewport (not center)
    // so the user message sits at the top and its reply reads downward.
    document
      .getElementById(`msg-${index}`)
      ?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <nav className={styles.rail} aria-label="Conversation timeline">
      {groups.map((group, groupIndex) => (
        <div className={styles.group} key={`${group.label}-${groupIndex}`}>
          <div className={styles.groupLabel}>{group.label}</div>
          <div className={styles.groupTicks}>
            {group.ticks.map(({ message, index, reply }) => (
              <div
                key={index}
                className={styles.tickWrap}
                onMouseEnter={() => setHoverIndex(index)}
                onMouseLeave={() =>
                  setHoverIndex((current) => (current === index ? null : current))
                }
              >
                <button
                  type="button"
                  className={`${styles.tick} ${hoverIndex === index ? styles.tickActive : ""}`}
                  onClick={() => handleClick(index)}
                  onFocus={() => setHoverIndex(index)}
                  onBlur={() => setHoverIndex((current) => (current === index ? null : current))}
                  aria-label={`Jump to message: ${truncate(message.text, USER_MAX_CHARS)}`}
                >
                  <span
                    className={styles.tickBar}
                    style={{ width: `${widthFor(index)}px` }}
                  />
                </button>
                {hoverIndex === index && (
                  <div className={styles.tooltip} role="tooltip">
                    <Markdown
                      text={truncate(message.text, USER_MAX_CHARS)}
                      className={styles.tooltipUser}
                    />
                    {reply && (
                      <Markdown
                        text={truncate(reply, REPLY_MAX_CHARS)}
                        className={styles.tooltipReply}
                      />
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      ))}
    </nav>
  );
}
