"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// Ordered timeline of an assistant reply's process, in TRUE arrival order --
// thinking blocks exactly as the model produced them. Each agentic round
// contributes at most one `thinking` item (its `<think>...</think>` block). A
// `thinking` item carries `live: true` while its round's `<think>` block is
// still streaming in (see `roundRawRef` below); the flag is cleared once the
// turn finishes via `done`.
export type ProcessItem =
  | { kind: "thinking"; text: string; live?: boolean };

export type ChatMessage = {
  role: "user" | "assistant";
  text: string;
  finishReason?: string;
  // ms epoch when the message was created (used for the per-message timestamp
  // footer). Set once on creation.
  timestamp: number;
  // Token count for this message. For assistant messages it's the EXACT
  // streamed output-token count (set on `done`). For user messages it's a
  // rough client-side estimate (chars/4) flagged with a leading "~" in the
  // UI, since the client has no tokenizer.
  tokens?: number;
  tokensEstimated?: boolean;
  // Ordered thinking/tool timeline for this (assistant) message -- see
  // ProcessItem. Always [] for user messages and for assistant messages
  // that neither thought nor called a tool.
  process: ProcessItem[];
};

type TokenEvent = {
  type: "token";
  token_id: number;
  text: string;
  top_logprobs: Record<string, number>;
};

type DoneEvent = {
  type: "done";
  // v1.2 adds "tool_limit": the model hit MAX_TOOL_ROUNDS (server-side) and
  // was stopped mid tool-use loop.
  finish_reason: "stop" | "length" | "aborted" | "tool_limit";
};

type ErrorEvent = {
  type: "error";
  message: string;
};

// Sent once, as the FIRST message of a turn (ahead of every
// `token` message) -- the input-side stats the client cannot derive itself
// (prompt length, KV-cache reuse). The client counts OUTPUT tokens and
// derives tokens/sec itself from the `token` stream it already receives.
type GenStatsEvent = {
  type: "gen_stats";
  prompt_tokens: number;
  cached_tokens: number;
};

// Live readout for the CURRENT/last response, exposed by the hook so the
// navbar can render "{in} → {out} ({cached} cached) · {n} tok/s". null
// before the first gen_stats of a session (or right after a reset).
export type GenStats = {
  promptTokens: number;
  cachedTokens: number;
  outputTokens: number;
  tokensPerSec: number;
};

export type SegmentKind = "system" | "user_msg" | "assistant_msg" | "thought" | "doc_chunk" | "scratch";
export type EditableBy = "user" | "model" | "both" | "none";

export type Segment = {
  id: string;
  kind: SegmentKind;
  text: string;
  emphasis: number;
  editable_by: EditableBy;
  provenance: string;
};

export type CacheImpact = {
  // Which request this reply corresponds to, so callers can ignore a reply
  // that landed for a segment/text they've since navigated away from. See
  // the `pendingEditsRef` FIFO queue below for how this is populated.
  segmentId: string;
  text: string;
  firstInvalidToken: number;
  tokensToReprefill: number;
  preview: boolean;
};

type ContextEvent = {
  type: "context";
  segments: Segment[];
};

type CacheImpactEvent = {
  type: "cache_impact";
  first_invalid_token: number;
  tokens_to_reprefill: number;
  preview: boolean;
};

type EditRejectedEvent = {
  type: "edit_rejected";
  message: string;
};

type ServerEvent =
  | TokenEvent
  | DoneEvent
  | ErrorEvent
  | ContextEvent
  | CacheImpactEvent
  | EditRejectedEvent
  | GenStatsEvent;

const MAX_RECONNECT_ATTEMPTS = 5;
const OPEN_TAG = "<think>";
const CLOSE_TAG = "</think>";

/**
 * Incrementally splits a raw, possibly-partial assistant stream into the
 * `<think>...</think>` disclosure and the answer text. Handles: no think
 * block, an open-but-unclosed block (mid-stream), an empty block, and text
 * following the closed block. Safe to call on every token since it just
 * re-derives from the full accumulated string.
 */
export function splitThinking(raw: string): { thoughts: string; text: string } {
  const openIdx = raw.indexOf(OPEN_TAG);
  if (openIdx === -1) {
    return { thoughts: "", text: raw };
  }
  const before = raw.slice(0, openIdx);
  const afterOpen = raw.slice(openIdx + OPEN_TAG.length);
  const closeIdx = afterOpen.indexOf(CLOSE_TAG);
  if (closeIdx === -1) {
    // Still inside the think block; nothing after it yet.
    return { thoughts: afterOpen.trim(), text: before };
  }
  const thoughts = afterOpen.slice(0, closeIdx).trim();
  const after = afterOpen.slice(closeIdx + CLOSE_TAG.length);
  return { thoughts, text: before + after };
}

export function useChatSocket() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [connected, setConnected] = useState(false);
  const [context, setContext] = useState<Segment[] | null>(null);
  const [cacheImpact, setCacheImpact] = useState<CacheImpact | null>(null);
  const [editError, setEditError] = useState<string | null>(null);
  // Live token-stats readout for the current/last response.
  const [stats, setStats] = useState<GenStats | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  // Accumulated raw text of the CURRENT ROUND only (not the whole turn).
  // The server starts a fresh `<think>` block every round (reasoning ->
  // tool call -> reasoning -> ... -> answer), so re-running `splitThinking`
  // over the WHOLE turn's concatenated text would see a second literal
  // `<think>` tag from round 2 as trailing answer text (the bug this
  // per-round accumulation fixes). Reset to "" on `tool_call` (a fresh round
  // is about to start) and on `send()`/socket-close (new turn).
  const roundRawRef = useRef("");
  // First/latest OUTPUT-token timestamps for the in-flight TURN
  // (spans every round), used to derive a live tokens/sec figure. Reset only
  // on a new turn (`send()`) or socket close -- NOT on each round's
  // `gen_stats`, so tokens/sec reflects the whole turn rather than just the
  // last round.
  const statsTimingRef = useRef<{ first: number; last: number } | null>(null);
  // Exact output-token count for the in-flight TURN (spans every round),
  // stamped onto the assistant message on `done`. Reset per turn (send()
  // only) -- NOT per round's `gen_stats` -- so a multi-round tool turn's
  // final `tokens` count includes every round's output, not just the last.
  const outputCountRef = useRef(0);
  const attemptsRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const unmountedRef = useRef(false);
  // FIFO queue of {segmentId, text} for in-flight preview_edit/apply_edit
  // requests, so each cache_impact/edit_rejected reply can be paired with
  // the request that produced it. This is safe because the server handles
  // one client message at a time on a given connection (app.py's
  // `while True: raw = await ws.receive_text()` loop awaits each edit fully
  // -- including sending its cache_impact or edit_rejected reply -- before
  // reading the next message), so replies land in the same order requests
  // were sent, one reply per request, with no other message type consuming
  // a slot in between.
  const pendingEditsRef = useRef<Array<{ segmentId: string; text: string }>>([]);

  const connect = useCallback(() => {
    if (typeof window === "undefined" || unmountedRef.current) return;

    const ws = new WebSocket(`ws://${window.location.hostname}:8321/ws`);
    wsRef.current = ws;

    ws.onopen = () => {
      if (wsRef.current !== ws) return; // stale socket (e.g. StrictMode double-mount); ignore
      attemptsRef.current = 0;
      setConnected(true);
    };

    ws.onclose = () => {
      if (wsRef.current !== ws) return; // stale socket; a newer one already took over
      setConnected(false);
      wsRef.current = null;
      // Reset any in-flight generation so the composer doesn't stay locked on
      // "Stop" forever after an unexpected disconnect, and finalize the
      // in-progress assistant message so its cursor stops blinking.
      roundRawRef.current = "";
      setStreaming(false);
      // No turn can still be "live" once the socket has dropped.
      statsTimingRef.current = null;
      setStats(null);
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (!last || last.role !== "assistant" || last.finishReason) return prev;
        const next = prev.slice(0, -1);
        next.push({ ...last, finishReason: "disconnected" });
        return next;
      });
      // The next connection starts a brand-new server-side ContextObject
      // (see app.py), so any requests still in flight on the old socket can
      // never get a reply -- drop them rather than let them misattribute a
      // future connection's replies.
      pendingEditsRef.current = [];
      if (unmountedRef.current) return;
      if (attemptsRef.current >= MAX_RECONNECT_ATTEMPTS) return;
      const delay = Math.min(1000 * 2 ** attemptsRef.current, 16000);
      attemptsRef.current += 1;
      reconnectTimerRef.current = setTimeout(connect, delay);
    };

    ws.onerror = () => {
      if (wsRef.current !== ws) return; // stale socket; nothing left to tear down
      ws.close();
    };

    ws.onmessage = (event) => {
      if (wsRef.current !== ws) return; // stale socket; drop late messages
      let data: ServerEvent;
      try {
        data = JSON.parse(event.data);
      } catch {
        return;
      }

      if (data.type === "token") {
        roundRawRef.current += data.text;
        const { thoughts, text } = splitThinking(roundRawRef.current);
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (!last || last.role !== "assistant") return prev;
          const next = prev.slice(0, -1);
          // Reflect this round's thinking as the LAST live thinking item:
          // update it in place if one already exists (the common case --
          // most tokens just grow the current round's block), else start a
          // new one the first time this round's thoughts become non-empty.
          const process = last.process.slice();
          const liveIdx = process.findIndex((p) => p.kind === "thinking" && p.live);
          if (liveIdx !== -1) {
            process[liveIdx] = { kind: "thinking", text: thoughts, live: true };
          } else if (thoughts !== "") {
            process.push({ kind: "thinking", text: thoughts, live: true });
          }
          next.push({ ...last, text, process });
          return next;
        });
        // Live output-token count + tokens/sec, ACCUMULATED
        // across every round of the turn (see outputCountRef/statsTimingRef
        // docs -- neither resets on `gen_stats` anymore). token_id -1 is the
        // engine's synthetic "aborted" marker (see engine.py), not an actual
        // generated token, so it's excluded from the count.
        if (data.token_id !== -1) {
          outputCountRef.current += 1;
          const now =
            typeof performance !== "undefined" ? performance.now() : Date.now();
          statsTimingRef.current = statsTimingRef.current
            ? { first: statsTimingRef.current.first, last: now }
            : { first: now, last: now };
          const { first, last } = statsTimingRef.current;
          setStats((prev) => {
            if (!prev) return prev; // no gen_stats yet for this turn -- ignore
            const outputTokens = prev.outputTokens + 1;
            const elapsedSec = (last - first) / 1000;
            const tokensPerSec = elapsedSec > 0 ? outputTokens / elapsedSec : 0;
            return { ...prev, outputTokens, tokensPerSec };
          });
        }
      } else if (data.type === "gen_stats") {
        // The server sends a fresh gen_stats at the START OF EVERY ROUND,
        // not just once per turn -- update the prompt-side figures to this
        // round's values, but keep output tokens/timing accumulating across
        // the whole turn (they're reset only in `send()`, on a genuinely new
        // turn). `prev` is null only for the turn's first gen_stats (send()
        // just cleared it), so outputTokens correctly starts at 0 there and
        // carries forward on every later round's gen_stats within the turn.
        setStats((prev) => ({
          promptTokens: data.prompt_tokens,
          cachedTokens: data.cached_tokens,
          outputTokens: prev?.outputTokens ?? 0,
          tokensPerSec: prev?.tokensPerSec ?? 0,
        }));
      } else if (data.type === "done") {
        setStreaming(false);
        roundRawRef.current = "";
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (!last || last.role !== "assistant") return prev;
          const next = prev.slice(0, -1);
          // Finalize the live thinking item, if any (clears its `live` flag
          // so the UI stops treating it as still-growing).
          let process = last.process;
          const liveIdx = process.findIndex((p) => p.kind === "thinking" && p.live);
          if (liveIdx !== -1) {
            process = process.slice();
            const item = process[liveIdx];
            if (item.kind === "thinking") process[liveIdx] = { kind: "thinking", text: item.text };
          }
          next.push({
            ...last,
            process,
            finishReason: data.finish_reason,
            tokens: outputCountRef.current,
          });
          return next;
        });
        // Keep the context inspector fresh after every turn (the assistant
        // reply just landed as a new segment) regardless of whether the
        // panel is currently open.
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "get_context" }));
        }
      } else if (data.type === "error") {
        setStreaming(false);
        roundRawRef.current = "";
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            text: `[error: ${data.message}]`,
            timestamp: Date.now(),
            process: [],
          },
        ]);
      } else if (data.type === "context") {
        setContext(data.segments);
      } else if (data.type === "cache_impact") {
        // Pair this reply with the request that produced it (see
        // pendingEditsRef above). If the queue is unexpectedly empty, drop
        // the reply rather than tag it to nothing -- an untagged impact
        // could never match any open editor anyway.
        const req = pendingEditsRef.current.shift();
        if (req) {
          setCacheImpact({
            segmentId: req.segmentId,
            text: req.text,
            firstInvalidToken: data.first_invalid_token,
            tokensToReprefill: data.tokens_to_reprefill,
            preview: data.preview,
          });
        }
        setEditError(null);
      } else if (data.type === "edit_rejected") {
        // A rejection also consumes one queue slot -- it's the *other*
        // possible reply to a preview_edit/apply_edit (see handle_edit in
        // app.py: exactly one of cache_impact or edit_rejected is sent per
        // request). Without this, a rejected request would leave a stale
        // entry at the head of the queue and desync every reply after it.
        pendingEditsRef.current.shift();
        setEditError(data.message);
      }
    };
  }, []);

  useEffect(() => {
    unmountedRef.current = false;
    connect();
    return () => {
      unmountedRef.current = true;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const send = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      const ws = wsRef.current;
      if (!trimmed || !ws || ws.readyState !== WebSocket.OPEN) return;
      roundRawRef.current = "";
      // A new TURN starts here -- this is the one place output-token
      // accumulation resets (NOT each round's gen_stats -- see outputCountRef
      // docs), and a new turn's gen_stats hasn't arrived yet, so clear the
      // previous turn's readout rather than let it read stale during the gap.
      statsTimingRef.current = null;
      setStats(null);
      outputCountRef.current = 0;
      const now = Date.now();
      setMessages((prev) => [
        ...prev,
        {
          role: "user",
          text: trimmed,
          timestamp: now,
          // No client-side tokenizer, so estimate ~chars/4 (flagged with "~").
          tokens: Math.max(1, Math.round(trimmed.length / 4)),
          tokensEstimated: true,
          process: [], // never populated on user messages -- see ChatMessage type docs
        },
        { role: "assistant", text: "", timestamp: now, process: [] },
      ]);
      setStreaming(true);
      ws.send(JSON.stringify({ type: "user_message", text: trimmed }));
    },
    [],
  );

  const stop = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: "abort" }));
  }, []);

  const getContext = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: "get_context" }));
  }, []);

  // Shared sender for the one wire-legal edit op the UI offers
  // (replace_text). "append"/"delete"/"move" are deliberately not exposed
  // here -- v1's UI only supports replace-in-place, and the server also
  // rejects "append" over the wire regardless. Pushes onto pendingEditsRef
  // so the eventual cache_impact/edit_rejected reply can be traced back to
  // this request (see pendingEditsRef's docstring above).
  const sendEdit = useCallback(
    (msgType: "preview_edit" | "apply_edit", segmentId: string, text: string) => {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      setEditError(null);
      pendingEditsRef.current.push({ segmentId, text });
      ws.send(
        JSON.stringify({
          type: msgType,
          event: { op: "replace_text", segment_id: segmentId, payload: { text }, actor: "user" },
        }),
      );
    },
    [],
  );

  const previewEdit = useCallback(
    (segmentId: string, newText: string) => {
      sendEdit("preview_edit", segmentId, newText);
    },
    [sendEdit],
  );

  const applyEdit = useCallback(
    (segmentId: string, newText: string) => {
      sendEdit("apply_edit", segmentId, newText);
    },
    [sendEdit],
  );

  return {
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
  };
}
