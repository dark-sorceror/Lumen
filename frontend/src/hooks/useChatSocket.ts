"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// A single attachment as displayed inside a sent chat message (NOT the
// composer's upload-tracking `Attachment` -- see Composer.tsx -- this is
// just the slice of it worth remembering after send: enough to render the
// same visual chip in the chat log and estimate its token contribution).
export type MessageAttachment = {
  name: string;
  kind: "image" | "pdf" | "file";
  // Object URL for image previews only (mirrors Composer's Attachment).
  // Ownership transfers to the message on send -- see Composer.handleSend's
  // revoke logic -- so it stays valid for the rest of the session.
  previewUrl?: string;
  // Extracted-text character count the server reported for this attachment
  // (POST /attachments' `chars`), used to estimate its context-token impact
  // with the same chars/4 heuristic as user-text. Undefined if the upload
  // response didn't carry it.
  chars?: number;
};

// One native tool call + its (eventual) result, in the SAME order the
// `tool_call`/`tool_result` events arrived over the wire (see `steps` below
// -- callers never need to re-sort). Starts "pending" on `tool_call` and is
// resolved in place -- matched by `callId` -- when the paired `tool_result`
// arrives; a step whose result never arrives (e.g. the turn was aborted or
// disconnected mid-call) just stays "pending" forever, which the UI renders
// the same as "still running".
export type ToolStep = {
  callId: string;
  name: string;
  arguments: Record<string, unknown>;
  result?: string;
  error?: boolean;
  status: "pending" | "done" | "error";
};

// Ordered timeline of an assistant reply's process, in TRUE arrival order --
// thinking blocks and tool calls interleaved exactly as the model produced
// them (reasoning -> tool -> reasoning -> ... -> answer), rather than "all
// thoughts, then all tool chips" the way the old `thoughts` + `steps` pair
// rendered. Each agentic round contributes at most one `thinking` item
// (its `<think>...</think>` block) optionally followed by one `tool` item
// (the round's tool call, if it made one instead of answering). A
// `thinking` item carries `live: true` while its round's `<think>` block is
// still streaming in (see `roundRawRef` below); the flag is cleared once
// that round ends (a `tool_call` arrives, or the turn finishes via `done`).
export type ProcessItem =
  | { kind: "thinking"; text: string; live?: boolean }
  | { kind: "tool"; step: ToolStep };

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
  // Attachments carried into this (user) message at send time, for display
  // above the message text. Never set on assistant messages.
  attachments?: MessageAttachment[];
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

// v1.2: the model requested a native tool call. `arguments` arrives already
// parsed (the server decodes the model's JSON before sending), and
// `call_id` is opaque -- just a matching key for the paired `tool_result`.
type ToolCallEvent = {
  type: "tool_call";
  call_id: string;
  name: string;
  arguments: Record<string, unknown>;
};

// v1.2: the paired result for a `tool_call` with the same `call_id`. `name`
// is redundant with the call but included by the server for convenience;
// not relied on here since the call already carries it.
type ToolResultEvent = {
  type: "tool_result";
  call_id: string;
  name: string;
  result: string;
  error: boolean;
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
  | GenStatsEvent
  | ToolCallEvent
  | ToolResultEvent;

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
      } else if (data.type === "tool_call") {
        // This round's thinking is complete -- finalize the live thinking
        // item (or drop it if it never grew any actual text), then append
        // the new pending tool step, in arrival order relative to everything
        // before it. Resolved in place by the matching `tool_result` below.
        // Reset roundRawRef so the NEXT round's tokens are split fresh --
        // this is the fix for the multi-<think>-block bug: without it, the
        // next round's raw `<think>...</think>` tag would concatenate onto
        // this round's already-closed text and leak into the answer.
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (!last || last.role !== "assistant") return prev;
          const next = prev.slice(0, -1);
          const process = last.process.slice();
          const liveIdx = process.findIndex((p) => p.kind === "thinking" && p.live);
          if (liveIdx !== -1) {
            const item = process[liveIdx];
            if (item.kind === "thinking" && item.text.trim() === "") {
              process.splice(liveIdx, 1);
            } else if (item.kind === "thinking") {
              process[liveIdx] = { kind: "thinking", text: item.text };
            }
          }
          const step: ToolStep = {
            callId: data.call_id,
            name: data.name,
            arguments: data.arguments,
            status: "pending",
          };
          process.push({ kind: "tool", step });
          next.push({ ...last, process });
          return next;
        });
        roundRawRef.current = "";
      } else if (data.type === "tool_result") {
        // Resolve the pending tool item with the same call_id. If it's
        // somehow missing (e.g. the call arrived on a stale/dropped socket),
        // drop the result rather than fabricate an item for it.
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (!last || last.role !== "assistant") return prev;
          const idx = last.process.findIndex(
            (p) => p.kind === "tool" && p.step.callId === data.call_id,
          );
          if (idx === -1) return prev;
          const process = last.process.slice();
          const item = process[idx];
          if (item.kind !== "tool") return prev;
          process[idx] = {
            kind: "tool",
            step: {
              ...item.step,
              result: data.result,
              error: data.error,
              status: data.error ? "error" : "done",
            },
          };
          const next = prev.slice(0, -1);
          next.push({ ...last, process });
          return next;
        });
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
    (text: string, attachmentIds?: string[], attachmentsDisplay?: MessageAttachment[]) => {
      const trimmed = text.trim();
      const hasAttachments = !!attachmentIds && attachmentIds.length > 0;
      const ws = wsRef.current;
      // A message needs *some* content to be worth sending -- either text or
      // at least one uploaded attachment (mirrors MarkdownEditor's
      // hasAttachments submit gate, which already lets an attachment-only send
      // through with empty text).
      if ((!trimmed && !hasAttachments) || !ws || ws.readyState !== WebSocket.OPEN) return;
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
          ...(attachmentsDisplay && attachmentsDisplay.length > 0
            ? { attachments: attachmentsDisplay }
            : {}),
        },
        { role: "assistant", text: "", timestamp: now, process: [] },
      ]);
      setStreaming(true);
      // attachment_ids is only added to the payload when non-empty, so a
      // plain text send is byte-for-byte the same message it always was.
      ws.send(
        JSON.stringify({
          type: "user_message",
          text: trimmed,
          ...(hasAttachments ? { attachment_ids: attachmentIds } : {}),
        }),
      );
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
