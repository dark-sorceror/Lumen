"use client";

import type { ReactNode } from "react";
import { useState } from "react";
import type {
  ChatMessage as ChatMessageType,
  MessageAttachment,
  ProcessItem,
  ToolStep,
} from "@/hooks/useChatSocket";
import Markdown, { withCursor } from "./Markdown";
import MessageAttachments from "./MessageAttachments";
import styles from "./ChatMessage.module.css";

type Props = {
  message: ChatMessageType;
  isStreaming: boolean;
  onOpenAttachment: (attachment: MessageAttachment) => void;
};

const EXCERPT_MAX = 90;

// A stage must reach this length before a discourse marker is allowed to
// start a new one; keeps rapid-fire "Wait. Hmm. Okay." pivots from
// exploding into dozens of one-sentence stages.
const MIN_STAGE_CHARS = 80;

// Discourse markers that typically open a new phase of reasoning in
// Qwen3/R1-style thinking traces ("Wait", "Alternatively", "Let me check",
// enumerations, wrap-ups). A sentence starting with one of these begins a
// new stage. Curated and deliberately conservative -- tune against real
// traces; a future version replaces this whole heuristic with model-generated
// stage segmentation + summaries.
const STAGE_MARKERS =
  /^(Wait\b|Hm+\b|Okay\b|Alright\b|Actually\b|Alternatively\b|But wait\b|Oh wait\b|Let me\b|Let's\b|First\b|Second\b|Third\b|Next\b|Now\b|Another\b|Then again\b|On the other hand\b|Finally\b|To summarize\b|In summary\b|Therefore\b|Overall\b|So,|So the\b|Putting it\b|The user\b|I need to\b|I should\b)/;

// Rough sentence tokenizer: runs of text ending in `.`/`!`/`?` (plus any
// closing quotes/brackets), or a trailing unterminated run -- which is
// exactly what the still-streaming tail of the last stage looks like.
function splitSentences(text: string): string[] {
  return text.match(/[^.!?]+[.!?]+["')\]]*\s*|[^.!?]+$/g) ?? [text];
}

// Splits one blank-line-delimited chunk into reasoning stages: walk its
// sentences and open a new stage whenever a sentence starts with a stage
// marker and the current stage is long enough to stand on its own;
// otherwise the sentence joins the current stage.
function splitChunkIntoStages(chunk: string): string[] {
  const stages: string[] = [];
  for (const sentence of splitSentences(chunk)) {
    const s = sentence.replace(/\s+/g, " ").trim();
    if (s === "") continue;
    const last = stages[stages.length - 1];
    if (last === undefined || (STAGE_MARKERS.test(s) && last.length >= MIN_STAGE_CHARS)) {
      stages.push(s);
    } else {
      stages[stages.length - 1] = `${last} ${s}`;
    }
  }
  return stages;
}

// Splits a raw thoughts string into reasoning stages. Blank lines are
// always stage boundaries (models that emit clean paragraphs get one stage
// per paragraph as a baseline); traces with no blank lines fall back to
// single newlines; and every chunk is then further segmented on discourse
// markers, so even a single jumbled block separates into distinct stages.
// Pure and cheap enough to re-derive from the full thoughts string on
// every streaming render.
function splitStages(thoughts: string): string[] {
  const byBlankLine = thoughts.split(/\n{2,}/);
  const coarse = byBlankLine.length > 1 ? byBlankLine : thoughts.split(/\n/);
  return coarse
    .map((p) => p.trim())
    .filter((p) => p !== "")
    .flatMap(splitChunkIntoStages);
}

// Heuristic for the one-liner shown per reasoning stage: the first
// sentence (up to the first `.`/`!`/`?` followed by whitespace or the
// stage's end), hard-capped to a fixed character budget. This runs on
// every render while the trace is still streaming in, so the excerpt for
// the last, still-growing stage updates token by token. A future version will
// replace this with a model-generated summary per stage.
function excerptLine(paragraph: string): string {
  const trimmed = paragraph.trim();
  const sentenceMatch = trimmed.match(/^.*?[.!?](?=\s|$)/);
  const sentence = sentenceMatch ? sentenceMatch[0] : trimmed;
  if (sentence.length <= EXCERPT_MAX) return sentence;
  return `${sentence.slice(0, EXCERPT_MAX - 1).trimEnd()}…`;
}

const ARGS_MAX = 60;
const RESULT_MAX = 80;

// Compact, readable rendering of a tool call's arguments for the chip
// label -- e.g. `calculator("2+2")` or `search_context("query", 3)`. A
// single-argument call drops the key entirely (it's almost always
// self-evident from the tool name); multi-argument calls keep `key: value`
// pairs so each value stays attributable. The full JSON is always available
// via the chip's `title` tooltip, so truncation/key-dropping here loses
// nothing irrecoverably.
function formatToolArgs(args: Record<string, unknown>): string {
  const entries = Object.entries(args ?? {});
  if (entries.length === 0) return "";
  const parts = entries.map(([key, value]) => {
    const rendered = typeof value === "string" ? `"${value}"` : JSON.stringify(value);
    return entries.length === 1 ? rendered : `${key}: ${rendered}`;
  });
  const joined = parts.join(", ");
  return joined.length > ARGS_MAX ? `${joined.slice(0, ARGS_MAX - 1).trimEnd()}…` : joined;
}

function truncate(text: string, max: number): string {
  const trimmed = text.trim();
  return trimmed.length > max ? `${trimmed.slice(0, max - 1).trimEnd()}…` : trimmed;
}

// One white-box tool-call chip: 🔧 name(args) [-> result | pending dot].
// Rendered identically whether it's sitting in the live collapsed preview
// or inline among the expanded reasoning stages (see the two call sites
// below) -- the process reads the same either way.
function ToolStepChip({ step }: { step: ToolStep }) {
  const fullArgs = JSON.stringify(step.arguments);
  return (
    <span
      className={`${styles.toolChip} ${step.status === "error" ? styles.toolChipError : ""}`}
      title={`${step.name}\narguments: ${fullArgs}${
        step.result != null ? `\nresult: ${step.result}` : ""
      }`}
    >
      <span className={styles.toolChipIcon} aria-hidden="true">
        🔧
      </span>
      {/* Only the args ellipsis-truncate; the tool name and the enclosing
          parens always stay visible, so a call never renders as a
          dangling "name(...k:…" with the ")" clipped off. */}
      <span className={styles.toolChipLabel}>
        {step.name}(<span className={styles.toolChipArgs}>{formatToolArgs(step.arguments)}</span>)
      </span>
      {step.status === "pending" ? (
        <span className={styles.toolChipPending} role="status" aria-label={`running ${step.name}`}>
          <span className={styles.toolChipDot} aria-hidden="true" />
          running…
        </span>
      ) : (
        <span className={styles.toolChipResult}>
          <span aria-hidden="true">→</span> {truncate(step.result ?? "", RESULT_MAX)}
        </span>
      )}
    </span>
  );
}

// Pretty-prints a tool result for the JSON box: `step.result` is a raw
// string that may itself be a JSON payload (most tool results are) or plain
// text -- pretty-print when it parses as JSON, otherwise show it verbatim.
function formatToolResult(result: string): string {
  try {
    return JSON.stringify(JSON.parse(result), null, 2);
  } catch {
    return result;
  }
}

// Streaming-only compact disclosure: shows a tool call's full input/output as
// JSON while it's the live focus of the collapsed preview (see `showToolBox`
// in the component below). A quieter, more detailed sibling of ToolStepChip
// -- that chip stays the collapsed/expanded summary everywhere else; this
// box only appears for the ONE tool call currently in flight or just
// resolved, right before the model either resumes thinking or answers.
function ToolJsonBox({ step }: { step: ToolStep }) {
  return (
    <div className={styles.toolJsonBox}>
      <div className={styles.toolJsonHeader}>
        <span className={styles.toolChipIcon} aria-hidden="true">
          🔧
        </span>
        <span className={styles.toolJsonName}>{step.name}</span>
        {step.status === "pending" && (
          <span className={styles.toolChipPending} role="status" aria-label={`running ${step.name}`}>
            <span className={styles.toolChipDot} aria-hidden="true" />
            running…
          </span>
        )}
      </div>
      <pre className={styles.toolJsonBlock}>{JSON.stringify(step.arguments, null, 2)}</pre>
      {step.status !== "pending" && (
        <pre
          className={`${styles.toolJsonBlock} ${step.status === "error" ? styles.toolChipError : ""}`}
        >
          {formatToolResult(step.result ?? "")}
        </pre>
      )}
    </div>
  );
}

// Expanded-view renderer: walks `process` IN ORDER and emits reasoning
// stages and tool chips interleaved exactly as they arrived -- thinking ->
// 🔧 tool -> thinking -> ... -> (answer, rendered separately below). This is
// what replaces the old "all thoughts, then all chips" layout.
function renderProcess(process: ProcessItem[]): ReactNode[] {
  const nodes: ReactNode[] = [];
  process.forEach((item, i) => {
    if (item.kind === "thinking") {
      splitStages(item.text).forEach((stage, j) => {
        nodes.push(
          <p key={`think-${i}-${j}`} className={styles.thoughtsParagraph}>
            {stage}
          </p>,
        );
      });
    } else {
      nodes.push(
        <div key={`tool-${i}`} className={styles.toolChips}>
          <ToolStepChip step={item.step} />
        </div>,
      );
    }
  });
  return nodes;
}

export default function ChatMessage({ message, isStreaming, onOpenAttachment }: Props) {
  const isUser = message.role === "user";
  const [expanded, setExpanded] = useState(false);
  // `process` is always [] on user messages (see useChatSocket's ChatMessage
  // type). The Thoughts toggle appears whenever there's ANY process item.
  const process = isUser ? [] : message.process;
  const hasProcess = process.length > 0;
  const toolItems = process.filter((p): p is Extract<ProcessItem, { kind: "tool" }> => p.kind === "tool");
  const hasSteps = toolItems.length > 0;
  const lastItem = process[process.length - 1];
  const lastThinkingItem = [...process]
    .reverse()
    .find((p): p is Extract<ProcessItem, { kind: "thinking" }> => p.kind === "thinking");
  const lastThinkingStages = lastThinkingItem ? splitStages(lastThinkingItem.text) : [];
  // Tool JSON box condition: the trailing process state IS a tool call that
  // hasn't been superseded by new thinking yet -- i.e. the LAST item is a
  // `tool`. Streaming-only: past turns show the tool chip list instead.
  const showToolBox = isStreaming && lastItem?.kind === "tool";
  const showToolLimitNote = !isUser && message.finishReason === "tool_limit";
  // The streaming cursor lives with whatever is actively being written. While
  // the model is still THINKING the answer is empty, so the cursor belongs at
  // the end of the live thinking line -- not on the empty answer line below
  // it. Once the answer starts, it moves to the answer text.
  const answerStarted = message.text.trim() !== "";
  const displayText =
    isStreaming && answerStarted ? withCursor(message.text) : message.text;
  const cursorOnThoughts = isStreaming && !answerStarted && !showToolBox;

  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(message.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable (e.g. insecure context) -- no-op */
    }
  };

  const time = new Date(message.timestamp).toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
  });
  const tokenLabel =
    message.tokens != null
      ? `${message.tokensEstimated ? "~" : ""}${message.tokens.toLocaleString()} tok`
      : null;
  // Hide the footer while the assistant reply is still streaming (its token
  // count isn't final yet); user messages are never the streaming message.
  const showFooter = !isStreaming;

  return (
    <div className={`${styles.row} ${isUser ? styles.rowUser : styles.rowAssistant}`}>
     <div className={`${styles.column} ${isUser ? styles.columnUser : styles.columnAssistant}`}>
      {isUser && message.attachments && message.attachments.length > 0 && (
        <MessageAttachments attachments={message.attachments} onOpen={onOpenAttachment} />
      )}
      {isUser ? (
        <div className={`${styles.bubble} ${styles.user}`}>
          <div className={styles.text}>
            <Markdown text={displayText} />
          </div>
        </div>
      ) : (
        <div className={`${styles.bubble} ${styles.assistant}`}>
          {hasProcess && (
            <button
              type="button"
              className={styles.thoughts}
              aria-expanded={expanded}
              onClick={() => setExpanded((v) => !v)}
            >
              <span className={styles.thoughtsHeader}>
                <svg
                  className={`${styles.caret} ${expanded ? styles.caretOpen : ""}`}
                  width="10"
                  height="10"
                  viewBox="0 0 10 10"
                  aria-hidden="true"
                >
                  <path
                    d="M3 1.5L7 5L3 8.5"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.4"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
                Thoughts
              </span>
              {expanded ? (
                <div className={styles.thoughtsFull}>{renderProcess(process)}</div>
              ) : isStreaming ? (
                <>
                  {lastThinkingStages.length > 0 && (
                    // Carriage-return semantics: a SINGLE live line showing
                    // only the most recent reasoning stage (from the last
                    // thinking item in `process`), overwritten in place as
                    // each new stage opens -- never a growing vertical stack.
                    <div className={styles.thoughtsLine}>
                      <span className={styles.thoughtsLineText}>
                        {excerptLine(lastThinkingStages[lastThinkingStages.length - 1])}
                      </span>
                      {cursorOnThoughts && (
                        <span className={styles.streamCursor} aria-hidden="true" />
                      )}
                    </div>
                  )}
                  {/* Tool JSON box: only while a tool call is the trailing,
                      not-yet-superseded-by-new-thinking process state (see
                      `showToolBox`). Disappears the instant the model resumes
                      thinking -- the preview falls back to the line above. */}
                  {showToolBox && lastItem?.kind === "tool" && <ToolJsonBox step={lastItem.step} />}
                </>
              ) : (
                // Collapsed + done (a past turn): the tool chips are the key
                // white-box signal, so they stay visible collapsed -- but no
                // reasoning text (that's behind the expand toggle).
                hasSteps && (
                  <div className={styles.toolChips}>
                    {toolItems.map((item) => (
                      <ToolStepChip key={item.step.callId} step={item.step} />
                    ))}
                  </div>
                )
              )}
              {showToolLimitNote && (
                <div className={styles.toolLimitNote}>Stopped after 5 tool rounds</div>
              )}
            </button>
          )}
          <div className={styles.text}>
            <Markdown text={displayText} />
          </div>
        </div>
      )}
      {showFooter && (
        <div className={styles.footer}>
          <span>{time}</span>
          {tokenLabel && (
            <>
              <span aria-hidden="true">·</span>
              <span
                title={
                  message.tokensEstimated
                    ? "estimated tokens (no client-side tokenizer)"
                    : "output tokens"
                }
              >
                {tokenLabel}
              </span>
            </>
          )}
          {message.text.trim() !== "" && (
            <button
              type="button"
              className={styles.copyBtn}
              onClick={copy}
              aria-label="Copy message as Markdown"
              title="Copy as Markdown"
            >
              {copied ? (
                <svg width="13" height="13" viewBox="0 0 16 16" aria-hidden="true">
                  <path
                    d="M3.5 8.5L6.5 11.5L12.5 4.5"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.6"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              ) : (
                <svg width="13" height="13" viewBox="0 0 16 16" aria-hidden="true">
                  <rect
                    x="5.5"
                    y="5.5"
                    width="7.5"
                    height="7.5"
                    rx="1.5"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.4"
                  />
                  <path
                    d="M10.5 5.5V4A1.5 1.5 0 009 2.5H4A1.5 1.5 0 002.5 4v5A1.5 1.5 0 004 10.5h1.5"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.4"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              )}
            </button>
          )}
        </div>
      )}
     </div>
    </div>
  );
}
