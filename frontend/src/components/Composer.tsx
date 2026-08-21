"use client";

import { useRef, useState } from "react";
import MarkdownEditor, { type MarkdownEditorHandle } from "./MarkdownEditor";
import styles from "./Composer.module.css";

type Props = {
  streaming: boolean;
  disabled: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
  contextOpen: boolean;
  onToggleContext: () => void;
};

export default function Composer({
  streaming,
  disabled,
  onSend,
  onStop,
  contextOpen,
  onToggleContext,
}: Props) {
  // `isEmpty` mirrors the old `!value.trim()` check but is now reported by
  // the editor itself (via MarkdownEditor's onEmptyChange) since the
  // Lexical EditorState, not a plain string, is this component's source of
  // truth while typing. The markdown *string* only gets materialized once,
  // on send (see MarkdownEditor.attemptSubmit's `$convertToMarkdownString`).
  const [isEmpty, setIsEmpty] = useState(true);
  const editorRef = useRef<MarkdownEditorHandle | null>(null);

  // Called by MarkdownEditor with the serialized markdown once a send has
  // actually been committed (Enter, or the Send button below calling
  // editorRef.current.submit()); the editor has already cleared itself by
  // the time this fires.
  const handleSend = (markdown: string) => {
    onSend(markdown);
    editorRef.current?.focus();
  };

  const sendDisabled = disabled || isEmpty;

  return (
    <div className={`${styles.composer} ${streaming ? styles.streaming : ""}`}>
      <MarkdownEditor
        ref={editorRef}
        placeholder="Message the model..."
        disabled={disabled}
        streaming={streaming}
        canSubmitEmpty={false}
        blockSubmit={false}
        onSend={handleSend}
        onEmptyChange={setIsEmpty}
        onPasteFiles={() => {}}
      />
      <div className={styles.toolbar}>
        <div className={styles.toolbarGroup}>
          <button
            type="button"
            className={`${styles.chip} ${contextOpen ? styles.chipActive : ""}`}
            disabled={disabled}
            aria-label="Context inspector"
            aria-pressed={contextOpen}
            title="Context inspector"
            onClick={onToggleContext}
          >
            <svg className={styles.chipIcon} width="13" height="13" viewBox="0 0 14 14" aria-hidden="true">
              <rect x="1.5" y="1.5" width="8" height="8" rx="1.3" fill="none" stroke="currentColor" strokeWidth="1.3" />
              <rect x="4.5" y="4.5" width="8" height="8" rx="1.3" fill="none" stroke="currentColor" strokeWidth="1.3" />
            </svg>
            Context
          </button>
        </div>
        <div className={styles.toolbarGroup}>
          <button
            type="button"
            className={styles.chip}
            disabled={disabled}
            aria-label="Model"
            title="Model (fixed in v1)"
          >
            Qwen3-8B
            <svg className={styles.chevron} width="9" height="9" viewBox="0 0 10 10" aria-hidden="true">
              <path
                d="M2 3.5L5 6.5L8 3.5"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.4"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
          <button
            type="button"
            className={styles.iconBtn}
            disabled={disabled}
            aria-label="Voice input"
            title="Voice input (coming soon)"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
              <rect x="5" y="1" width="4" height="7" rx="2" fill="none" stroke="currentColor" strokeWidth="1.3" />
              <path
                d="M3 7a4 4 0 008 0M7 11v2"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.3"
                strokeLinecap="round"
              />
            </svg>
          </button>
          {streaming ? (
            <button
              type="button"
              className={styles.sendBtn}
              onClick={onStop}
              aria-label="Stop generating"
              title="Stop generating"
            >
              <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
                <rect x="1" y="1" width="10" height="10" rx="1.5" fill="currentColor" />
              </svg>
            </button>
          ) : (
            <button
              type="button"
              className={styles.sendBtn}
              onClick={() => editorRef.current?.submit()}
              disabled={sendDisabled}
              aria-label="Send message"
              title="Send message"
            >
              <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
                <path
                  d="M8 13.5V3.5M8 3.5L3.5 8M8 3.5L12.5 8"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.6"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
