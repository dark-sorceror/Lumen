"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import MarkdownEditor, { type MarkdownEditorHandle } from "./MarkdownEditor";
import AttachmentRow from "./AttachmentRow";
import type { MessageAttachment } from "@/hooks/useChatSocket";
import styles from "./Composer.module.css";

export type AttachmentKind = "image" | "pdf" | "file";

/** Eager-upload status: attachments start POSTing to /attachments the
 *  instant they're added (see addFiles/uploadAttachment below), not on
 *  send -- so by the time the user hits Send, most attachments are already
 *  "done" and just need their server id collected. */
export type UploadState = "uploading" | "done" | "error";

export type Attachment = {
  id: string;
  file: File;
  kind: AttachmentKind;
  name: string;
  /** Object URL for image previews only; undefined for pdf/file. Must be
   *  revoked (see Composer's removeAttachment / unmount cleanup) once it's
   *  no longer needed -- object URLs otherwise leak for the tab's lifetime. */
  previewUrl?: string;
  uploadState: UploadState;
  /** Set once uploadState becomes "done" -- the id the backend assigned this
   *  attachment, referenced in the WS user_message's attachment_ids. */
  serverId?: string;
  /** Set once uploadState becomes "done" -- the extracted-text character
   *  count POST /attachments reported for this file, used to estimate its
   *  context-token contribution once it's shown in a sent message (see
   *  handleSend / ChatMessage's footer). */
  chars?: number;
  /** Short, human-readable failure reason, shown on hover when
   *  uploadState === "error" (see AttachmentRow). */
  errorMsg?: string;
};

function attachmentKindOf(file: File): AttachmentKind {
  if (file.type.startsWith("image/")) return "image";
  if (file.type === "application/pdf") return "pdf";
  return "file";
}

// Monotonic counter (rather than crypto.randomUUID, which needs a secure
// context) so attachment ids are stable and collision-free within a session
// -- good enough as a React `key` and removal token, never persisted.
let attachmentIdSeq = 0;
function nextAttachmentId(): string {
  attachmentIdSeq += 1;
  return `attachment-${attachmentIdSeq}`;
}

type Props = {
  streaming: boolean;
  disabled: boolean;
  onSend: (text: string, attachmentIds?: string[], attachmentsDisplay?: MessageAttachment[]) => void;
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
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const editorRef = useRef<MarkdownEditorHandle | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Mirrors `attachments` so the unmount-cleanup effect below (registered
  // once, empty deps) can still see the *latest* list instead of whatever
  // was current when the effect was first set up.
  const attachmentsRef = useRef(attachments);
  attachmentsRef.current = attachments;

  // Revoke every outstanding image object URL when the composer unmounts,
  // so navigating away mid-draft doesn't leak blob URLs for the rest of the
  // tab's life.
  useEffect(() => {
    return () => {
      for (const attachment of attachmentsRef.current) {
        if (attachment.previewUrl) URL.revokeObjectURL(attachment.previewUrl);
      }
    };
  }, []);

  // Eagerly POSTs a single attachment to the backend the moment it's added
  // (rather than waiting for send -- see the module doc on UploadState).
  const uploadAttachment = useCallback((id: string, file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    fetch(`http://${window.location.hostname}:8321/attachments`, {
      method: "POST",
      body: formData,
    })
      .then(async (res) => {
        if (!res.ok) {
          let message = `Upload failed (${res.status})`;
          try {
            const body = await res.json();
            if (typeof body?.detail === "string") message = body.detail;
          } catch {
            // Non-JSON error body -- fall back to the generic status message.
          }
          throw new Error(message);
        }
        return res.json() as Promise<{ id: string; chars?: number }>;
      })
      .then((json) => {
        setAttachments((prev) =>
          prev.map((a) =>
            a.id === id
              ? { ...a, uploadState: "done", serverId: json.id, chars: json.chars }
              : a,
          ),
        );
      })
      .catch((err: unknown) => {
        const message = err instanceof Error ? err.message : "Upload failed";
        setAttachments((prev) =>
          prev.map((a) => (a.id === id ? { ...a, uploadState: "error", errorMsg: message } : a)),
        );
      });
  }, []);

  // Shared by both add paths: the toolbar's hidden <input type="file"> and
  // MarkdownEditor's paste-intercept callback.
  const addFiles = useCallback(
    (files: FileList | File[]) => {
      const list = Array.from(files);
      if (list.length === 0) return;
      const additions: Attachment[] = list.map((file) => {
        const kind = attachmentKindOf(file);
        return {
          id: nextAttachmentId(),
          file,
          kind,
          name: file.name,
          // Blob URL retained for image AND pdf so both can be opened in the
          // right-side media viewer once sent (see MessageAttachments/
          // MediaPreview). The draft chip still only renders a thumbnail for
          // images; a pdf URL here changes nothing visually until it's clicked.
          previewUrl: kind === "image" || kind === "pdf" ? URL.createObjectURL(file) : undefined,
          uploadState: "uploading",
        };
      });
      setAttachments((prev) => [...prev, ...additions]);
      for (const attachment of additions) {
        uploadAttachment(attachment.id, attachment.file);
      }
    },
    [uploadAttachment],
  );

  const removeAttachment = useCallback((id: string) => {
    setAttachments((prev) => {
      const target = prev.find((a) => a.id === id);
      if (target?.previewUrl) URL.revokeObjectURL(target.previewUrl);
      return prev.filter((a) => a.id !== id);
    });
  }, []);

  const handleAttachClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) addFiles(event.target.files);
    // Reset the input's value so choosing the exact same file(s) again
    // still fires a change event next time.
    event.target.value = "";
  };

  // Called by MarkdownEditor with the serialized markdown once a send has
  // actually been committed; the editor has already cleared itself by the
  // time this fires.
  const handleSend = (markdown: string) => {
    const doneAttachments = attachments.filter(
      (a): a is Attachment & { serverId: string } => a.uploadState === "done" && !!a.serverId,
    );
    const serverIds = doneAttachments.map((a) => a.serverId);
    const attachmentsDisplay: MessageAttachment[] = doneAttachments.map((a) => ({
      name: a.name,
      kind: a.kind,
      previewUrl: a.previewUrl,
      chars: a.chars,
    }));
    onSend(markdown, serverIds, attachmentsDisplay.length > 0 ? attachmentsDisplay : undefined);
    // Only revoke previewUrls of attachments that did NOT make it into the
    // message; the ones carried in transfer ownership to that message.
    for (const attachment of attachments) {
      if (attachment.previewUrl && attachment.uploadState !== "done") {
        URL.revokeObjectURL(attachment.previewUrl);
      }
    }
    setAttachments([]);
    editorRef.current?.focus();
  };

  // An attachment still uploading hard-blocks send (Enter + button), so a
  // send can never drop a not-yet-ready attachment. An empty-text message is
  // sendable only once at least one attachment has finished uploading.
  const uploadsPending = attachments.some((a) => a.uploadState === "uploading");
  const hasReadyAttachment = attachments.some((a) => a.uploadState === "done");
  const sendDisabled = disabled || uploadsPending || (isEmpty && !hasReadyAttachment);

  return (
    <div
      className={`${styles.composer} ${streaming ? styles.streaming : ""} ${disabled ? styles.disabled : ""}`}
    >
      <AttachmentRow attachments={attachments} onRemove={removeAttachment} disabled={disabled} />
      <MarkdownEditor
        ref={editorRef}
        placeholder="Message the model..."
        disabled={disabled}
        streaming={streaming}
        canSubmitEmpty={hasReadyAttachment}
        blockSubmit={uploadsPending}
        onSend={handleSend}
        onEmptyChange={setIsEmpty}
        onPasteFiles={addFiles}
      />
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept="image/*,application/pdf,.txt,.md,.csv,.json"
        onChange={handleFileInputChange}
        className={styles.hiddenFileInput}
        tabIndex={-1}
        aria-hidden="true"
      />
      <div className={styles.toolbar}>
        <div className={styles.toolbarGroup}>
          <button
            type="button"
            className={styles.iconBtn}
            disabled={disabled}
            aria-label="Attach files"
            title="Attach files"
            onClick={handleAttachClick}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">
              <path
                d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
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
              title={uploadsPending ? "Waiting for attachment upload…" : "Send message"}
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
