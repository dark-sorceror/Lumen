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
  /** Set once "done" -- byte size, effective mime, and the FULL extracted
   *  text the model sees, surfaced in the media viewer's info bar. */
  size?: number;
  mime?: string;
  text?: string;
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
  // tab's life. Per-attachment revocation on removal/send is handled where
  // those happen (removeAttachment, handleSend) -- this is only the
  // unmount-time backstop for whatever's left.
  useEffect(() => {
    return () => {
      for (const attachment of attachmentsRef.current) {
        if (attachment.previewUrl) URL.revokeObjectURL(attachment.previewUrl);
      }
    };
  }, []);

  // Eagerly POSTs a single attachment to the backend the moment it's added
  // (rather than waiting for send -- see the module doc on UploadState).
  // Updates are always keyed by the local attachment `id` and go through the
  // functional setState form, so a slow response landing after the
  // attachment was removed (or after others finished) both (a) never
  // clobbers a sibling's concurrent update and (b) simply no-ops if `id` is
  // no longer in the list (removeAttachment already dropped it).
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
        return res.json() as Promise<{
          id: string;
          chars?: number;
          size?: number;
          mime?: string;
          text?: string;
        }>;
      })
      .then((json) => {
        setAttachments((prev) =>
          prev.map((a) =>
            a.id === id
              ? {
                  ...a,
                  uploadState: "done",
                  serverId: json.id,
                  chars: json.chars,
                  size: json.size,
                  mime: json.mime,
                  text: json.text,
                }
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
          // images (AttachmentRow/MessageAttachments branch on kind), so a
          // pdf URL here changes nothing visually until it's clicked. Object-
          // URL lifecycle (transfer-on-send for "done", revoke otherwise) is
          // already keyed on `previewUrl` presence, not kind, in handleSend /
          // removeAttachment / the unmount cleanup -- so pdf URLs are managed
          // identically to image ones with no further changes.
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
  // actually been committed (Enter, or the Send button below calling
  // editorRef.current.submit()); the editor has already cleared itself by
  // the time this fires. MarkdownEditor now permits an empty-text commit as
  // long as attachments are present (see its `hasAttachments` prop), so this
  // can run with `markdown === ""`.
  const handleSend = (markdown: string) => {
    // Only attachments that finished uploading have a serverId the backend
    // can resolve; a still-uploading or errored attachment is silently
    // dropped from this message rather than blocking send -- simplest and
    // most predictable behavior (see task notes). In the rare case where
    // every attachment is still mid-upload and the text is also empty,
    // this sends a text-only, attachment-less message (MarkdownEditor's
    // `hasAttachments` gate still lets it through since it only checks
    // attachment *presence*, not upload completion) rather than blocking
    // the composer on network latency.
    const doneAttachments = attachments.filter(
      (a): a is Attachment & { serverId: string } => a.uploadState === "done" && !!a.serverId,
    );
    const serverIds = doneAttachments.map((a) => a.serverId);
    // Display copy carried into the chat message itself (see ChatMessage's
    // attachment strip + footer token estimate) -- built from the same
    // "done" set as serverIds, so what's shown in the chat always matches
    // what was actually sent to the backend.
    const attachmentsDisplay: MessageAttachment[] = doneAttachments.map((a) => ({
      name: a.name,
      kind: a.kind,
      previewUrl: a.previewUrl,
      chars: a.chars,
      size: a.size,
      mime: a.mime,
      text: a.text,
    }));
    onSend(markdown, serverIds, attachmentsDisplay.length > 0 ? attachmentsDisplay : undefined);
    // Object-URL lifecycle: image previewUrls for attachments carried into
    // the message above transfer ownership to that message -- it's now
    // displayed in the chat log for the rest of the session, so revoking
    // here would blank the thumbnail out from under it. Only revoke
    // previewUrls of attachments that did NOT make it into the message
    // (still-uploading/error ones normally never have one anyway, since
    // only images get a previewUrl and image uploads are typically fast /
    // don't error); the unmount-cleanup effect above remains the backstop
    // for whatever's left in `attachments` state after this.
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
                {/* The chevron head packs more stroke length into the top
                    half than the bare shaft does into the bottom half, so a
                    path that's geometrically centered in the viewBox (equal
                    3px margins top/bottom) still reads as sitting slightly
                    high -- its ink-weighted visual center sits above the
                    box's true center. Nudging the whole path down by 0.5
                    units re-balances the visual weight without the
                    overcorrection a full mathematical fix would introduce. */}
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
