"use client";

import type { Attachment } from "./Composer";
import styles from "./AttachmentRow.module.css";

type Props = {
  attachments: Attachment[];
  onRemove: (id: string) => void;
  disabled: boolean;
};

// Rendered inside the composer, above the Lexical editor, only when there's
// something to show -- returning null (rather than an empty wrapper) keeps
// the composer's height/gap untouched when there are no attachments.
export default function AttachmentRow({ attachments, onRemove, disabled }: Props) {
  if (attachments.length === 0) return null;

  return (
    <div className={styles.row}>
      {attachments.map((attachment) =>
        attachment.kind === "image" ? (
          <ImageChip
            key={attachment.id}
            attachment={attachment}
            onRemove={onRemove}
            disabled={disabled}
          />
        ) : (
          <FileChip
            key={attachment.id}
            attachment={attachment}
            onRemove={onRemove}
            disabled={disabled}
          />
        ),
      )}
    </div>
  );
}

// Small subtle status indicator for the eager-upload flow: a pulsing dot
// while the POST is in flight, a solid red dot on failure, and nothing at
// all once uploadState is "done" (a finished attachment looks exactly like
// it always did). Kept as a standalone dot rather than replacing the
// preview/icon so the chip's identity (thumbnail, filename) never flashes
// away mid-upload.
function StatusDot({ uploadState }: { uploadState: Attachment["uploadState"] }) {
  if (uploadState === "done") return null;
  return (
    <span
      className={`${styles.statusDot} ${
        uploadState === "uploading" ? styles.statusDotUploading : styles.statusDotError
      }`}
      aria-hidden="true"
    />
  );
}

function chipTitle(attachment: Attachment): string {
  if (attachment.uploadState === "error") {
    return `${attachment.name} — upload failed${attachment.errorMsg ? `: ${attachment.errorMsg}` : ""}`;
  }
  if (attachment.uploadState === "uploading") {
    return `${attachment.name} — uploading…`;
  }
  return attachment.name;
}

function ImageChip({
  attachment,
  onRemove,
  disabled,
}: {
  attachment: Attachment;
  onRemove: (id: string) => void;
  disabled: boolean;
}) {
  return (
    <div
      className={`${styles.imageChip} ${
        attachment.uploadState === "error" ? styles.chipError : ""
      }`}
      title={chipTitle(attachment)}
    >
      {/* eslint-disable-next-line @next/next/no-img-element -- a local blob:
          preview URL, not a remote/optimizable image, so next/image buys
          nothing here. */}
      <img src={attachment.previewUrl} alt={attachment.name} className={styles.imageThumb} />
      <StatusDot uploadState={attachment.uploadState} />
      <button
        type="button"
        className={styles.removeBtnOverlay}
        aria-label={`Remove ${attachment.name}`}
        title={`Remove ${attachment.name}`}
        disabled={disabled}
        onClick={() => onRemove(attachment.id)}
      >
        <CloseIcon />
      </button>
    </div>
  );
}

function FileChip({
  attachment,
  onRemove,
  disabled,
}: {
  attachment: Attachment;
  onRemove: (id: string) => void;
  disabled: boolean;
}) {
  const isPdf = attachment.kind === "pdf";
  return (
    <div
      className={`${styles.fileChip} ${
        attachment.uploadState === "error" ? styles.chipError : ""
      }`}
      title={chipTitle(attachment)}
    >
      <div
        className={`${styles.fileTile} ${isPdf ? styles.fileTilePdf : styles.fileTileGeneric}`}
      >
        {isPdf ? <PdfIcon /> : <GenericFileIcon />}
        <StatusDot uploadState={attachment.uploadState} />
      </div>
      <span className={styles.fileName}>{attachment.name}</span>
      <button
        type="button"
        className={styles.removeBtnInline}
        aria-label={`Remove ${attachment.name}`}
        title={`Remove ${attachment.name}`}
        disabled={disabled}
        onClick={() => onRemove(attachment.id)}
      >
        <CloseIcon />
      </button>
    </div>
  );
}

function CloseIcon() {
  return (
    <svg width="9" height="9" viewBox="0 0 10 10" aria-hidden="true">
      <path
        d="M1.5 1.5L8.5 8.5M8.5 1.5L1.5 8.5"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  );
}

function PdfIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M6 2h8l6 6v14a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1z"
        fill="none"
        stroke="#fff"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
      <path d="M14 2v6h6" fill="none" stroke="#fff" strokeWidth="1.6" strokeLinejoin="round" />
      <path
        d="M8 13.5h1.2a1.15 1.15 0 0 1 0 2.3H8v-2.3zM8 15.8v2.2M11.6 13.5v4.5M11.6 13.5h1a1.25 1.25 0 0 1 1.25 1.25v2a1.25 1.25 0 0 1-1.25 1.25h-1M15.6 13.5v4.5M15.6 15.6h1.5"
        fill="none"
        stroke="#fff"
        strokeWidth="1"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function GenericFileIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M6 2h8l6 6v14a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1z"
        fill="none"
        stroke="#fff"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
      <path d="M14 2v6h6" fill="none" stroke="#fff" strokeWidth="1.6" strokeLinejoin="round" />
      <path d="M8 13h8M8 16h8M8 19h5" stroke="#fff" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}
