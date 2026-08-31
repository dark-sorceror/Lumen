"use client";

import type { MessageAttachment } from "@/hooks/useChatSocket";
import styles from "./MessageAttachments.module.css";

type Props = {
  attachments: MessageAttachment[];
  // Opens the attachment in the right-side media viewer. Only image/pdf
  // attachments that retained a blob preview URL are actually openable (see
  // `previewable` below); everything else renders as a non-interactive chip.
  onOpen: (attachment: MessageAttachment) => void;
};

// Read-only echo of a sent message's attachments, rendered ABOVE the
// message text inside the user bubble/column (see ChatMessage.tsx). Reuses
// AttachmentRow's visual language (image -> small cropped thumbnail, pdf ->
// red tile + name, file -> grey tile + name) but drops everything that only
// makes sense for a still-editable draft: no upload-status dot, no remove
// button. Image/pdf chips are clickable -> open in the media viewer.
export default function MessageAttachments({ attachments, onOpen }: Props) {
  if (attachments.length === 0) return null;

  return (
    <div className={styles.row}>
      {attachments.map((attachment, i) => {
        // Only image/pdf with a retained blob URL can be previewed (see
        // Composer's createObjectURL for image+pdf). A generic file, or a
        // pdf whose upload never finished (no URL), stays a plain chip.
        const previewable =
          !!attachment.previewUrl && (attachment.kind === "image" || attachment.kind === "pdf");
        const title = previewable ? `${attachment.name} — click to preview` : attachment.name;

        if (attachment.kind === "image" && attachment.previewUrl) {
          return (
            <button
              key={i}
              type="button"
              className={`${styles.imageChip} ${styles.clickable}`}
              title={title}
              onClick={() => onOpen(attachment)}
            >
              {/* eslint-disable-next-line @next/next/no-img-element -- a local
                  blob: preview URL (its ownership transferred to this message
                  on send, see Composer.handleSend), not a remote/optimizable
                  image, so next/image buys nothing here. */}
              <img src={attachment.previewUrl} alt={attachment.name} className={styles.imageThumb} />
            </button>
          );
        }

        const inner = (
          <>
            <div
              className={`${styles.fileTile} ${
                attachment.kind === "pdf" ? styles.fileTilePdf : styles.fileTileGeneric
              }`}
            >
              {attachment.kind === "pdf" ? <PdfIcon /> : <GenericFileIcon />}
            </div>
            <span className={styles.fileName}>{attachment.name}</span>
          </>
        );

        return previewable ? (
          <button
            key={i}
            type="button"
            className={`${styles.fileChip} ${styles.clickable}`}
            title={title}
            onClick={() => onOpen(attachment)}
          >
            {inner}
          </button>
        ) : (
          <div key={i} className={styles.fileChip} title={title}>
            {inner}
          </div>
        );
      })}
    </div>
  );
}

function PdfIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true">
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
    <svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true">
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
