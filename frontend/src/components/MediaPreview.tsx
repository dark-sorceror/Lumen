"use client";

import { useEffect } from "react";
import type { MessageAttachment } from "@/hooks/useChatSocket";
import styles from "./MediaPreview.module.css";

type Props = {
  attachment: MessageAttachment;
  onClose: () => void;
};

export default function MediaPreview({ attachment, onClose }: Props) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const canPreview =
    !!attachment.previewUrl && (attachment.kind === "image" || attachment.kind === "pdf");

  return (
    <aside className={styles.panel} aria-label="Attachment preview">
      <header className={styles.header}>
        <span className={styles.name} title={attachment.name}>
          {attachment.name}
        </span>
        <button
          type="button"
          className={styles.closeBtn}
          onClick={onClose}
          aria-label="Close preview"
          title="Close preview"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
            <path d="M4 4l8 8M12 4l-8 8" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
        </button>
      </header>

      <div className={styles.body}>
        {!canPreview ? (
          <div className={styles.empty}>No preview available for this file.</div>
        ) : attachment.kind === "image" ? (
          // eslint-disable-next-line @next/next/no-img-element -- local blob:
          // preview URL, not a remote/optimizable image.
          <img src={attachment.previewUrl} alt={attachment.name} className={styles.image} />
        ) : (
          <iframe src={attachment.previewUrl} className={styles.frame} title={attachment.name} />
        )}
      </div>
    </aside>
  );
}
