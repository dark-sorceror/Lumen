"use client";

import { useEffect, useState } from "react";
import type { MessageAttachment } from "@/hooks/useChatSocket";
import styles from "./MediaPreview.module.css";

type Props = {
  attachment: MessageAttachment;
  onClose: () => void;
};

function formatBytes(bytes: number | undefined): string | null {
  if (bytes == null) return null;
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb < 10 ? kb.toFixed(1) : Math.round(kb)} KB`;
  const mb = kb / 1024;
  return `${mb < 10 ? mb.toFixed(1) : Math.round(mb)} MB`;
}

// ~chars/4 token estimate (same heuristic used elsewhere -- no client
// tokenizer), formatted compactly ("~1.2k tok" / "~840 tok").
function formatTokens(chars: number | undefined): string | null {
  if (!chars) return null;
  const tok = Math.max(1, Math.round(chars / 4));
  return tok >= 1000 ? `~${(tok / 1000).toFixed(1)}k tok` : `~${tok} tok`;
}

// "image/png" -> "PNG", "application/pdf" -> "PDF", else a reasonable label.
function formatFormat(mime: string | undefined, kind: MessageAttachment["kind"]): string {
  if (mime) {
    if (mime === "application/pdf") return "PDF";
    const sub = mime.split("/")[1];
    if (sub) return sub.split("+")[0].toUpperCase();
  }
  return kind.toUpperCase();
}

export default function MediaPreview({ attachment, onClose }: Props) {
  const [dims, setDims] = useState<{ w: number; h: number } | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Reset measured dimensions when the shown attachment changes.
  useEffect(() => {
    setDims(null);
  }, [attachment.previewUrl, attachment.name]);

  const canPreview =
    !!attachment.previewUrl && (attachment.kind === "image" || attachment.kind === "pdf");

  // Compact metadata row (70% panel top): dimensions (images) · format · tokens.
  const metaBits = [
    dims && dims.w > 0 && dims.h > 0 ? `${dims.w} × ${dims.h}` : null,
    formatFormat(attachment.mime, attachment.kind),
    formatTokens(attachment.chars),
  ].filter(Boolean);

  const modelText = (attachment.text ?? "").trim();
  const sizeLabel = formatBytes(attachment.size);

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
          <img
            src={attachment.previewUrl}
            alt={attachment.name}
            className={styles.image}
            onLoad={(e) => setDims({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
          />
        ) : (
          <iframe src={attachment.previewUrl} className={styles.frame} title={attachment.name} />
        )}
      </div>

      {/* 70/30 info bar: LEFT = metadata + the extracted text the model sees;
          RIGHT = file identity (name/size/type). */}
      <div className={styles.infoBar}>
        <div className={styles.meta}>
          {metaBits.length > 0 && (
            <div className={styles.metaRow}>
              {metaBits.map((bit, i) => (
                <span key={i} className={styles.metaBit}>
                  {i > 0 && <span className={styles.metaDot} aria-hidden="true">·</span>}
                  {bit}
                </span>
              ))}
            </div>
          )}
          <div className={styles.metaLabel}>What the model sees</div>
          <div className={styles.metaText}>
            {modelText ? modelText : <span className={styles.metaMuted}>No text extracted.</span>}
          </div>
        </div>
        <div className={styles.file}>
          <div className={styles.fileRow}>
            <span className={styles.fileKey}>Name</span>
            <span className={styles.fileVal} title={attachment.name}>{attachment.name}</span>
          </div>
          {sizeLabel && (
            <div className={styles.fileRow}>
              <span className={styles.fileKey}>Size</span>
              <span className={styles.fileVal}>{sizeLabel}</span>
            </div>
          )}
          <div className={styles.fileRow}>
            <span className={styles.fileKey}>Type</span>
            <span className={styles.fileVal}>{attachment.mime ?? attachment.kind}</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
