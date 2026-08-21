"use client";

import { useEffect, useRef, useState } from "react";
import type { CacheImpact, Segment } from "@/hooks/useChatSocket";
import styles from "./ContextPanel.module.css";

type Props = {
  open: boolean;
  onClose: () => void;
  segments: Segment[] | null;
  streaming: boolean;
  cacheImpact: CacheImpact | null;
  editError: string | null;
  onRequestContext: () => void;
  onPreviewEdit: (segmentId: string, newText: string) => void;
  onApplyEdit: (segmentId: string, newText: string) => void;
};

const KIND_LABEL: Record<Segment["kind"], string> = {
  system: "system",
  user_msg: "user",
  assistant_msg: "assistant",
  thought: "thought",
  doc_chunk: "doc",
  scratch: "scratch",
};

const KIND_BADGE_CLASS: Record<Segment["kind"], string> = {
  system: styles.badgeSystem,
  user_msg: styles.badgeUser,
  assistant_msg: styles.badgeAssistant,
  thought: styles.badgeThought,
  doc_chunk: styles.badgeDoc,
  scratch: styles.badgeScratch,
};

// Debounce delay before an in-progress edit is sent as a preview_edit, so we
// don't hammer the socket on every keystroke.
const PREVIEW_DEBOUNCE_MS = 400;

function isEditable(seg: Segment): boolean {
  return seg.editable_by === "user" || seg.editable_by === "both";
}

export default function ContextPanel({
  open,
  onClose,
  segments,
  streaming,
  cacheImpact,
  editError,
  onRequestContext,
  onPreviewEdit,
  onApplyEdit,
}: Props) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftText, setDraftText] = useState("");
  const [pendingApply, setPendingApply] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Refresh on open; the hook also refreshes after every "done", so this
  // just covers the case where the panel is opened mid-session before any
  // turn has completed since the last edit.
  useEffect(() => {
    if (open) onRequestContext();
  }, [open, onRequestContext]);

  // Editing is disabled entirely while a generation is in flight (the
  // server rejects wire edits during streaming); bail out of any open
  // editor rather than leave it in a state that can only fail.
  useEffect(() => {
    if (streaming && editingId) {
      setEditingId(null);
      setPendingApply(false);
    }
  }, [streaming, editingId]);

  // Once the refreshed context shows our applied text for the segment we
  // were waiting on, the apply succeeded server-side -- close the editor.
  useEffect(() => {
    if (!pendingApply || !editingId || !segments) return;
    const seg = segments.find((s) => s.id === editingId);
    if (seg && seg.text === draftText) {
      setEditingId(null);
      setPendingApply(false);
    }
  }, [segments, pendingApply, editingId, draftText]);

  // If an apply we were waiting on gets rejected, stop waiting so the user
  // can see the error and retry or cancel.
  useEffect(() => {
    if (pendingApply && editError) setPendingApply(false);
  }, [pendingApply, editError]);

  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  const startEdit = (seg: Segment) => {
    setEditingId(seg.id);
    setDraftText(seg.text);
    setPendingApply(false);
  };

  const cancelEdit = () => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    setEditingId(null);
    setPendingApply(false);
  };

  const handleDraftChange = (segmentId: string, text: string) => {
    setDraftText(text);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      onPreviewEdit(segmentId, text);
    }, PREVIEW_DEBOUNCE_MS);
  };

  const handleApply = (segmentId: string) => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    setPendingApply(true);
    onApplyEdit(segmentId, draftText);
  };

  return (
    <>
      {open && <div className={styles.scrim} onClick={onClose} aria-hidden="true" />}
      <aside
        className={`${styles.panel} ${open ? styles.panelOpen : ""}`}
        aria-hidden={!open}
        aria-label="Context inspector"
      >
        <div className={styles.header}>
          <span className={styles.title}>Context</span>
          <button
            type="button"
            className={styles.closeBtn}
            onClick={onClose}
            aria-label="Close context inspector"
            title="Close"
          >
            <svg width="13" height="13" viewBox="0 0 13 13" aria-hidden="true">
              <path
                d="M1.5 1.5L11.5 11.5M11.5 1.5L1.5 11.5"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.4"
                strokeLinecap="round"
              />
            </svg>
          </button>
        </div>

        {streaming && (
          <div className={styles.streamingHint}>
            Pause or wait for the reply to finish to edit context.
          </div>
        )}

        {editError && <div className={styles.errorBanner}>{editError}</div>}

        <div className={styles.body}>
          {!segments || segments.length === 0 ? (
            <div className={styles.empty}>No context yet — send a message.</div>
          ) : (
            <ul className={styles.list}>
              {segments.map((seg) => {
                const editable = isEditable(seg) && !streaming;
                const isEditing = editingId === seg.id;
                const hasChanged = isEditing && draftText !== seg.text;
                // Apply is gated on a *landed* cache_impact preview that
                // matches both this segment and the exact draft text
                // currently on screen -- not just "some preview arrived at
                // some point". `cacheImpact` is tagged with the
                // segmentId/text it was computed for (see useChatSocket's
                // pendingEditsRef), so a preview that landed for a stale
                // draft (user kept typing) or for a different segment
                // (user switched editors) never counts as fresh here.
                const freshPreview =
                  hasChanged &&
                  cacheImpact !== null &&
                  cacheImpact.preview === true &&
                  cacheImpact.segmentId === seg.id &&
                  cacheImpact.text === draftText;
                const previewPending = hasChanged && !freshPreview;
                return (
                  <li
                    key={seg.id}
                    className={`${styles.card} ${seg.kind === "scratch" ? styles.cardMuted : ""}`}
                  >
                    <div className={styles.cardHeader}>
                      <span className={`${styles.badge} ${KIND_BADGE_CLASS[seg.kind]}`}>
                        {KIND_LABEL[seg.kind]}
                      </span>
                      <span className={styles.provenance}>{seg.provenance}</span>
                      {!isEditable(seg) && (
                        <svg
                          className={styles.lockIcon}
                          width="10"
                          height="10"
                          viewBox="0 0 10 10"
                          aria-hidden="true"
                        >
                          <rect
                            x="1.5"
                            y="4.5"
                            width="7"
                            height="4.5"
                            rx="1"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="1"
                          />
                          <path
                            d="M3 4.5V3a2 2 0 014 0v1.5"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="1"
                          />
                        </svg>
                      )}
                      {editable && !isEditing && (
                        <button
                          type="button"
                          className={styles.editBtn}
                          onClick={() => startEdit(seg)}
                        >
                          Edit
                        </button>
                      )}
                    </div>

                    {isEditing ? (
                      <div className={styles.editArea}>
                        <textarea
                          className={styles.textarea}
                          value={draftText}
                          autoFocus
                          onChange={(e) => handleDraftChange(seg.id, e.target.value)}
                        />
                        {freshPreview && cacheImpact && (
                          <div className={styles.impactNote}>
                            Editing re-prefills {cacheImpact.tokensToReprefill.toLocaleString()} token
                            {cacheImpact.tokensToReprefill === 1 ? "" : "s"}
                          </div>
                        )}
                        {previewPending && (
                          <div className={styles.impactNotePending}>Computing cost…</div>
                        )}
                        <div className={styles.editActions}>
                          <button
                            type="button"
                            className={styles.cancelBtn}
                            onClick={cancelEdit}
                            disabled={pendingApply}
                          >
                            Cancel
                          </button>
                          <button
                            type="button"
                            className={styles.applyBtn}
                            onClick={() => handleApply(seg.id)}
                            disabled={pendingApply || !freshPreview}
                            title={
                              previewPending
                                ? "Waiting for the cache-impact preview to land"
                                : undefined
                            }
                          >
                            {pendingApply ? "Applying…" : "Apply"}
                          </button>
                        </div>
                      </div>
                    ) : (
                      <pre className={styles.text}>{seg.text}</pre>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </aside>
    </>
  );
}
