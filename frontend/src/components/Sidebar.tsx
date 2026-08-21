"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import styles from "./Sidebar.module.css";

// Layout-only scaffolding for future chat persistence.
// There is no chat-history state anywhere in the app yet -- this is a
// static list purely so the ChatGPT/Claude-style shell reads correctly.
// "New chat" and every row below are inert (no onClick behavior beyond a
// no-op) until a backend-backed conversation list exists to wire them to.
const PLACEHOLDER_CHATS = [
  "Untitled chat",
  "France vs Japan",
  "Inference engine notes",
  "Debugging the KV cache eviction",
  "Prompt caching strategy",
  "Context window experiments",
];

// "New chat" split/dropdown button. The main segment stays the
// plain "New chat" action; the caret opens a menu with the plain action
// plus two future variants. Open/close is real local UI state (no
// backend involved), but every menu action itself is still a no-op --
// skill templates and contained mode don't exist yet.
function NewChatSplitButton() {
  const [open, setOpen] = useState(false);
  const [menuPos, setMenuPos] = useState<{ top: number; left: number; width: number } | null>(
    null,
  );
  const [mounted, setMounted] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  // The menu is portaled to document.body (below), so `document` must
  // exist before we ever try to render it -- true after mount, never
  // during the static export's server render.
  useEffect(() => {
    setMounted(true);
  }, []);

  const openMenu = () => {
    const rect = wrapRef.current?.getBoundingClientRect();
    if (rect) {
      setMenuPos({ top: rect.bottom + 6, left: rect.left, width: rect.width });
    }
    setOpen(true);
  };

  // Close on outside click or Escape.
  useEffect(() => {
    if (!open) return;
    const handlePointerDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (wrapRef.current?.contains(target) || menuRef.current?.contains(target)) return;
      setOpen(false);
    };
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [open]);

  const closeAnd = (fn: () => void) => () => {
    setOpen(false);
    fn();
  };

  // Portaled to document.body rather than positioned absolute inside
  // .sidebar: the sidebar has `overflow: hidden` (for its scrollable chat
  // list), which would clip an absolutely-positioned menu. `position:
  // fixed` + a portal escapes that ancestor entirely; menuPos is a
  // viewport-relative snapshot of the button's rect taken on open.
  const menu =
    mounted && open && menuPos
      ? createPortal(
          <div
            ref={menuRef}
            role="menu"
            aria-label="New chat options"
            className={styles.newChatMenu}
            style={{ top: menuPos.top, left: menuPos.left, minWidth: menuPos.width }}
          >
            <button
              type="button"
              role="menuitem"
              className={styles.newChatMenuItem}
              onClick={closeAnd(() => {
                // TODO: create a new conversation once chat
                // persistence lands on the backend. No-op for now.
              })}
            >
              <span className={styles.newChatMenuIcon} aria-hidden="true">
                +
              </span>
              New chat
            </button>
            <button
              type="button"
              role="menuitem"
              className={styles.newChatMenuItem}
              onClick={closeAnd(() => {
                // TODO: skill templates don't exist yet --
                // this is scaffolding only. No-op for now.
              })}
            >
              <svg
                className={styles.newChatMenuIcon}
                width="13"
                height="13"
                viewBox="0 0 14 14"
                aria-hidden="true"
                fill="currentColor"
              >
                <path d="M7 1l1.1 3.4L11.5 5.5 8.1 6.6 7 10l-1.1-3.4L2.5 5.5l3.4-1.1z" />
                <path d="M11.5 8.5l.5 1.5 1.5.5-1.5.5-.5 1.5-.5-1.5-1.5-.5 1.5-.5z" />
              </svg>
              New chat with skill template
            </button>
            <button
              type="button"
              role="menuitem"
              className={styles.newChatMenuItem}
              onClick={closeAnd(() => {
                // TODO: contained/document-grounded mode is a
                // future feature. No-op for now.
              })}
            >
              <svg
                className={styles.newChatMenuIcon}
                width="13"
                height="13"
                viewBox="0 0 14 14"
                aria-hidden="true"
              >
                <path
                  d="M3 1.5h5l3 3v8a.5.5 0 01-.5.5h-8a.5.5 0 01-.5-.5v-10a.5.5 0 01.5-.5z"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1"
                />
                <path d="M8 1.5v3h3" fill="none" stroke="currentColor" strokeWidth="1" />
                <rect
                  x="5"
                  y="7.6"
                  width="4"
                  height="3.2"
                  rx="0.6"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1"
                />
                <path
                  d="M5.8 7.6V6.6a1.2 1.2 0 012.4 0v1"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1"
                />
              </svg>
              New chat in contained mode
            </button>
          </div>,
          document.body,
        )
      : null;

  return (
    <div className={styles.newChatWrap} ref={wrapRef}>
      <button
        type="button"
        className={styles.newChatBtn}
        onClick={() => {
          // TODO: create a new conversation once chat persistence
          // lands on the backend. No-op for now.
        }}
      >
        <svg
          className={styles.newChatIcon}
          width="16"
          height="16"
          viewBox="0 0 16 16"
          aria-hidden="true"
        >
          <path
            d="M8 4v8M4 8h8"
            stroke="currentColor"
            strokeWidth="1.75"
            strokeLinecap="round"
          />
        </svg>
        <span className={styles.newChatLabel}>New chat</span>
      </button>
      <button
        type="button"
        className={styles.newChatCaretBtn}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label="More new chat options"
        onClick={() => (open ? setOpen(false) : openMenu())}
      >
        <svg
          className={styles.newChatCaretIcon}
          width="16"
          height="16"
          viewBox="0 0 16 16"
          aria-hidden="true"
        >
          <path
            d="M4 6L8 10L12 6"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>
      {menu}
    </div>
  );
}

export default function Sidebar() {
  return (
    <aside className={styles.sidebar} aria-label="Chat history">
      <NewChatSplitButton />
      <nav className={styles.history} aria-label="Previous chats">
        <div className={styles.historyLabel}>Recent</div>
        <ul className={styles.historyList}>
          {PLACEHOLDER_CHATS.map((title) => (
            <li key={title}>
              <button
                type="button"
                className={styles.historyItem}
                title={title}
                onClick={() => {
                  // TODO: load the selected conversation once chat
                  // persistence lands on the backend. No-op for now.
                }}
              >
                {title}
              </button>
            </li>
          ))}
        </ul>
      </nav>
      {/* Bottom-pinned Account/Settings/GitHub scaffolding.
          `.history` above has flex: 1 (see Sidebar.module.css), so this
          section is naturally pushed to the bottom of the sidebar without
          needing margin-top: auto. Layout-only -- no account/settings
          surfaces exist yet, so these are inert. */}
      <div className={styles.bottomSection}>
        {/* GitHub sits topmost of the three. TODO:
            point at the real GitHub repo URL once one exists publicly;
            href="#" is a placeholder. */}
        <a href="#" className={styles.bottomItem}>
          <svg
            className={styles.bottomIcon}
            width="14"
            height="14"
            viewBox="0 0 16 16"
            aria-hidden="true"
            fill="currentColor"
          >
            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8Z" />
          </svg>
          GitHub
        </a>
        <button
          type="button"
          className={styles.bottomItem}
          onClick={() => {
            // TODO: no account surface exists yet. No-op for now.
          }}
        >
          <svg
            className={styles.bottomIcon}
            width="14"
            height="14"
            viewBox="0 0 14 14"
            aria-hidden="true"
          >
            <circle cx="7" cy="4.5" r="2.5" fill="none" stroke="currentColor" strokeWidth="1.2" />
            <path
              d="M2 12c0-2.76 2.24-4.5 5-4.5s5 1.74 5 4.5"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.2"
              strokeLinecap="round"
            />
          </svg>
          Account
        </button>
        <button
          type="button"
          className={styles.bottomItem}
          onClick={() => {
            // TODO: no settings surface exists yet. No-op for now.
          }}
        >
          <svg
            className={styles.bottomIcon}
            width="14"
            height="14"
            viewBox="0 0 14 14"
            aria-hidden="true"
          >
            <circle cx="7" cy="7" r="2" fill="none" stroke="currentColor" strokeWidth="1.2" />
            <path
              d="M7 1.2v1.4M7 11.4v1.4M12.8 7h-1.4M2.6 7H1.2M11.06 2.94l-.99.99M3.93 10.07l-.99.99M11.06 11.06l-.99-.99M3.93 3.93l-.99-.99"
              stroke="currentColor"
              strokeWidth="1.1"
              strokeLinecap="round"
            />
          </svg>
          Settings
        </button>
      </div>
    </aside>
  );
}
