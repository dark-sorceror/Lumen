"use client";

import styles from "./DisconnectedBanner.module.css";

type Props = {
  connected: boolean;
};

// Purely presentational: explains *why* the composer is disabled (see
// `disabled={!connected}` on <Composer> in page.tsx). Reacts to the same
// `connected` flag the useChatSocket hook already exposes -- no socket,
// reconnect, or streaming logic lives here or is touched by this component.
//
// This is now a floating, rounded toast (position: fixed, bottom
// right) instead of a full-width banner pushed into the document flow --
// it never displaces the header, message list, or composer. It stays
// mounted at all times and animates opacity/transform between the two
// states so it can fade/slide in and out reactively as `connected` flips,
// rather than mounting/unmounting (which would skip the transition).
export default function DisconnectedBanner({ connected }: Props) {
  return (
    <div
      className={`${styles.toast} ${connected ? styles.hidden : styles.visible}`}
      role="status"
      aria-hidden={connected}
    >
      <span className={styles.dot} aria-hidden="true" />
      <span className={styles.text}>Server not reachable — start the backend to chat.</span>
    </div>
  );
}
