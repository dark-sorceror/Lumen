"use client";

import styles from "./ConnectionDot.module.css";

type Props = {
  connected: boolean;
};

export default function ConnectionDot({ connected }: Props) {
  return (
    <span
      className={`${styles.dot} ${connected ? styles.on : styles.off}`}
      title={connected ? "connected" : "disconnected"}
      aria-label={connected ? "connected" : "disconnected"}
    />
  );
}
