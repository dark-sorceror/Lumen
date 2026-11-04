import type { ReactNode } from "react";
import styles from "./Hero.module.css";

/** The row of ways into the drawing under the window, for either view: `lead`
 * heads it, and its buttons are the drawing's own. Its styles are the hero's,
 * which owns the slot under the window. */
export default function TryRow({
  id,
  label,
  lead,
  children,
}: {
  id: string;
  label: string;
  lead: string;
  children: ReactNode;
}) {
  return (
    <div className={styles.try} id={id} role="group" aria-label={label}>
      <span className={styles.tryLbl}>{lead}</span>
      {children}
    </div>
  );
}
