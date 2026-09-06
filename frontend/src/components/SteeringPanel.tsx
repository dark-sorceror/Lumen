"use client";

import { useEffect, useState } from "react";
import type { ActiveConfig } from "@/hooks/useChatSocket";
import styles from "./SteeringPanel.module.css";

type Props = {
  config: ActiveConfig | null;
  configNames: string[];
  streaming: boolean;
  onDerive: (positive: string[], negative: string[], layer: number,
             strength: number, label: string) => void;
  onClear: () => void;
  onSave: (name: string) => void;
  onLoad: (name: string) => void;
  onList: () => void;
};

// Blank lines are dropped so a trailing newline in the textarea is not sent as
// an empty prompt (which the server rejects).
const lines = (s: string) => s.split("\n").map((l) => l.trim()).filter(Boolean);

export default function SteeringPanel({
  config, configNames, streaming, onDerive, onClear, onSave, onLoad, onList,
}: Props) {
  const [open, setOpen] = useState(false);
  const [positive, setPositive] = useState("I love this\nWhat a joyful day");
  const [negative, setNegative] = useState("I hate this\nWhat a bleak day");
  const [layer, setLayer] = useState("18");
  const [strength, setStrength] = useState("16");
  const [label, setLabel] = useState("tone");
  const [name, setName] = useState("");

  useEffect(() => {
    if (open) onList();
  }, [open, onList]);

  const active = config?.steering ?? [];
  const canDerive = lines(positive).length > 0 && lines(negative).length > 0 && !streaming;

  return (
    <div className={styles.wrap}>
      <button
        type="button"
        className={styles.toggle}
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span>Steering</span>
        <span className={styles.summary}>
          {active.length === 0
            ? "none"
            : active.map((s) => `L${s.layer} ${s.strength > 0 ? "+" : ""}${s.strength}`).join(", ")}
        </span>
      </button>

      {open && (
        <div className={styles.body}>
          {/* A direction is derived from what the two prompt sets DIFFER in, so
              they should differ in one thing only. */}
          <label className={styles.field}>
            <span>positive prompts</span>
            <textarea rows={3} value={positive} onChange={(e) => setPositive(e.target.value)} />
          </label>
          <label className={styles.field}>
            <span>negative prompts</span>
            <textarea rows={3} value={negative} onChange={(e) => setNegative(e.target.value)} />
          </label>
          <div className={styles.row}>
            <label className={styles.small}>
              <span>layer</span>
              <input value={layer} onChange={(e) => setLayer(e.target.value)} inputMode="numeric" />
            </label>
            <label className={styles.small}>
              <span>strength</span>
              <input value={strength} onChange={(e) => setStrength(e.target.value)} inputMode="numeric" />
            </label>
            <label className={styles.small}>
              <span>label</span>
              <input value={label} onChange={(e) => setLabel(e.target.value)} />
            </label>
          </div>
          <div className={styles.row}>
            <button
              type="button"
              className={styles.action}
              disabled={!canDerive}
              onClick={() =>
                onDerive(lines(positive), lines(negative),
                         Number(layer) || 0, Number(strength) || 0, label)
              }
            >
              Derive
            </button>
            <button type="button" className={styles.action} onClick={() => onClear()}>
              Clear
            </button>
          </div>

          <div className={styles.row}>
            <input
              className={styles.nameInput}
              placeholder="config name"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
            <button
              type="button"
              className={styles.action}
              disabled={!name.trim()}
              onClick={() => onSave(name.trim())}
            >
              Save
            </button>
          </div>
          {configNames.length > 0 && (
            <div className={styles.saved}>
              {configNames.map((n) => (
                <button key={n} type="button" className={styles.chip} onClick={() => onLoad(n)}>
                  {n}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
