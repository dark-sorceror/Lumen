import kit from "@/app/(landing)/page.module.css";

// The few 16px line icons the landing's frame draws: strokes in the text
// colour, decorative (a word always sits by them).
const PATHS = {
  arrow: "M3 8h9.5M8.8 4.3 12.5 8l-3.7 3.7",
  up: "M8 13V3.5M4 7.2 8 3.2l4 4",
  check: "m3.5 8.5 3 3 6-7",
  laptop: "M3.5 3.5h9v6.5h-9zM2 12.5h12",
  copy: "M5.5 5.5h8v8h-8zM10.5 5.5V4A1.5 1.5 0 0 0 9 2.5H4A1.5 1.5 0 0 0 2.5 4v5A1.5 1.5 0 0 0 4 10.5h1.5",
} as const;

export function Icon({ name, small = false }: { name: keyof typeof PATHS; small?: boolean }) {
  return (
    <svg className={small ? `${kit.ic} ${kit.icSm}` : kit.ic} viewBox="0 0 16 16" aria-hidden="true">
      <path d={PATHS[name]} />
    </svg>
  );
}

/** Claude Code's own mark as its banner prints it, a text glyph in its
 * orange: no logo or image of Anthropic's. */
export function ClaudeMark() {
  return (
    <span className={kit.claude} aria-hidden="true">
      ✻
    </span>
  );
}
