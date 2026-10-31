import kit from "@/app/(landing)/page.module.css";

// The 16px line icons the lower sections draw.
const PATHS = {
  file: "M4.5 2h4.3L12 5.2v7.3a1.5 1.5 0 0 1-1.5 1.5h-6A1.5 1.5 0 0 1 3 12.5v-9A1.5 1.5 0 0 1 4.5 2zM8.6 2v3.4H12",
  quote: "M3 5.5h4v4H4.5L3 12zM9 5.5h4v4h-2.5L9 12z",
  agents: "M4 5.1v5.8M4 6.4c0 1.2.8 1.6 2.2 1.6h4.3M5.5 3.6a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0zM5.5 12.4a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0zM13.5 8a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0z",
  terminal: "M4 3h8a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2zM5 6.8l1.9 1.4L5 9.6M8.4 10h2.6",
  laptop: "M4.5 3.5h7a1 1 0 0 1 1 1V9a1 1 0 0 1-1 1h-7a1 1 0 0 1-1-1V4.5a1 1 0 0 1 1-1zM2 12.5h12",
  key: "M8 10.6a2.6 2.6 0 1 1-5.2 0 2.6 2.6 0 0 1 5.2 0zM7.3 8.7 13 3M11 5l1.5 1.5M9.4 6.6l1 1",
  check: "m3.5 8.5 3 3 6-7",
  up: "M8 13V3.5M4 7.2 8 3.2l4 4",
} as const;

export type SectionIconName = keyof typeof PATHS;

export function SIcon({ name }: { name: SectionIconName }) {
  return (
    <svg className={`${kit.ic} ${kit.icSm}`} viewBox="0 0 16 16" aria-hidden="true">
      <path d={PATHS[name]} />
    </svg>
  );
}

/** The up arrow at the normal size, for the send button. */
export function SIconLg({ name }: { name: SectionIconName }) {
  return (
    <svg className={kit.ic} viewBox="0 0 16 16" aria-hidden="true">
      <path d={PATHS[name]} />
    </svg>
  );
}
