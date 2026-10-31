"use client";

import { type RefObject, useEffect, useState } from "react";

/** Whether the element behind `ref` is on screen. With `once`, it stays
 * true from the first time it shows: a part that reveals or plays once.
 * Without, it goes false again off screen, which is how a loop pauses.
 * Where IntersectionObserver is missing, everything counts as on screen. */
export function useInView(
  ref: RefObject<Element | null>,
  { once = false, margin = "0px", threshold = 0 }: { once?: boolean; margin?: string; threshold?: number } = {},
): boolean {
  const [inView, setInView] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (typeof IntersectionObserver === "undefined") {
      setInView(true);
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        // A busy page can queue an enter and a leave together; the last
        // says where the part is now.
        const entry = entries[entries.length - 1];
        // isIntersecting is true from the first pixel, whatever the
        // threshold; the share seen is what the threshold asks about.
        const seen = entry.isIntersecting && entry.intersectionRatio >= threshold;
        setInView(seen);
        if (seen && once) observer.disconnect();
      },
      { rootMargin: margin, threshold },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [ref, once, margin, threshold]);
  return inView;
}
