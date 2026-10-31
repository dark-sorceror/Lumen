// What the landing's scripted motion asks before it moves anything.

/** Whether the person asked for less motion: then a part is simply in its
 * new place, with every state still there to reach. */
export function reducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}
