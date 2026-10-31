// The drawing's open layers (the island, the turn's peek, the Context panel),
// newest last. Esc closes the top one only, and focus goes back to its opener.
export type Layer = { id: string; opener: string | null };

export function openLayer(stack: readonly Layer[], id: string, opener: string | null): Layer[] {
  return stack.some((l) => l.id === id) ? [...stack] : [...stack, { id, opener }];
}

export function closeLayer(stack: readonly Layer[], id: string): Layer[] {
  return stack.filter((l) => l.id !== id);
}

export function topLayer(stack: readonly Layer[]): Layer | null {
  return stack.length ? stack[stack.length - 1] : null;
}
