// Where an in-page link should land: just under the sticky nav, with a
// little air; the logo's "#top" goes to the very top.
export function glideY(targetTop: number | null, scrollY: number, navHeight: number): number {
  if (targetTop === null) return 0;
  return Math.max(0, Math.round(targetTop + scrollY - navHeight - 12));
}

// The id a link points at on this page: "top" for "#" or "#top", the id for
// "#name", and null for anything that leaves the page.
export function inPageId(href: string | null): string | null {
  if (!href || !href.startsWith("#")) return null;
  const id = href.slice(1);
  return id === "" ? "top" : id;
}
