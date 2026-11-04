// The landing is served two ways: by the app on your Mac (localhost), where
// "Get started" opens a new chat, and as the public site (illumen.lol, or
// GitHub Pages), where it can't reach anyone's Mac and explains how to install.
export function isHosted(hostname: string): boolean {
  const h = hostname.toLowerCase();
  return h === "illumen.lol" || h.endsWith(".illumen.lol") || h.endsWith(".github.io");
}
