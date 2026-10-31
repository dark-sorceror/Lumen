// The landing page at / and the chat app at /chat/.

/** Where the chat app lives; with no fragment it opens a new chat, which
 * is all "Get started" does: there is no account to make. */
export const CHAT_PATH = "/chat/";

/** The two lines that install illumen's Claude Code plugin: add
 * the repository's plugin list, then install from it. The landing's "Get
 * plugin" shows and copies exactly these. */
export const PLUGIN_INSTALL = [
  "claude plugin marketplace add dark-sorceror/Lumen",
  "claude plugin install illumen@illumen",
] as const;

/** About how many tokens `text` is, at 4 characters a token: the hero
 * composer's "≈ +N tokens" while the person types. 0 for blank text. */
export function estimateTokens(text: string): number {
  const chars = text.trim().length;
  return chars === 0 ? 0 : Math.ceil(chars / 4);
}

/** A token count as the drawings print it: 640, 9.4k, 61k, 1M, 1.2M. */
export function tokensText(n: number): string {
  const v = Math.max(0, Math.round(n));
  if (Math.round(v / 1_000) >= 1_000) return `${parseFloat((v / 1_000_000).toFixed(1))}M`;
  if (v >= 10_000) return `${Math.round(v / 1_000)}k`;
  if (v >= 1_000) return `${parseFloat((v / 1_000).toFixed(1))}k`;
  return String(v);
}

// Every address the app made before it moved is /#c=<id> (lib/chats.ts),
// and a fragment never reaches the server, so the landing itself sends such
// an address on to /chat/. It forwards any fragment in the app's key=value
// form, #c= and any it adds; the landing's own anchors (#how-it-works)
// never hold "=", so they stay.
const APP_FRAGMENT = /^#[A-Za-z][\w-]*=/;

// How long the landing stays hidden for a navigation that never commits
// (one stopped by the person): then it shows after all.
const HIDDEN_AT_MOST_MS = 3000;

// The public site (illumen.lol, GitHub Pages) has no chat to forward to.
// This is isHosted (components/landing/served.ts) again, for a script that
// runs before React; tests/test_landing_hosted.py holds the two to the same
// hosts.
const NOT_WHERE_HOSTED =
  "var h=location.hostname.toLowerCase();" +
  'if(h==="illumen.lol"||/\\.(illumen\\.lol|github\\.io)$/.test(h))return;';

/** The forwarding as an inline script, for the very top of the landing's
 * body: it runs while the page is parsed, before the landing is drawn, and
 * again when the fragment changes on an open landing (a pasted /#c=<id>).
 * The query rides along. location.replace only starts the navigation and
 * the page goes on parsing and painting, so the document is hidden first:
 * an old link goes from nothing to the chat, never by way of the landing.
 * A landing kept for Back (the back-forward cache) comes back shown. */
export const FORWARD_SCRIPT =
  "(function(){" +
  NOT_WHERE_HOSTED +
  "var root=document.documentElement;" +
  'function show(){root.style.visibility=""}' +
  "function go(){" +
  `if(!${APP_FRAGMENT}.test(location.hash))return;` +
  'root.style.visibility="hidden";' +
  `location.replace(${JSON.stringify(CHAT_PATH)}+location.search+location.hash);` +
  `setTimeout(show,${HIDDEN_AT_MOST_MS})}` +
  'go();addEventListener("hashchange",go);' +
  'addEventListener("pageshow",function(e){if(e.persisted)show()})})();';

/** Where the landing leaves what was typed in its composer, for the chat
 * app to pick up. sessionStorage: it is this tab's, and it ends with it. */
export const LANDING_DRAFT_KEY = "lumen.landingDraft";

const DRAFT_MAX_CHARS = 20_000;
const DRAFT_MAX_AGE_MS = 30 * 60 * 1000;

/** Leaves `text` for the chat app's composer. Nothing happens for empty
 * text or when the storage is unavailable: the chat just opens empty. */
export function handOff(text: string, storage?: Pick<Storage, "setItem">): void {
  try {
    const trimmed = text.trim().slice(0, DRAFT_MAX_CHARS);
    if (!trimmed) return;
    (storage ?? window.sessionStorage).setItem(
      LANDING_DRAFT_KEY,
      JSON.stringify({ text: trimmed, at: Date.now() }),
    );
  } catch {
    // Storage blocked or full: the words are lost, the chat still opens.
  }
}

/** The text the landing left, or null. The entry is removed whatever it
 * held, so a draft is handed over once and never comes back later. */
export function takeLandingDraft(
  storage?: Pick<Storage, "getItem" | "removeItem">,
  now: number = Date.now(),
): string | null {
  try {
    const store = storage ?? window.sessionStorage;
    const raw = store.getItem(LANDING_DRAFT_KEY);
    if (raw === null) return null;
    store.removeItem(LANDING_DRAFT_KEY);
    const entry: unknown = JSON.parse(raw);
    if (typeof entry !== "object" || entry === null) return null;
    const { text, at } = entry as { text?: unknown; at?: unknown };
    if (typeof text !== "string" || !text.trim()) return null;
    if (typeof at !== "number" || now - at > DRAFT_MAX_AGE_MS) return null;
    return text;
  } catch {
    return null;
  }
}

/** Whether a key press in a landing composer sends it: Enter alone. Shift+
 * Enter is a new line, and Enter that confirms an IME candidate (CJK, say)
 * is not a send. */
export function isSendKey(event: { key: string; shiftKey: boolean; isComposing: boolean }): boolean {
  return event.key === "Enter" && !event.shiftKey && !event.isComposing;
}

/** What a landing composer does to send: leave `text` for the chat, then go
 * to it. `go` is `location.assign` on the page. Empty text just opens the
 * chat. */
export function startChat(text: string, go: (path: string) => void, storage?: Pick<Storage, "setItem">): void {
  if (text.trim()) handOff(text, storage);
  go(CHAT_PATH);
}
