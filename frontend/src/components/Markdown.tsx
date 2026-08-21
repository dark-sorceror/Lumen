"use client";

import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import styles from "./Markdown.module.css";

// Streaming cursor sentinel: rendered as an inline code span (see
// `withCursor` below) so it lands inline with the last line of text -- the
// same visual spot the old plain-text cursor occupied -- instead of on its
// own line the way a block-level trailing sibling would. The `code`
// component override below recognizes this exact glyph and swaps in the
// blinking-cursor span instead of normal inline-code styling.
const CURSOR_GLYPH = "▍";

// Appends the streaming cursor to a raw (possibly partial) markdown string.
// Wrapping it in backticks turns it into an inline code span in the
// rendered output, which keeps it inline with the current line of text. In
// the rare case the trailing text is inside an *unclosed* fenced code block,
// the backticks are just swallowed as literal code content -- a harmless
// cosmetic artifact, not a crash, and it self-corrects once the fence closes
// or the message finishes.
export function withCursor(text: string): string {
  return `${text}\`${CURSOR_GLYPH}\``;
}

function isCursorGlyph(children: React.ReactNode): boolean {
  if (typeof children === "string") return children === CURSOR_GLYPH;
  if (Array.isArray(children) && children.length === 1) {
    return children[0] === CURSOR_GLYPH;
  }
  return false;
}

const components: Components = {
  // Open links in a new tab without granting them access back to this
  // window/origin.
  a: ({ children, ...props }) => (
    <a {...props} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  ),
  code: ({ children, ...props }) => {
    if (isCursorGlyph(children)) {
      return <span className={styles.cursor} aria-hidden="true" />;
    }
    return <code {...props}>{children}</code>;
  },
};

type Props = {
  text: string;
  className?: string;
};

// Shared Markdown renderer for both user and assistant message text.
// GFM (tables/strikethrough/task lists/autolinks) via remark-gfm, plus
// rehype-highlight for fenced-code-block syntax highlighting (see
// Markdown.module.css for the light/dark token palette). rehype-highlight
// only walks `<pre><code>` elements it finds in the already-parsed hast tree
// and *adds* `hljs-*` classNames to the text it finds there -- it never
// introduces new elements from raw markup and never touches inline `code`
// (i.e. anything not directly inside a `<pre>`), so it can't affect the
// cursor glyph handling below. Crucially, rehype-raw is NOT registered here,
// so react-markdown's default behavior still applies: raw HTML in the
// source is never rendered as HTML (it's escaped to plain text) -- this is
// what keeps arbitrary user or model text from being able to inject
// markup/scripts.
export default function Markdown({ text, className }: Props) {
  return (
    <div className={`${styles.markdown} ${className ?? ""}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={components}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
