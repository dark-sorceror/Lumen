"use client";

import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
} from "react";
import {
  $getRoot,
  COMMAND_PRIORITY_HIGH,
  COMMAND_PRIORITY_LOW,
  KEY_ENTER_COMMAND,
  PASTE_COMMAND,
  type LexicalEditor,
} from "lexical";
import { LexicalComposer } from "@lexical/react/LexicalComposer";
import { useLexicalComposerContext } from "@lexical/react/LexicalComposerContext";
import { RichTextPlugin } from "@lexical/react/LexicalRichTextPlugin";
import { ContentEditable } from "@lexical/react/LexicalContentEditable";
import { HistoryPlugin } from "@lexical/react/LexicalHistoryPlugin";
import { ListPlugin } from "@lexical/react/LexicalListPlugin";
import { LinkPlugin } from "@lexical/react/LexicalLinkPlugin";
import { OnChangePlugin } from "@lexical/react/LexicalOnChangePlugin";
import { EditorRefPlugin } from "@lexical/react/LexicalEditorRefPlugin";
import { LexicalErrorBoundary } from "@lexical/react/LexicalErrorBoundary";
import { MarkdownShortcutPlugin } from "@lexical/react/LexicalMarkdownShortcutPlugin";
import { $convertToMarkdownString, TRANSFORMERS } from "@lexical/markdown";
import { HeadingNode, QuoteNode } from "@lexical/rich-text";
import { ListNode, ListItemNode } from "@lexical/list";
import { CodeNode } from "@lexical/code";
import { LinkNode } from "@lexical/link";
import styles from "./MarkdownEditor.module.css";

export type MarkdownEditorHandle = {
  /** Focuses the editor (used after send, mirroring the old textarea focus). */
  focus: () => void;
  /** Same submit path Enter uses; wired to the toolbar Send button. */
  submit: () => void;
};

type Props = {
  placeholder: string;
  disabled: boolean;
  streaming: boolean;
  // Allow an empty-text send only when there's a fully-uploaded attachment
  // to carry (a file with no message). Separately, `blockSubmit` hard-blocks
  // Enter while any attachment is still uploading, so a send can never drop a
  // not-yet-ready attachment. Both mirror Composer's computed gate.
  canSubmitEmpty: boolean;
  blockSubmit: boolean;
  onSend: (markdown: string) => void;
  onEmptyChange: (empty: boolean) => void;
  onPasteFiles: (files: File[]) => void;
};

// Splits Enter (send) from Shift+Enter (newline). Runs at
// COMMAND_PRIORITY_HIGH, which fires *before* the plain-text/rich-text
// default handler that @lexical/rich-text's RichTextPlugin registers at
// COMMAND_PRIORITY_EDITOR (the lowest priority) -- see
// node_modules/@lexical/rich-text/src/index.ts. Returning `true` stops the
// command from reaching that default handler, so a plain Enter never
// inserts a new paragraph; returning `false` for Shift+Enter lets the
// default handler run and insert a soft line break, exactly like the old
// textarea's native Shift+Enter behavior.
function EnterKeyPlugin({ onEnter }: { onEnter: () => void }) {
  const [editor] = useLexicalComposerContext();

  useEffect(() => {
    return editor.registerCommand(
      KEY_ENTER_COMMAND,
      (event) => {
        if (event?.shiftKey) return false;
        event?.preventDefault();
        onEnter();
        return true;
      },
      COMMAND_PRIORITY_HIGH,
    );
  }, [editor, onEnter]);

  return null;
}

// Intercepts paste when the clipboard carries files (a pasted screenshot,
// or files copied from the OS file manager) so they become attachments
// instead of being dumped into the editor as binary/garbage text. Runs at
// COMMAND_PRIORITY_LOW, which is *higher* priority than the plain-text
// paste handler RichTextPlugin registers at COMMAND_PRIORITY_EDITOR (the
// lowest priority -- see node_modules/@lexical/rich-text/src/index.ts), so
// this handler is asked first on every paste. Returning `true` (files
// found; event.preventDefault() already called) stops the command from
// reaching that default handler. Returning `false` (no files -- a normal
// text/markdown paste) lets it fall through to Lexical's default behavior
// unchanged.
function PastePlugin({
  onPasteFiles,
  disabled,
}: {
  onPasteFiles: (files: File[]) => void;
  disabled: boolean;
}) {
  const [editor] = useLexicalComposerContext();
  const disabledRef = useRef(disabled);
  disabledRef.current = disabled;
  const onPasteFilesRef = useRef(onPasteFiles);
  onPasteFilesRef.current = onPasteFiles;

  useEffect(() => {
    return editor.registerCommand(
      PASTE_COMMAND,
      (event) => {
        if (disabledRef.current) return false;
        // PASTE_COMMAND's payload is ClipboardEvent | InputEvent |
        // KeyboardEvent -- only ClipboardEvent carries clipboardData.
        const clipboardData = "clipboardData" in event ? event.clipboardData : null;
        if (!clipboardData) return false;

        let files = Array.from(clipboardData.files ?? []);
        if (files.length === 0 && clipboardData.items) {
          // Some browsers (notably a pasted OS screenshot) surface the
          // image only via `items` with kind "file", not `.files`.
          files = Array.from(clipboardData.items)
            .filter((item) => item.kind === "file")
            .map((item) => item.getAsFile())
            .filter((file): file is File => file !== null);
        }

        if (files.length === 0) return false;
        event.preventDefault();
        onPasteFilesRef.current(files);
        return true;
      },
      COMMAND_PRIORITY_LOW,
    );
  }, [editor]);

  return null;
}

function onError(error: Error) {
  // eslint-disable-next-line no-console
  console.error(error);
}

const MarkdownEditor = forwardRef<MarkdownEditorHandle, Props>(
  function MarkdownEditor(
    { placeholder, disabled, streaming, canSubmitEmpty, blockSubmit, onSend, onEmptyChange, onPasteFiles },
    ref,
  ) {
    const editorRef = useRef<LexicalEditor | null>(null);
    // Refs (not just props) so the stable `attemptSubmit`/`EnterKeyPlugin`
    // command handler always reads the *current* disabled/streaming values
    // instead of whatever they were when the command was registered.
    const disabledRef = useRef(disabled);
    const streamingRef = useRef(streaming);
    const canSubmitEmptyRef = useRef(canSubmitEmpty);
    const blockSubmitRef = useRef(blockSubmit);
    disabledRef.current = disabled;
    streamingRef.current = streaming;
    canSubmitEmptyRef.current = canSubmitEmpty;
    blockSubmitRef.current = blockSubmit;

    const attemptSubmit = useCallback(() => {
      const editor = editorRef.current;
      if (!editor || disabledRef.current || streamingRef.current) return;
      let markdown = "";
      editor.getEditorState().read(() => {
        markdown = $convertToMarkdownString(TRANSFORMERS);
      });
      // Never send while an attachment is still uploading (would drop it).
      if (blockSubmitRef.current) return;
      // Empty text alone blocks a send -- unless a fully-uploaded attachment
      // is present, in which case an attachment-only message is allowed.
      if (!markdown.trim() && !canSubmitEmptyRef.current) return;
      onSend(markdown);
      editor.update(() => {
        $getRoot().clear();
      });
    }, [onSend]);

    useImperativeHandle(
      ref,
      () => ({
        focus: () => {
          editorRef.current?.focus();
        },
        submit: attemptSubmit,
      }),
      [attemptSubmit],
    );

    useEffect(() => {
      editorRef.current?.setEditable(!disabled);
    }, [disabled]);

    const initialConfig = useMemo(
      () => ({
        namespace: "composer",
        nodes: [HeadingNode, QuoteNode, ListNode, ListItemNode, CodeNode, LinkNode],
        onError,
        editable: !disabled,
        theme: {
          paragraph: styles.paragraph,
          heading: {
            h1: `${styles.heading} ${styles.h1}`,
            h2: `${styles.heading} ${styles.h2}`,
            h3: `${styles.heading} ${styles.h3}`,
            h4: `${styles.heading} ${styles.h4}`,
            h5: `${styles.heading} ${styles.h5}`,
            h6: `${styles.heading} ${styles.h6}`,
          },
          quote: styles.quote,
          list: {
            ul: styles.ul,
            ol: styles.ol,
            listitem: styles.listItem,
          },
          link: styles.link,
          text: {
            bold: styles.bold,
            italic: styles.italic,
            strikethrough: styles.strikethrough,
            underline: styles.underline,
            code: styles.inlineCode,
          },
          code: styles.codeBlock,
        },
      }),
      [],
    );

    return (
      <LexicalComposer initialConfig={initialConfig}>
        <div className={styles.editorShell}>
          <RichTextPlugin
            contentEditable={
              <ContentEditable
                className={`${styles.editor} ${disabled ? styles.editorDisabled : ""}`}
                aria-placeholder={placeholder}
                placeholder={<div className={styles.placeholder}>{placeholder}</div>}
              />
            }
            ErrorBoundary={LexicalErrorBoundary}
          />
        </div>
        <HistoryPlugin />
        <ListPlugin />
        <LinkPlugin />
        <MarkdownShortcutPlugin transformers={TRANSFORMERS} />
        <EnterKeyPlugin onEnter={attemptSubmit} />
        <PastePlugin onPasteFiles={onPasteFiles} disabled={disabled} />
        <EditorRefPlugin editorRef={editorRef} />
        <OnChangePlugin
          ignoreSelectionChange
          onChange={(editorState) => {
            editorState.read(() => {
              onEmptyChange($getRoot().getTextContent().trim() === "");
            });
          }}
        />
      </LexicalComposer>
    );
  },
);

export default MarkdownEditor;
