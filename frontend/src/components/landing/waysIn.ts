// The ways in: where the chat can run, and where what you send goes.
// The pills are the same row a new chat opens with in the app.

export type Way = {
  id: string;
  name: string;
  icon: "terminal" | "laptop" | "key";
  /** "ok" for a ready dot, "wait" for a not-loaded-yet dot, "" for none. */
  dot: "ok" | "wait" | "";
  text: string;
  leave: string;
};

export const WAYS: Way[] = [
  { id: "cc", name: "Claude Code", icon: "terminal", dot: "ok", text: "Runs Claude Code in a folder you pick, on your own login.", leave: "What you send goes to Anthropic, as it does in the terminal." },
  { id: "cx", name: "Codex", icon: "terminal", dot: "ok", text: "Runs Codex in a folder you pick, on your own login.", leave: "What you send goes to OpenAI, as it does in the terminal." },
  { id: "qw", name: "Qwen3-8B", icon: "laptop", dot: "wait", text: "Runs on this Mac. Your first message loads the model, about 8 GB.", leave: "What you send stays on your Mac." },
  { id: "an", name: "Anthropic", icon: "key", dot: "", text: "Sent to Claude on your own API key. illumen builds each request, so you can edit what it sees.", leave: "What you send goes to the model you chose." },
  { id: "oa", name: "OpenAI", icon: "key", dot: "", text: "Sent to GPT on your own API key. illumen builds each request, so you can edit what it sees.", leave: "What you send goes to the model you chose." },
];
