# Lumen

A hand-owned decode loop over 0.5-4B open weights for training loops, up to 8B for inference; post-train → RL → inspect, on one machine.
Qwen3 on Apple Silicon via MLX.

- open weights you can actually open → decode loop, KV cache, every hidden layer → measure what training changed, don't guess
- what RL really does → which layers moved, how far activations drift, was reasoning taught or already there
- rollouts you can edit → drop a step, replay, watch the reward move → credit per step, not one score at the end
- the conversation is a database → every turn typed and rewritable → one object: chat, training example, log
- changing the past isn't free → attention caches cut cleanly, recurrent state doesn't → checkpoint and replay
- tell the model what matters → real weight on a span, not bold text → pointing is free, editing costs a rebuild
- many writers, one history → who wrote it, where it came from, what the model actually read, what got undone → rank context without labels
- internals stream out live → per-token probabilities, attention, activations → not rebuilt after

## Project structure

- `workbench/` — Python backend
  - `engine/` — hand-owned MLX generation loop, KV-cache reuse, control queue, taps
  - `context/` — event-sourced, editable context (typed segments), token assembly, eviction policies
  - `server/` — FastAPI WebSocket app, wire protocol, chat-template framing
  - `config/` — durable task configuration (steering, saved by name)
  - `attachments/` — text/PDF/OCR/VLM extraction pipeline and transient store
  - `tools/` — safe tool registry (calculator, current-time, context search)
  - `cli/` — terminal client: streaming, context inspection, priced edits, per-token logprobs
  - `static/` — minimal fallback client
- `frontend/` — Next.js app (chat UI, context inspector, media viewer)
- `tests/` — pytest suite
- `experiments/` — engine ↔ stock mlx-lm parity harness

https://arxiv.org/pdf/2607.24653
