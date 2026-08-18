# Lumen

A hand-owned decode loop over 0.5-4B open weights for training loops, up to 8B for inference; post-train → RL → inspect, on one machine.
Qwen3 on Apple Silicon via MLX.

- open weights you can actually open → decode loop, KV cache, every hidden layer → measure what training changed, don't guess
- what RL really does → which layers moved, how far activations drift, was reasoning taught or already there
- rollouts you can edit → drop a step, replay, watch the reward move → credit per step, not one score at the end
- the conversation is a database → every turn typed and rewritable → one object: chat, training example, log
- changing the past isn't free → attention caches cut cleanly, recurrent state doesn't → checkpoint and replay
- internals stream out live → per-token probabilities, attention, activations → not rebuilt after

https://arxiv.org/pdf/2607.24653
