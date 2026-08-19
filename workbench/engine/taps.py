"""Introspection taps. Each tap is a pure function over per-step engine state."""
from __future__ import annotations

import mlx.core as mx


def top_k_logprobs(logits: mx.array, k: int) -> dict[int, float]:
    """Top-k token log-probabilities from raw last-position logits [1, V]."""
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    flat = logprobs[0]
    top_ids = mx.argpartition(-flat, kth=k - 1)[:k]
    pairs = sorted(
        ((int(i.item()), float(flat[int(i.item())].item())) for i in top_ids),
        key=lambda p: -p[1],
    )
    return dict(pairs)
