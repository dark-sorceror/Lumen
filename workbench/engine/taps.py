"""Introspection taps. Each tap is a pure function over per-step engine state."""
from __future__ import annotations

from contextlib import contextmanager

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


def _layer_list(model):
    """The model's transformer blocks, as a mutable list.

    mlx-lm models iterate a plain `self.layers` list in their forward pass, so
    replacing entries in it is enough to observe every block -- no reimplementation
    of the forward pass, and nothing to keep in sync with upstream."""
    if hasattr(model, "layers"):
        return model.layers
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner.layers
    raise AttributeError("model exposes no layer list to tap")


class _RecordingLayer:
    """Forwards to the real block, keeping its output's last position."""

    def __init__(self, inner, index: int, sink: dict):
        self._inner, self._index, self._sink = inner, index, sink

    def __call__(self, *args, **kwargs):
        out = self._inner(*args, **kwargs)
        h = out[0] if isinstance(out, tuple) else out
        # Last position only: during decode that is the token being produced,
        # and during prefill it is the one whose logits seed the loop.
        self._sink[self._index] = h[0, -1]
        return out


@contextmanager
def capture_layer_outputs(model, layer_indices):
    """Record the hidden state leaving each requested block, for one forward pass.

    Yields a dict that is populated as the pass runs. The layer list is always
    restored, so a raised exception cannot leave the model wrapped."""
    layers = _layer_list(model)
    captured: dict[int, mx.array] = {}
    originals = {i: layers[i] for i in layer_indices}
    for i, original in originals.items():
        layers[i] = _RecordingLayer(original, i, captured)
    try:
        yield captured
    finally:
        for i, original in originals.items():
            layers[i] = original
