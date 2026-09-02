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


def _readout(model):
    """The model's (final norm, unembedding) pair, as callables.

    mlx-lm keeps the norm on the inner model and the head either as `lm_head`
    or, when embeddings are tied, as `embed_tokens.as_linear`."""
    inner = getattr(model, "model", model)
    norm = getattr(inner, "norm", None) or (lambda h: h)
    head = getattr(model, "lm_head", None)
    if head is None:
        embed = getattr(inner, "embed_tokens", None)
        head = getattr(embed, "as_linear", None) if embed is not None else None
    if head is None:
        raise AttributeError("model exposes no unembedding to read out")
    return norm, head


def logit_lens(model, hidden: dict, k: int = 5) -> dict[int, dict[int, float]]:
    """Decode captured hidden states through the model's own readout.

    Answers "what would the model predict if it stopped at this depth" -- the
    layer at which an answer is already decided is visible as the depth where
    the top token stops changing."""
    norm, head = _readout(model)
    out: dict[int, dict[int, float]] = {}
    for layer, h in hidden.items():
        logits = head(norm(h))
        out[layer] = top_k_logprobs(logits[None], k)
    return out


def attention_mass_by_segment(weights, spans) -> list[float]:
    """Fold a per-position attention row into per-segment totals.

    `weights` is one query position's attention over all key positions;
    `spans` are [start, end) token ranges, normally taken straight from the
    ContextObject's segments. Positions covered by no span are dropped, so a
    partial span list (just the document chunks, say) is meaningful on its own."""
    return [float(weights[start:end].sum().item()) for start, end in spans]


@contextmanager
def capture_attention(model, layer_indices):
    """Record each requested block's attention over key positions, for one pass.

    mlx-lm routes every block through a FUSED `scaled_dot_product_attention`
    that never materialises the weight matrix, so the weights are recomputed
    from the queries and keys the fused call receives -- post-RoPE, post-norm,
    with no duplication of upstream projection logic. Blocks run in order, so
    the call index is the layer index.

    Only the LAST query position is kept, which is the token being produced.
    The causal mask can be ignored for that row specifically: the final query
    attends to every key present, so masking would be a no-op there.

    A quantized KV cache hands the fused kernel a non-array key representation;
    those layers are skipped rather than guessed at.

    Costs an extra QK^T for the requested layers, hence opt-in."""
    import sys

    module = sys.modules[type(model).__module__]
    original = module.scaled_dot_product_attention
    captured: dict[int, mx.array] = {}
    calls = {"n": 0}

    def recording(queries, keys, values, *args, **kwargs):
        index = calls["n"]
        calls["n"] += 1
        if index in layer_indices and isinstance(keys, mx.array):
            scale = kwargs.get("scale", 1.0)
            # Grouped-query attention: several query heads share one KV head.
            # The fused kernel broadcasts internally; recomputing by hand has
            # to expand the KV heads to match, or the matmul cannot broadcast.
            k = keys
            q_heads, kv_heads = queries.shape[1], keys.shape[1]
            if kv_heads and q_heads != kv_heads and q_heads % kv_heads == 0:
                k = mx.repeat(keys, q_heads // kv_heads, axis=1)
            scores = (queries * scale) @ k.swapaxes(-1, -2)
            weights = mx.softmax(scores[0, :, -1, :].astype(mx.float32), axis=-1)
            captured[index] = weights.mean(axis=0)   # average over heads
        return original(queries, keys, values, *args, **kwargs)

    module.scaled_dot_product_attention = recording
    try:
        yield captured
    finally:
        module.scaled_dot_product_attention = original
