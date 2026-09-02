import mlx.core as mx
import pytest


@pytest.fixture(autouse=True)
def _disable_vlm_by_default(monkeypatch):
    """Session-wide safety net: attachment/server tests that are
    NOT specifically about the VLM must never accidentally trigger
    mlx-vlm's real (multi-GB, network-downloaded) model load just because
    an image-mime attachment reaches process_attachment() -- e.g. the
    corrupt/garbage-image 400 tests in test_server.py, or OCR-focused tests
    in test_attachments.py that predate VLM support and expect the OCR path.

    Default the WHOLE test suite to WORKBENCH_DISABLE_VLM=1 (VlmImageProcessor
    disabled, images fall through to ImageOcrProcessor -- see
    workbench/attachments/processors.py). Tests that specifically exercise
    VlmImageProcessor override this locally (test_attachments.py's
    `_reset_vlm_singleton` fixture `delenv`s it back off; the real @slow
    round-trip test does the same)."""
    monkeypatch.setenv("WORKBENCH_DISABLE_VLM", "1")


class FakeModel:
    """Deterministic LM: always predicts (last_token + 1) % vocab_size.
    Token 250 acts as EOS. Ignores cache (accepts the kwarg like a real model)."""
    vocab_size = 500

    def __call__(self, inputs, cache=None):
        n = inputs.shape[1]
        last = int(inputs[0, -1].item())
        row = [0.0] * self.vocab_size
        row[(last + 1) % self.vocab_size] = 10.0
        return mx.broadcast_to(mx.array(row), (1, n, self.vocab_size))

    def make_cache(self):
        return []


class FakeLayer:
    """One transformer block's worth of behaviour: a pure function of the
    hidden state, so a captured activation is predictable from the layer index."""

    def __init__(self, index: int):
        self.index = index

    def __call__(self, h, *args, **kwargs):
        # Route one attention call through the module entry point, so a single
        # fake exercises hidden-state capture, the lens, and attention capture.
        n_keys = h.shape[1]
        q = mx.array([[[[1.0, 0.0]]]])
        k = mx.array([[[[float(j) * (self.index + 1), 0.0] for j in range(n_keys)]]])
        v = mx.zeros((1, 1, n_keys, 2))
        scaled_dot_product_attention(q, k, v, scale=1.0)
        return h + (self.index + 1)


class FakeNorm:
    """Final norm stand-in: identity, so lens assertions stay arithmetic."""

    def __call__(self, h):
        return h


class FakeHead:
    """Unembedding stand-in: peaks the logit at int(h[0]), so each layer's
    hidden state maps to a predictable 'prediction' for that depth."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __call__(self, h):
        row = [0.0] * self.vocab_size
        row[int(h[0].item()) % self.vocab_size] = 10.0
        return mx.array(row)


class FakeLayeredModel:
    """Like FakeModel, but actually runs a `self.layers` list the way mlx-lm's
    models do -- so layer-level activation taps have something to hook."""
    vocab_size = 500
    hidden_dim = 8

    def __init__(self, n_layers: int = 3):
        self.layers = [FakeLayer(i) for i in range(n_layers)]
        self.norm = FakeNorm()
        self.lm_head = FakeHead(self.vocab_size)

    def __call__(self, inputs, cache=None):
        n = inputs.shape[1]
        last = int(inputs[0, -1].item())
        h = mx.zeros((1, n, self.hidden_dim)) + float(last)
        for layer in self.layers:
            h = layer(h)
        row = [0.0] * self.vocab_size
        row[(last + 1) % self.vocab_size] = 10.0
        return mx.broadcast_to(mx.array(row), (1, n, self.vocab_size))

    def make_cache(self):
        return []


class FakeDetokenizer:
    """Mirrors mlx_lm's StreamingDetokenizer contract closely enough to pin
    the engine's finalize-and-flush wiring: `last_segment` is a delta since
    it was last read, and a token's text may be withheld until finalize()
    flushes it -- without disturbing the immediate, fully-resolved
    `"<token_id>"` segment every other test already pins."""

    def __init__(self):
        self.text = ""
        self.offset = 0
        self._pending_flush = False

    def reset(self):
        self.text = ""
        self.offset = 0
        self._pending_flush = False

    def add_token(self, token_id):
        self.text += f"<{token_id}>"
        self._pending_flush = True

    def finalize(self):
        if self._pending_flush:
            # Simulates flushing a withheld remainder (e.g. an unresolved
            # multibyte tail) that only becomes available once decoding stops.
            self.text += "~"
            self._pending_flush = False

    @property
    def last_segment(self):
        segment = self.text[self.offset:]
        self.offset = len(self.text)
        return segment


class FakeTokenizer:
    """Encode/decode treat text as space-separated integer token ids (so tests
    can assert on token sequences directly). `apply_chat_template` mimics a
    ChatML-style template (role tags wrap the content, generation prompt is a
    pure additive suffix) using integer sentinel tags, keeping every piece of
    template-produced text encodable by the same whitespace-split `encode`."""
    eos_token_ids = {250}

    # "tool" (401/402) is additive for tool-calling: real Qwen3
    # wraps a tool result as a `user` turn containing <tool_response>...
    # </tool_response> (see framing.frame_tool_result), but framing derives
    # its prefix/suffix purely by rendering role="tool" through
    # apply_chat_template and diffing against the placeholder -- it never
    # inspects _ROLE_KIND/_ROLE_PROVENANCE (those are frame_message-only), so
    # this fake just needs *a* distinct tag pair for role="tool" to exercise
    # the same placeholder-diff mechanism the other roles already use.
    _ROLE_TAGS = {"user": (101, 102), "assistant": (201, 202), "system": (301, 302),
                 "tool": (401, 402)}
    _GENERATION_PROMPT_TOKEN = 900

    def __init__(self):
        self._detok = FakeDetokenizer()

    @property
    def detokenizer(self):
        return self._detok

    # Non-integer whitespace tokens (e.g. English words in an attachment
    # preface) map to a fixed sentinel id rather than raising -- additive:
    # every existing integer-token test is unaffected since integer tokens
    # still keep their exact value.
    _NON_INT_SENTINEL = 999

    def encode(self, text, add_special_tokens=True):
        ids = []
        for t in text.split():
            try:
                ids.append(int(t))
            except ValueError:
                ids.append(self._NON_INT_SENTINEL)
        return ids

    def decode(self, ids):
        return " ".join(str(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        parts = []
        for m in messages:
            start, end = self._ROLE_TAGS[m["role"]]
            parts.append(f"{start} {m['content']} {end}")
        text = " ".join(parts)
        if add_generation_prompt:
            text += f" {self._GENERATION_PROMPT_TOKEN}"
        return self.encode(text) if tokenize else text


@pytest.fixture
def fake_model():
    return FakeModel()


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


@pytest.fixture
def fake_layered_model():
    return FakeLayeredModel()


def scaled_dot_product_attention(queries, keys, values, scale=1.0, mask=None, **kw):
    """Stand-in for mlx-lm's fused entry point, called through the MODULE
    namespace exactly as the real model modules call theirs -- which is the
    seam attention capture intercepts. Expands grouped-query KV heads the way
    the real kernel does internally."""
    q_heads, kv_heads = queries.shape[1], keys.shape[1]
    if kv_heads and q_heads != kv_heads and q_heads % kv_heads == 0:
        repeats = q_heads // kv_heads
        keys = mx.repeat(keys, repeats, axis=1)
        values = mx.repeat(values, repeats, axis=1)
    scores = (queries * scale) @ keys.swapaxes(-1, -2)
    return mx.softmax(scores, axis=-1) @ values


class FakeAttnLayer:
    """A block whose attention routes through the module-level entry point."""

    def __init__(self, index: int, n_keys: int):
        self.index = index
        self.n_keys = n_keys

    def __call__(self, h, *args, **kwargs):
        # queries: one position, 2 dims. keys: n_keys positions whose first
        # component grows with position, scaled per layer so each layer
        # produces a visibly different distribution.
        q = mx.array([[[[1.0, 0.0]]]])
        rows = [[float(j) * (self.index + 1), 0.0] for j in range(self.n_keys)]
        k = mx.array([[rows]])
        v = mx.zeros((1, 1, self.n_keys, 2))
        scaled_dot_product_attention(q, k, v, scale=1.0)
        return h


class FakeAttnModel:
    """Layers that each perform one attention call, so capture can be indexed."""
    vocab_size = 500
    n_keys = 4

    def __init__(self, n_layers: int = 3):
        self.layers = [FakeAttnLayer(i, self.n_keys) for i in range(n_layers)]

    def __call__(self, inputs, cache=None):
        n = inputs.shape[1]
        h = mx.zeros((1, n, 2))
        for layer in self.layers:
            h = layer(h)
        last = int(inputs[0, -1].item())
        row = [0.0] * self.vocab_size
        row[(last + 1) % self.vocab_size] = 10.0
        return mx.broadcast_to(mx.array(row), (1, n, self.vocab_size))

    def make_cache(self):
        return []


@pytest.fixture
def fake_attn_model():
    return FakeAttnModel()


class FakeGQALayer:
    """Grouped-query attention: more query heads than key/value heads, which is
    what every modern Qwen/Llama checkpoint actually does."""

    def __init__(self, index: int, n_keys: int, n_q_heads: int = 4, n_kv_heads: int = 2):
        self.index, self.n_keys = index, n_keys
        self.n_q_heads, self.n_kv_heads = n_q_heads, n_kv_heads

    def __call__(self, h, *args, **kwargs):
        q = mx.ones((1, self.n_q_heads, 1, 2))
        k = mx.ones((1, self.n_kv_heads, self.n_keys, 2))
        v = mx.zeros((1, self.n_kv_heads, self.n_keys, 2))
        scaled_dot_product_attention(q, k, v, scale=1.0)
        return h


class FakeGQAModel:
    vocab_size = 500
    n_keys = 4

    def __init__(self, n_layers: int = 2):
        self.layers = [FakeGQALayer(i, self.n_keys) for i in range(n_layers)]

    def __call__(self, inputs, cache=None):
        n = inputs.shape[1]
        h = mx.zeros((1, n, 2))
        for layer in self.layers:
            h = layer(h)
        row = [0.0] * self.vocab_size
        row[1] = 10.0
        return mx.broadcast_to(mx.array(row), (1, n, self.vocab_size))

    def make_cache(self):
        return []


@pytest.fixture
def fake_gqa_model():
    return FakeGQAModel()
