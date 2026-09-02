import pytest

import math

import mlx.core as mx

from workbench.engine.engine import Engine, GenParams
from workbench.engine.taps import attention_mass_by_segment, logit_lens, top_k_logprobs


def test_top_k_logprobs_normalized():
    logits = mx.array([[0.0, 1.0, 2.0, 3.0]])
    result = top_k_logprobs(logits, k=2)
    assert list(result.keys()) == [3, 2]                      # sorted desc
    assert math.isclose(sum(math.exp(v) for v in top_k_logprobs(logits, k=4).values()),
                        1.0, rel_tol=1e-5)                    # full k sums to 1
    assert result[3] > result[2]


def test_engine_emits_logprobs_when_requested(fake_model, fake_tokenizer):
    engine = Engine(fake_model, fake_tokenizer)
    events = list(engine.generate([1], GenParams(max_tokens=2, top_k_logprobs=3)))
    for e in events:
        assert len(e.top_logprobs) == 3
        assert e.token_id == max(e.top_logprobs, key=e.top_logprobs.get)


def test_engine_emits_no_logprobs_by_default(fake_model, fake_tokenizer):
    engine = Engine(fake_model, fake_tokenizer)
    events = list(engine.generate([1], GenParams(max_tokens=2)))
    assert all(e.top_logprobs == {} for e in events)


def test_engine_captures_hidden_states_for_requested_layers(fake_layered_model, fake_tokenizer):
    engine = Engine(fake_layered_model, fake_tokenizer)
    events = list(engine.generate([1], GenParams(max_tokens=2, hidden_layers=(0, 2))))

    assert events, "expected at least one token event"
    for e in events:
        assert set(e.hidden) == {0, 2}
        assert e.hidden[0].shape == (fake_layered_model.hidden_dim,)


def test_hidden_capture_is_off_by_default(fake_layered_model, fake_tokenizer):
    """The default path must not wrap layers or carry activations."""
    engine = Engine(fake_layered_model, fake_tokenizer)
    events = list(engine.generate([1], GenParams(max_tokens=2)))
    assert all(e.hidden == {} for e in events)


def test_captured_hidden_is_that_layer_s_output(fake_layered_model, fake_tokenizer):
    """Each FakeLayer adds (index + 1) to a hidden state seeded with the last
    token id, so the captured value pins WHICH block's output was recorded --
    not merely that something was recorded."""
    engine = Engine(fake_layered_model, fake_tokenizer)
    first = next(iter(engine.generate([1], GenParams(max_tokens=1, hidden_layers=(0, 2)))))
    # seed 1.0 -> +1 (layer 0) = 2.0 -> +2 = 4.0 -> +3 (layer 2) = 7.0
    assert float(first.hidden[0][0].item()) == 2.0
    assert float(first.hidden[2][0].item()) == 7.0


def test_layer_list_is_restored_after_capture(fake_layered_model, fake_tokenizer):
    """A leaked wrapper would silently tap every later generation."""
    originals = list(fake_layered_model.layers)
    engine = Engine(fake_layered_model, fake_tokenizer)
    list(engine.generate([1], GenParams(max_tokens=2, hidden_layers=(0, 1, 2))))
    assert fake_layered_model.layers == originals


def test_logit_lens_reads_each_layer_through_the_unembedding(fake_layered_model, fake_tokenizer):
    """Different depths decode to different next tokens -- the whole point of a
    logit lens. Layer 0's hidden state is 2.0 and layer 2's is 7.0, and the fake
    head peaks at int(h[0])."""
    engine = Engine(fake_layered_model, fake_tokenizer)
    first = next(iter(engine.generate([1], GenParams(max_tokens=1, hidden_layers=(0, 2)))))

    lens = logit_lens(fake_layered_model, first.hidden, k=1)

    assert list(lens[0]) == [2]
    assert list(lens[2]) == [7]


def test_attention_mass_by_segment_sums_weights_within_each_span():
    """Per-token attention is not the useful unit -- per-SEGMENT is, because
    segments are what the context object lets you edit."""
    weights = mx.array([0.1, 0.2, 0.3, 0.4])
    spans = [(0, 2), (2, 4)]

    mass = attention_mass_by_segment(weights, spans)

    assert mass[0] == pytest.approx(0.3)
    assert mass[1] == pytest.approx(0.7)


def test_attention_mass_ignores_positions_outside_any_span():
    """A span list may cover only part of the context (framing tokens, say)."""
    weights = mx.array([0.5, 0.25, 0.25])
    mass = attention_mass_by_segment(weights, [(1, 3)])
    assert mass[0] == pytest.approx(0.5)


def test_capture_attention_records_a_distribution_for_requested_layers(fake_attn_model):
    from workbench.engine.taps import capture_attention

    with capture_attention(fake_attn_model, (0, 2)) as captured:
        fake_attn_model(mx.array([[1]]))

    assert set(captured) == {0, 2}
    for layer, row in captured.items():
        assert row.shape == (fake_attn_model.n_keys,)
        assert float(row.sum().item()) == pytest.approx(1.0)
    # different layers weight the same keys differently
    assert float(captured[0][-1].item()) != pytest.approx(float(captured[2][-1].item()))


def test_capture_attention_restores_the_entry_point(fake_attn_model):
    import sys
    from workbench.engine.taps import capture_attention

    module = sys.modules[type(fake_attn_model).__module__]
    original = module.scaled_dot_product_attention
    with capture_attention(fake_attn_model, (0,)):
        pass
    assert module.scaled_dot_product_attention is original


def test_capture_attention_handles_grouped_query_attention(fake_gqa_model):
    """Query heads outnumber KV heads; the fused kernel broadcasts internally,
    so the recomputation has to expand the KV heads to match."""
    from workbench.engine.taps import capture_attention

    with capture_attention(fake_gqa_model, (0,)) as captured:
        fake_gqa_model(mx.array([[1]]))

    assert captured[0].shape == (fake_gqa_model.n_keys,)
    assert float(captured[0].sum().item()) == pytest.approx(1.0)


def test_engine_inspect_returns_lens_and_attention_per_layer(fake_layered_model, fake_tokenizer):
    """One forward pass answers both 'what would this depth predict' and
    'where did this depth look' -- without disturbing the session cache."""
    engine = Engine(fake_layered_model, fake_tokenizer)

    out = engine.inspect([1, 2, 3], layers=(0, 2), top_k=1)

    assert set(out) == {0, 2}
    # last token is 3 -> layer 0 hidden 4.0 -> fake head peaks at token 4
    assert list(out[0]["lens"]) == [4]
    assert out[0]["attention"].shape == (3,)
    assert float(out[0]["attention"].sum().item()) == pytest.approx(1.0)


def test_engine_inspect_leaves_the_session_cache_untouched(fake_layered_model, fake_tokenizer):
    engine = Engine(fake_layered_model, fake_tokenizer)
    engine.start_session()
    list(engine.generate_with_cache([1, 2], GenParams(max_tokens=1)))
    before = list(engine._cached_tokens)

    engine.inspect([9, 9, 9], layers=(0,), top_k=1)

    assert engine._cached_tokens == before
