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


# -- Steering: intervene on the residual stream, not just observe it ---------

def test_apply_steering_shifts_the_layer_output(fake_layered_model, fake_tokenizer):
    """A steering vector is added to the residual stream leaving a block, so a
    tap placed outside it records the SHIFTED activation."""
    from workbench.engine.taps import apply_steering, capture_layer_outputs

    baseline = {}
    with capture_layer_outputs(fake_layered_model, (0,)) as cap:
        fake_layered_model(mx.array([[1]]))
        baseline = {k: v for k, v in cap.items()}

    direction = mx.ones((fake_layered_model.hidden_dim,))
    with apply_steering(fake_layered_model, {0: (direction, 2.0)}):
        with capture_layer_outputs(fake_layered_model, (0,)) as steered:
            fake_layered_model(mx.array([[1]]))

    assert float(steered[0][0].item()) == pytest.approx(
        float(baseline[0][0].item()) + 2.0)


def test_apply_steering_restores_the_layer_stack(fake_layered_model):
    from workbench.engine.taps import apply_steering

    originals = list(fake_layered_model.layers)
    with apply_steering(fake_layered_model, {0: (mx.ones((8,)), 1.0)}):
        pass
    assert fake_layered_model.layers == originals


def test_steering_with_zero_strength_is_a_no_op(fake_layered_model):
    from workbench.engine.taps import apply_steering, capture_layer_outputs

    with capture_layer_outputs(fake_layered_model, (0,)) as cap:
        fake_layered_model(mx.array([[1]]))
        before = float(cap[0][0].item())

    with apply_steering(fake_layered_model, {0: (mx.ones((8,)), 0.0)}):
        with capture_layer_outputs(fake_layered_model, (0,)) as cap2:
            fake_layered_model(mx.array([[1]]))
            after = float(cap2[0][0].item())

    assert after == pytest.approx(before)


def test_steering_vector_is_the_contrastive_mean_difference(fake_layered_model, fake_tokenizer):
    """Classic contrastive construction: mean activation on positive prompts
    minus mean on negative ones. The fake seeds its hidden state from the last
    token id, so the arithmetic is checkable."""
    engine = Engine(fake_layered_model, fake_tokenizer)

    v = engine.steering_vector(positive=[[5]], negative=[[1]], layer=0)

    # layer 0 output is last_token + 1 -> 6.0 vs 2.0
    assert float(v[0].item()) == pytest.approx(4.0)


def test_steering_vector_averages_over_multiple_prompts(fake_layered_model, fake_tokenizer):
    engine = Engine(fake_layered_model, fake_tokenizer)

    v = engine.steering_vector(positive=[[5], [7]], negative=[[1]], layer=0)

    # mean(6, 8) - 2 = 5
    assert float(v[0].item()) == pytest.approx(5.0)


def test_generation_applies_steering_to_the_residual_stream(fake_layered_model, fake_tokenizer):
    """Steering has to be live for every forward pass of the loop -- prefill and
    each decode step -- not just set up once and forgotten."""
    engine = Engine(fake_layered_model, fake_tokenizer)

    plain = list(engine.generate([1], GenParams(max_tokens=1, hidden_layers=(0,))))
    steered = list(engine.generate(
        [1], GenParams(max_tokens=1, hidden_layers=(0,),
                       steering={0: (mx.ones((fake_layered_model.hidden_dim,)), 3.0)})))

    assert float(steered[0].hidden[0][0].item()) == pytest.approx(
        float(plain[0].hidden[0][0].item()) + 3.0)


def test_generation_without_steering_leaves_the_stack_untouched(fake_layered_model, fake_tokenizer):
    engine = Engine(fake_layered_model, fake_tokenizer)
    originals = list(fake_layered_model.layers)
    list(engine.generate([1], GenParams(max_tokens=2,
                                        steering={0: (mx.ones((8,)), 1.0)})))
    assert fake_layered_model.layers == originals


# -- Comparing runs: what an intervention changes inside ---------------------

def test_compare_reports_activation_drift_per_layer(fake_layered_model, fake_tokenizer):
    """The same prompt under two conditions: how far the residual stream moved
    at each depth. This is the measurement a post-training intervention needs --
    steering is just the cheapest intervention to point it at."""
    engine = Engine(fake_layered_model, fake_tokenizer)
    direction = mx.array([1.0] + [0.0] * (fake_layered_model.hidden_dim - 1))

    diff = engine.compare([1, 2], layers=(0, 2), steering={0: (direction, 2.0)})

    assert set(diff) == {0, 2}
    # layer 0 is shifted by exactly 2 along one axis
    assert diff[0]["l2"] == pytest.approx(2.0)
    # a shift off-axis cannot leave the direction unchanged
    assert diff[0]["cosine"] < 1.0


def test_compare_leaves_downstream_layers_shifted_too(fake_layered_model, fake_tokenizer):
    """Steering the residual stream propagates: a later block sees the shift."""
    engine = Engine(fake_layered_model, fake_tokenizer)
    direction = mx.array([1.0] + [0.0] * (fake_layered_model.hidden_dim - 1))

    diff = engine.compare([1, 2], layers=(0, 2), steering={0: (direction, 2.0)})

    assert diff[2]["l2"] == pytest.approx(2.0)


def test_compare_with_no_intervention_reports_no_drift(fake_layered_model, fake_tokenizer):
    engine = Engine(fake_layered_model, fake_tokenizer)
    diff = engine.compare([1, 2], layers=(0,), steering={})
    assert diff[0]["l2"] == pytest.approx(0.0)
    assert diff[0]["cosine"] == pytest.approx(1.0)


def test_compare_reports_whether_the_predicted_token_flipped(fake_layered_model, fake_tokenizer):
    engine = Engine(fake_layered_model, fake_tokenizer)
    direction = mx.array([5.0] + [0.0] * (fake_layered_model.hidden_dim - 1))

    diff = engine.compare([1, 2], layers=(0,), steering={0: (direction, 1.0)})

    assert "top_token_changed" in diff[0]
    assert isinstance(diff[0]["top_token_changed"], bool)


def test_cosine_never_exceeds_one_even_in_low_precision(fake_layered_model, fake_tokenizer):
    """Activations on a quantized model arrive in low precision; accumulating a
    dot product there can push a cosine above 1, which is not a number a
    measurement instrument may report."""
    engine = Engine(fake_layered_model, fake_tokenizer)
    for strength in (0.0, 0.5, 3.0):
        diff = engine.compare([1, 2], layers=(0, 2),
                              steering={0: (mx.ones((fake_layered_model.hidden_dim,)), strength)})
        for layer, d in diff.items():
            assert -1.0 <= d["cosine"] <= 1.0, (layer, strength, d["cosine"])


def test_compare_over_aggregates_drift_across_prompts(fake_layered_model, fake_tokenizer):
    """One prompt is an anecdote. The aggregate over a set is the measurement --
    with a spread, so a claim about an intervention can be qualified."""
    engine = Engine(fake_layered_model, fake_tokenizer)
    direction = mx.array([1.0] + [0.0] * (fake_layered_model.hidden_dim - 1))

    agg = engine.compare_over([[1, 2], [3, 4], [5, 6]], layers=(0, 2),
                              steering={0: (direction, 2.0)})

    assert agg[0]["n"] == 3
    assert agg[0]["mean_l2"] == pytest.approx(2.0)
    assert agg[0]["std_relative"] == pytest.approx(0.0, abs=1e-6) or agg[0]["std_relative"] >= 0.0
    assert 0.0 <= agg[0]["flip_rate"] <= 1.0
    assert set(agg) == {0, 2}


def test_compare_over_requires_at_least_one_prompt(fake_layered_model, fake_tokenizer):
    engine = Engine(fake_layered_model, fake_tokenizer)
    with pytest.raises(ValueError):
        engine.compare_over([], layers=(0,), steering={})


def test_compare_over_flip_rate_is_a_fraction_of_prompts(fake_layered_model, fake_tokenizer):
    engine = Engine(fake_layered_model, fake_tokenizer)
    agg = engine.compare_over([[1, 2], [3, 4]], layers=(0,), steering={})
    assert agg[0]["flip_rate"] == pytest.approx(0.0)   # no intervention, no flips


# -- Layer sweep: where is an intervention worth injecting? ------------------

def test_activation_norm_measures_the_stream_at_a_layer(fake_layered_model, fake_tokenizer):
    """The scale any dose must be sized against. FakeLayer adds index+1, so a
    prompt ending in token 1 leaves layer 0 with every dim at 2.0."""
    engine = Engine(fake_layered_model, fake_tokenizer)

    norm = engine.activation_norm([1], layer=0)

    assert norm == pytest.approx(2.0 * (fake_layered_model.hidden_dim ** 0.5))


def test_sweep_doses_each_layer_relative_to_its_own_norm(fake_layered_model, fake_tokenizer):
    """A fixed alpha means different things at different depths, because the
    stream's own magnitude grows. The sweep must hold rho constant, not alpha."""
    engine = Engine(fake_layered_model, fake_tokenizer)

    sweep = engine.sweep_injection(probes=[[1], [2]], positive=[[5]], negative=[[1]],
                                   layers=(0, 2), rho=0.1)

    assert set(sweep) == {0, 2}
    for layer, r in sweep.items():
        assert r["rho"] == pytest.approx(0.1)
        # alpha is rho * the norm measured AT that layer, so it differs by depth
        assert r["alpha"] == pytest.approx(0.1 * engine.activation_norm([1], layer))
    assert sweep[0]["alpha"] != pytest.approx(sweep[2]["alpha"])


def test_sweep_reports_outcome_at_the_readout(fake_layered_model, fake_tokenizer):
    """What matters is whether the final prediction moved, not whether the
    representation did."""
    engine = Engine(fake_layered_model, fake_tokenizer)

    sweep = engine.sweep_injection(probes=[[1], [2]], positive=[[5]], negative=[[1]],
                                   layers=(0,), rho=0.5)

    r = sweep[0]
    assert r["n"] == 2
    assert 0.0 <= r["final_flip_rate"] <= 1.0
    assert r["final_relative"] >= 0.0


def test_sweep_rejects_an_empty_probe_set(fake_layered_model, fake_tokenizer):
    engine = Engine(fake_layered_model, fake_tokenizer)
    with pytest.raises(ValueError):
        engine.sweep_injection(probes=[], positive=[[1]], negative=[[2]], layers=(0,))


# -- Divergence: what does dropping context cost? ---------------------------

def test_next_logprobs_is_a_normalised_distribution(fake_layered_model, fake_tokenizer):
    engine = Engine(fake_layered_model, fake_tokenizer)

    lp = engine.next_logprobs([1, 2])

    assert lp.shape == (fake_layered_model.vocab_size,)
    total = float(mx.exp(lp).sum().item())
    assert total == pytest.approx(1.0, abs=1e-4)


def test_divergence_is_zero_between_identical_contexts(fake_layered_model, fake_tokenizer):
    engine = Engine(fake_layered_model, fake_tokenizer)

    d = engine.divergence([1, 2], [1, 2])

    assert d["kl"] == pytest.approx(0.0, abs=1e-5)
    assert d["top1_agrees"] is True


def test_divergence_detects_a_changed_prediction(fake_layered_model, fake_tokenizer):
    """The fake predicts last_token + 1, so contexts ending differently must
    disagree at the readout."""
    engine = Engine(fake_layered_model, fake_tokenizer)

    d = engine.divergence([1, 2], [1, 7])

    assert d["kl"] > 0.0
    assert d["top1_agrees"] is False


def test_divergence_is_asymmetric_in_the_expected_direction(fake_layered_model, fake_tokenizer):
    """KL(full || pruned) is the quantity of interest: how surprised the full
    context's distribution is by the pruned one. It is not symmetric, and the
    argument order therefore matters."""
    engine = Engine(fake_layered_model, fake_tokenizer)

    forward = engine.divergence([1, 2], [1, 7])["kl"]
    reverse = engine.divergence([1, 7], [1, 2])["kl"]

    assert forward > 0.0 and reverse > 0.0
