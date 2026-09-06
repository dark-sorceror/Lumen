"""Eviction policies: choosing what to drop from a context under a budget.

These are pure functions over spans and measured mass -- no model, no cache.
The point of separating them is that a policy can be argued about and unit
tested on its own, and only then measured against a real model by asking what
the drop cost at the readout (see Engine.divergence)."""
import pytest

from workbench.context.eviction import (
    keep_by_recency,
    keep_by_attention,
    rebuild_tokens,
)


SPANS = {
    "sys": (0, 10),
    "a": (10, 30),
    "b": (30, 40),
    "c": (40, 60),
}
# 'a' is long and ignored; 'b' is short and heavily attended.
MASS = {"sys": 0.50, "a": 0.05, "b": 0.35, "c": 0.10}


def test_recency_keeps_the_newest_segments_that_fit():
    kept = keep_by_recency(SPANS, budget=30)

    assert kept == {"b", "c"}  # 10 + 20 = 30, 'a' would overflow


def test_attention_keeps_the_most_attended_segments_that_fit():
    kept = keep_by_attention(SPANS, MASS, budget=30)

    # 'sys' (0.50, 10 tok) then 'b' (0.35, 10 tok) -> 20 used; 'c' (0.10) is
    # 20 tokens and would overflow, so it is skipped, not stopped at.
    assert kept == {"sys", "b"}


def test_the_two_policies_disagree_on_this_context():
    """The whole experiment is only interesting if the policies differ."""
    assert keep_by_recency(SPANS, budget=30) != keep_by_attention(SPANS, MASS, budget=30)


def test_a_budget_covering_everything_keeps_everything():
    assert keep_by_recency(SPANS, budget=60) == set(SPANS)
    assert keep_by_attention(SPANS, MASS, budget=60) == set(SPANS)


def test_a_zero_budget_keeps_nothing():
    assert keep_by_recency(SPANS, budget=0) == set()
    assert keep_by_attention(SPANS, MASS, budget=0) == set()


def test_rebuild_preserves_original_token_order_not_policy_order():
    """Attention ranks 'b' above 'sys' in *value*, but the rebuilt context must
    still read in positional order or it is not the same language."""
    tokens = list(range(60))

    out = rebuild_tokens(tokens, SPANS, {"b", "sys"})

    assert out == list(range(0, 10)) + list(range(30, 40))


def test_rebuild_with_everything_kept_is_the_identity():
    tokens = list(range(60))

    assert rebuild_tokens(tokens, SPANS, set(SPANS)) == tokens


def test_rebuild_ignores_unknown_segment_ids():
    tokens = list(range(60))

    assert rebuild_tokens(tokens, SPANS, {"sys", "nonexistent"}) == list(range(10))


# -- The engine-side harness that scores policies against a real model ------

import mlx.core as mx  # noqa: E402
from workbench.engine.engine import Engine  # noqa: E402


def test_attention_mass_returns_one_entry_per_segment(fake_layered_model, fake_tokenizer):
    engine = Engine(fake_layered_model, fake_tokenizer)
    spans = {"a": (0, 2), "b": (2, 4)}

    mass = engine.attention_mass([1, 2, 3, 4], spans, layers=[0, 1])

    assert set(mass) == {"a", "b"}
    assert sum(mass.values()) == pytest.approx(1.0, abs=1e-4)


def test_attention_mass_without_the_sink_reweights_away_from_position_zero(
    fake_layered_model, fake_tokenizer
):
    """Position 0 absorbs a large constant share of attention on real models.
    Excluding it must lower the first segment's share and still renormalise."""
    engine = Engine(fake_layered_model, fake_tokenizer)
    spans = {"a": (0, 2), "b": (2, 4)}

    with_sink = engine.attention_mass([1, 2, 3, 4], spans, layers=[0, 1])
    without = engine.attention_mass([1, 2, 3, 4], spans, layers=[0, 1], exclude_sink=True)

    assert without["a"] < with_sink["a"]
    assert sum(without.values()) == pytest.approx(1.0, abs=1e-4)


def test_evaluate_eviction_scores_every_policy_against_the_full_context(
    fake_layered_model, fake_tokenizer
):
    engine = Engine(fake_layered_model, fake_tokenizer)
    spans = {"a": (0, 2), "b": (2, 4), "q": (4, 6)}

    out = engine.evaluate_eviction([1, 2, 3, 4, 5, 6], spans, budget=4, pinned=("q",))

    assert set(out) == {"recency", "attention", "attention_nosink"}
    for result in out.values():
        assert "q" in result["kept"]          # pinned segments are never dropped
        assert result["n_tokens"] <= 6
        assert result["kl"] >= 0.0
        assert isinstance(result["top1_agrees"], bool)


def test_evaluate_eviction_at_full_budget_is_lossless(fake_layered_model, fake_tokenizer):
    """Keeping everything must cost exactly nothing -- this is the sanity check
    that the metric is measuring eviction and not some artifact of rebuilding."""
    engine = Engine(fake_layered_model, fake_tokenizer)
    spans = {"a": (0, 2), "b": (2, 4), "q": (4, 6)}

    out = engine.evaluate_eviction([1, 2, 3, 4, 5, 6], spans, budget=6)

    for name, result in out.items():
        assert result["n_tokens"] == 6, name
        assert result["kl"] == pytest.approx(0.0, abs=1e-5), name
        assert result["top1_agrees"] is True, name
