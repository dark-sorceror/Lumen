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
