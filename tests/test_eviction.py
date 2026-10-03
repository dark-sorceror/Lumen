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


def test_pinned_segments_are_charged_against_the_budget_not_added_on_top(
    fake_layered_model, fake_tokenizer
):
    """A pinned segment must be RESERVED before the policy fills, not unioned in
    afterwards. Unioning lets the result exceed the budget, which makes every
    policy comparison at that budget meaningless -- the policies would be
    holding different numbers of tokens for a reason unrelated to their
    ranking."""
    engine = Engine(fake_layered_model, fake_tokenizer)
    spans = {"a": (0, 2), "b": (2, 4), "q": (4, 6)}

    # Budget 2 fits exactly one segment. Recency alone would take "b" (newest);
    # pinning "a" must therefore cost "b" its place, not buy a second slot.
    out = engine.evaluate_eviction([1, 2, 3, 4, 5, 6], spans, budget=2, pinned=("a",))

    for name, result in out.items():
        assert "a" in result["kept"], name
        assert result["n_tokens"] <= 2, f"{name} exceeded budget: {result['kept']}"


def test_pinned_segments_do_not_compete_in_the_ranking(fake_layered_model, fake_tokenizer):
    """With the obvious segments handed to every policy for free, the policies
    must differ only on what is left -- otherwise a comparison 'holding the hand
    rule fixed' is not actually holding it fixed."""
    engine = Engine(fake_layered_model, fake_tokenizer)
    spans = {"a": (0, 2), "b": (2, 4), "c": (4, 6), "q": (6, 8)}

    out = engine.evaluate_eviction([1, 2, 3, 4, 5, 6, 7, 8], spans, budget=6,
                                   pinned=("a", "q"))

    for name, result in out.items():
        assert {"a", "q"} <= result["kept"], name
        assert result["n_tokens"] <= 6, name
        # 4 tokens reserved, so exactly one of b/c can still fit.
        assert len(result["kept"] & {"b", "c"}) == 1, name


# -- Two-coordinate assessment: KL alone cannot score a repair --------------

def test_gold_logprob_reports_the_answer_token_and_its_rank(
    fake_layered_model, fake_tokenizer
):
    """The fake predicts last_token + 1, so 3 is the argmax for [1, 2]."""
    engine = Engine(fake_layered_model, fake_tokenizer)

    top = engine.gold_score([1, 2], gold=3)
    other = engine.gold_score([1, 2], gold=9)

    assert top["rank"] == 1
    assert other["rank"] > 1
    assert top["logprob"] > other["logprob"]


def test_an_edit_that_changes_nothing_is_preserved(fake_layered_model, fake_tokenizer):
    engine = Engine(fake_layered_model, fake_tokenizer)

    out = engine.assess_edit([1, 2], [1, 2], gold=3)

    assert out["verdict"] == "preserved"
    assert out["kl"] == pytest.approx(0.0, abs=1e-5)
    assert out["gold_delta"] == pytest.approx(0.0, abs=1e-5)


def test_an_edit_that_moves_the_answer_away_from_gold_is_damaged(
    fake_layered_model, fake_tokenizer
):
    engine = Engine(fake_layered_model, fake_tokenizer)

    # full [1,2] predicts 3 (== gold); pruning to [1,7] predicts 8.
    out = engine.assess_edit([1, 2], [1, 7], gold=3)

    assert out["verdict"] == "damaged"
    assert out["gold_delta"] < 0
    assert out["kl"] > 0


def test_an_edit_that_moves_the_answer_toward_gold_is_repaired(
    fake_layered_model, fake_tokenizer
):
    """The case that broke KL-against-full: the full context is WRONG, the edit
    fixes it, and KL -- which measures distance from the full context -- is
    therefore large precisely because the edit is good."""
    engine = Engine(fake_layered_model, fake_tokenizer)

    # full [1,7] predicts 8 (gold is 3, so the full context is wrong);
    # pruning to [1,2] predicts 3 == gold.
    out = engine.assess_edit([1, 7], [1, 2], gold=3)

    assert out["verdict"] == "repaired"
    assert out["gold_delta"] > 0
    assert out["kl"] > 0


def test_a_large_kl_does_not_by_itself_mean_a_bad_edit(
    fake_layered_model, fake_tokenizer
):
    """The whole point of carrying two coordinates: the repair and the damage
    both score a large KL, and only the gold delta tells them apart."""
    engine = Engine(fake_layered_model, fake_tokenizer)

    damaged = engine.assess_edit([1, 2], [1, 7], gold=3)
    repaired = engine.assess_edit([1, 7], [1, 2], gold=3)

    assert damaged["kl"] > 0 and repaired["kl"] > 0
    assert damaged["verdict"] != repaired["verdict"]


class _ScriptedModel:
    """Returns an explicit logit row per context, so a test can pin cases the
    shared fake cannot express -- it spikes one token and ties the rest, which
    makes every non-argmax token rank 2 and hides any damage below the top-1."""

    vocab_size = 50
    hidden_dim = 8

    def __init__(self, rows: dict[tuple, list[float]]):
        self.rows = rows
        self.layers = []

    def __call__(self, inputs, cache=None):
        key = tuple(int(x) for x in inputs[0].tolist())
        row = self.rows[key]
        n = inputs.shape[1]
        return mx.broadcast_to(mx.array(row), (1, n, self.vocab_size))

    def make_cache(self):
        return []


def _row(pairs: dict[int, float], vocab: int = 50) -> list[float]:
    row = [-20.0] * vocab
    for tok, logit in pairs.items():
        row[tok] = logit
    return row


def test_an_edit_that_wrecks_the_gold_rank_is_damaged_even_if_top1_is_unchanged(
    fake_tokenizer,
):
    """The defect this pins: on a poisoned context the top-1 is already wrong,
    so an edit can destroy the answer's standing entirely without touching it.
    Keying the verdict on top-1 alone reports that as 'preserved'."""
    GOLD = 7
    model = _ScriptedModel({
        # full: top-1 is token 1 (wrong); gold sits just behind it at rank 2.
        (1, 2): _row({1: 5.0, GOLD: 4.0, 3: 3.0}),
        # pruned: top-1 is STILL token 1, but gold has collapsed below token 3.
        (1,):   _row({1: 5.0, 3: 3.0, GOLD: -15.0}),
    })
    engine = Engine(model, fake_tokenizer)

    out = engine.assess_edit([1, 2], [1], gold=GOLD)

    assert out["top1_agrees"] is True           # the top-1 never moved
    assert out["gold_delta"] < 0                # but the answer was destroyed
    assert out["gold_after"]["rank"] > out["gold_before"]["rank"]
    assert out["verdict"] == "damaged"


def test_an_edit_that_lifts_the_gold_rank_without_reaching_top1_is_repaired(
    fake_tokenizer,
):
    """The mirror case: real progress toward the answer that stops short of
    changing what the model says. Still an improvement, and worth seeing."""
    GOLD = 7
    model = _ScriptedModel({
        (1, 2): _row({1: 5.0, 3: 3.0, GOLD: -15.0}),
        (1,):   _row({1: 5.0, GOLD: 4.0, 3: 3.0}),
    })
    engine = Engine(model, fake_tokenizer)

    out = engine.assess_edit([1, 2], [1], gold=GOLD)

    assert out["top1_agrees"] is True
    assert out["gold_after"]["rank"] < out["gold_before"]["rank"]
    assert out["verdict"] == "repaired"


def test_sharpening_an_already_correct_answer_is_not_a_repair(fake_tokenizer):
    """Measured on 42 clean-context cells: dropping distractors nudges the gold
    logprob up by ~0.005-0.34 nats while the answer sits at rank 1 throughout.
    Calling that "repaired" mislabelled 20 of the 42. Nothing was repaired --
    the answer was correct before and after."""
    GOLD = 7
    model = _ScriptedModel({
        (1, 2): _row({GOLD: 5.0, 1: 3.0, 3: 2.0}),        # gold already rank 1
        (1,):   _row({GOLD: 5.05, 1: 3.0, 3: 2.0}),       # rank 1, slightly sharper
    })
    engine = Engine(model, fake_tokenizer)

    out = engine.assess_edit([1, 2], [1], gold=GOLD)

    assert out["gold_before"]["rank"] == out["gold_after"]["rank"] == 1
    assert out["gold_delta"] > 0
    assert out["verdict"] == "preserved"


def test_a_small_dip_that_leaves_the_answer_first_is_not_damage(fake_tokenizer):
    """The mirror. The verdict is about the answer's STANDING; a wobble in
    probability that leaves gold at rank 1 has not changed the outcome."""
    GOLD = 7
    model = _ScriptedModel({
        (1, 2): _row({GOLD: 5.0, 1: 3.0, 3: 2.0}),
        (1,):   _row({GOLD: 4.94, 1: 3.0, 3: 2.0}),
    })
    engine = Engine(model, fake_tokenizer)

    out = engine.assess_edit([1, 2], [1], gold=GOLD)

    assert out["gold_delta"] < 0
    assert out["verdict"] == "preserved"


from workbench.context.eviction import keep_by_score


SCORES = {"sys": -1.0, "a": 2.0, "b": 0.5, "c": 1.5}


def test_score_ranking_takes_the_highest_scoring_that_fit():
    kept = keep_by_score(SPANS, SCORES, budget=40)

    # 'a' (20) then 'c' (20) = 40; 'b' and 'sys' would overflow.
    assert kept == {"a", "c"}


def test_a_negative_score_still_sorts_last_rather_than_being_dropped():
    # With room for everything, even a penalised segment is kept: the budget
    # decides what is dropped, the score only decides the order.
    kept = keep_by_score(SPANS, SCORES, budget=60)

    assert kept == {"sys", "a", "b", "c"}


def test_an_unscored_segment_sorts_last_rather_than_raising():
    kept = keep_by_score(SPANS, {"a": 1.0}, budget=20)

    assert kept == {"a"}


def test_attention_ranking_is_score_ranking():
    assert keep_by_attention(SPANS, MASS, budget=30) == keep_by_score(
        SPANS, MASS, budget=30)


def test_an_unscored_segment_does_not_outrank_a_penalised_one():
    # The reason the sentinel is -inf and not 0.0: 'sys' is penalised, 'b' has
    # no score at all, and only one 10-token segment fits.
    kept = keep_by_score(SPANS, {"sys": -1.0}, budget=10)

    assert kept == {"sys"}


def test_a_supplied_ranking_is_a_candidate_filling_the_same_room(
    fake_layered_model, fake_tokenizer
):
    engine = Engine(fake_layered_model, fake_tokenizer)
    spans = {"a": (0, 2), "b": (2, 4), "q": (4, 6)}

    out = engine.evaluate_eviction(
        [1, 2, 3, 4, 5, 6], spans, budget=4, pinned=("q",),
        rankings={"prov": {"a": 1.0, "b": 2.0}})

    assert set(out) == {"recency", "attention", "attention_nosink", "prov"}
    assert out["prov"]["kept"] == {"b", "q"}   # ranked 'b' first, same room as the rest
    assert out["prov"]["n_tokens"] == out["recency"]["n_tokens"] == 4


def test_a_supplied_ranking_may_not_shadow_a_builtin_policy(
    fake_layered_model, fake_tokenizer
):
    engine = Engine(fake_layered_model, fake_tokenizer)
    spans = {"a": (0, 2), "b": (2, 4)}

    with pytest.raises(ValueError, match="shadows"):
        engine.evaluate_eviction([1, 2, 3, 4], spans, budget=2,
                                 rankings={"recency": {"a": 1.0}})


def test_a_ranking_named_nosink_does_not_receive_nosink_mass(
    fake_layered_model, fake_tokenizer
):
    engine = Engine(fake_layered_model, fake_tokenizer)
    tokens = [1, 2, 3, 4]
    spans = {"a": (0, 2), "b": (2, 4)}

    out = engine.evaluate_eviction(
        tokens, spans, budget=2, rankings={"anything_nosink": {"a": 1.0}})

    raw = engine.attention_mass(tokens, spans, None)
    nosink = engine.attention_mass(tokens, spans, None, exclude_sink=True)
    assert raw != nosink   # otherwise the assertion below proves nothing
    assert out["anything_nosink"]["mass"] == raw
    assert out["attention_nosink"]["mass"] == nosink
