"""The generator exists to create a measurement, not a demo.

Several 2026-08 bench runs failed by producing bands where every keep-set of a
given size scored the same -- nothing to order, so nothing to learn. The
headroom precondition is asserted BEFORE a sweep runs, and loudly, rather than
discovered afterwards in a flat table."""
import pytest

from experiments.workspace_gen import (
    Case,
    HeadroomError,
    generate,
    has_headroom,
    require_headroom,
)


def test_a_case_is_reproducible_from_its_seed():
    a = generate(seed=7)
    b = generate(seed=7)

    assert [s.text for s in a.segments] == [s.text for s in b.segments]
    assert a.gold_text == b.gold_text


def test_different_seeds_give_different_cases():
    assert generate(seed=1).segments[0].text != generate(seed=2).segments[0].text


def test_the_case_carries_a_superseded_pair():
    case = generate(seed=3)
    sources = [s.provenance.source for s in case.segments if s.provenance.source]

    assert any(sources.count(src) >= 2 for src in set(sources))


def test_the_question_is_pinned_so_no_policy_evicts_it():
    case = generate(seed=3)

    assert case.pinned
    assert all(p in {s.id for s in case.segments} for p in case.pinned)


def test_a_generation_is_derived_from_the_superseded_segment():
    case = generate(seed=3)

    assert any(s.provenance.derived_from for s in case.segments)


def test_headroom_exists_when_outcomes_differ():
    assert has_headroom({"recency": True, "provenance": False}) is True


def test_no_headroom_when_every_policy_agrees():
    assert has_headroom({"recency": True, "provenance": True}) is False
    assert has_headroom({"recency": False, "provenance": False}) is False


def test_require_headroom_raises_rather_than_returning_a_flat_table():
    with pytest.raises(HeadroomError, match="no headroom"):
        require_headroom({"recency": True, "provenance": True})


def test_the_correction_does_not_sit_at_a_fixed_rank_across_seeds():
    """The generator must not predetermine the answer.

    If the load-bearing segment always lands in the same place with the same
    author, a sweep measures the layout rather than the ranker — the trap
    docs/lab-notes.md escaped by placing the fact at each contested slot in
    turn. This asserts the layout actually moves; it says nothing about
    whether any policy ranks it well, which is Task 6's question."""
    from workbench.context.provenance import score_all

    outranked = set()
    authors = set()
    for seed in range(12):
        case = generate(seed=seed)
        scores = score_all(case.segments)
        correction = [s for s in case.segments if s.provenance.revision == 2][0]
        outranked.add(sum(1 for s in case.segments
                          if scores[s.id] > scores[correction.id]))
        authors.add(correction.provenance.author)

    assert len(outranked) > 1, f"correction always outranked by {outranked}"
    assert len(authors) > 1, f"correction always authored by {authors}"
