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
