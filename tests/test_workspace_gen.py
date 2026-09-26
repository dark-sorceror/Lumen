"""The generator exists to create a measurement, not a demo.

Several 2026-08 bench runs failed by producing bands where every keep-set of a
given size scored the same -- nothing to order, so nothing to learn. The
headroom precondition is asserted BEFORE a sweep runs, and loudly, rather than
discovered afterwards in a flat table."""
import pytest

from experiments.workspace_gen import (
    _FACTS,
    HeadroomError,
    generate,
    has_headroom,
    require_headroom,
)


def test_a_case_is_reproducible_from_its_seed():
    a = generate(seed=7)
    b = generate(seed=7)

    def shape(case):
        return [(s.id, s.kind, s.provenance.author, s.provenance.source,
                 s.provenance.revision, s.provenance.derived_from)
                for s in case.segments]

    assert [s.text for s in a.segments] == [s.text for s in b.segments]
    assert shape(a) == shape(b)
    assert a.gold_text == b.gold_text
    assert a.pinned == b.pinned


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


def test_the_correction_does_not_sit_at_a_fixed_slot_or_role():
    """The generator must not predetermine the answer.

    If the load-bearing segment always lands in the same place, or always
    belongs to the asker (or always does not), a sweep measures the layout
    rather than the ranker -- the trap docs/lab-notes.md escaped by placing
    the fact at each contested slot in turn.

    This asserts the SLOT and the asker ROLE both move. An earlier version
    asserted author NAMES and an outranked COUNT, and both pass under the
    original fixed layout: names vary because the author list is shuffled per
    seed, and the count varies because noise authorship varies. It says
    nothing about whether any policy ranks the correction well, which is the
    bench's question rather than the generator's."""
    slots = set()
    asker_roles = set()
    for seed in range(12):
        case = generate(seed=seed)
        ids = [s.id for s in case.segments]
        correction = next(s for s in case.segments if s.provenance.revision == 2)
        asker = next(s for s in case.segments
                     if s.id == case.pinned[0]).provenance.author
        slots.add(ids.index(correction.id))
        asker_roles.add(correction.provenance.author == asker)

    assert len(slots) > 1, f"correction always at slot {slots}"
    assert len(asker_roles) > 1, (
        f"correction's asker role never varies: always "
        f"{'the asker' if True in asker_roles else 'a non-asker'}")


def test_the_stale_chunk_is_older_earlier_and_the_reply_derives_from_it():
    for seed in range(12):
        case = generate(seed=seed)
        ids = [s.id for s in case.segments]
        stale = next(s for s in case.segments if s.provenance.revision == 1)
        correction = next(s for s in case.segments if s.provenance.revision == 2)

        assert stale.provenance.revision < correction.provenance.revision
        assert ids.index(stale.id) < ids.index(correction.id)
        assert any(stale.id in s.provenance.derived_from for s in case.segments)


@pytest.mark.parametrize("claim,right,wrong", _FACTS)
def test_every_fact_can_be_split_into_a_question_without_leaking(claim, right, wrong):
    assert " is " in claim, f"{claim!r} would leak the answer into the question"
    assert right != wrong
