"""The generator exists to create a measurement, not a demo.

Several 2026-08 bench runs failed by producing bands where every keep-set of a
given size scored the same -- nothing to order, so nothing to learn. The
headroom precondition is asserted BEFORE a sweep runs, and loudly, rather than
discovered afterwards in a flat table."""
import re
from types import SimpleNamespace

import pytest

from experiments.workspace_gen import (
    _FACTS,
    _NOISE,
    HeadroomError,
    cases,
    generate,
    has_headroom,
    require_headroom,
    scored,
)
from workbench.engine.engine import GenParams


class _Tok:
    """Chat-template stand-in with Qwen's shape and no BOS."""

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False):
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                      for m in messages)
        return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


TOK = _Tok()


def _seg(case, pred):
    return next(s for s in case.segments if pred(s))


def test_a_case_is_reproducible_from_its_factors():
    a = generate(TOK, fact=1, slot=3, by_asker=True, noise_seed=5)
    b = generate(TOK, fact=1, slot=3, by_asker=True, noise_seed=5)

    def shape(case):
        return [(s.id, s.kind, s.text, s.provenance) for s in case.segments]

    assert shape(a) == shape(b)
    assert a.answer_word == b.answer_word
    assert a.pinned == b.pinned


def test_every_factor_is_recorded_on_the_case():
    case = generate(TOK, fact=2, slot=4, by_asker=True, noise_seed=9)

    assert (case.fact, case.slot, case.by_asker, case.noise_seed) == (2, 4, True, 9)


def test_the_case_carries_a_superseded_pair():
    case = generate(TOK, fact=0, slot=3)
    sources = [s.provenance.source for s in case.segments if s.provenance.source]

    assert any(sources.count(src) >= 2 for src in set(sources))


def test_a_generation_is_derived_from_the_superseded_segment():
    case = generate(TOK, fact=0, slot=3)
    stale = _seg(case, lambda s: s.provenance.revision == 1)

    assert any(stale.id in s.provenance.derived_from for s in case.segments)


def test_the_question_is_open_and_names_neither_candidate():
    for f in range(len(_FACTS)):
        case = generate(TOK, fact=f)
        question = _seg(case, lambda s: s.id == case.question_id).text

        assert case.answer_word.lower() not in question.lower()
        assert case.wrong_word.lower() not in question.lower()


def test_the_right_word_appears_once_and_the_wrong_word_twice():
    """The poison must not be favoured by frequency of mention."""
    for f in range(len(_FACTS)):
        case = generate(TOK, fact=f, slot=2)
        text = " ".join(s.text for s in case.segments)

        assert len(re.findall(case.answer_word, text, re.I)) == 1
        assert len(re.findall(case.wrong_word, text, re.I)) == 2


def test_the_answer_words_do_not_share_a_prefix():
    for _, right, wrong in _FACTS:
        assert right[0] != wrong[0] or right[:3] != wrong[:3]
        assert right != wrong


def test_the_instruction_lives_in_a_pinned_sys_turn_not_the_question():
    case = generate(TOK)
    sys_seg = _seg(case, lambda s: "<|im_start|>system" in s.text)
    question = _seg(case, lambda s: s.id == case.question_id)

    assert sys_seg.id in case.pinned
    assert "one word" in case.sys_text.lower()
    assert "one word" not in question.text.lower()


def test_the_context_ends_with_an_assistant_header_to_complete():
    case = generate(TOK)
    last = case.segments[-1]

    assert last.text == "<|im_start|>assistant\n"
    assert last.id in case.pinned


def test_one_segment_is_one_whole_framed_turn():
    case = generate(TOK, slot=3)
    for s in case.segments[:-1]:
        assert s.text.startswith("<|im_start|>")
        assert s.text.endswith("<|im_end|>\n")
        assert s.text.count("<|im_start|>") == 1


def test_pinned_is_exactly_sys_question_and_generation_header():
    case = generate(TOK)

    assert len(case.pinned) == 3
    assert case.question_id in case.pinned


def test_the_author_is_rendered_into_the_turn_text():
    case = generate(TOK, slot=3)
    correction = _seg(case, lambda s: s.provenance.revision == 2)

    assert f"{correction.provenance.author}: " in correction.text


@pytest.mark.parametrize("slot", range(8))
def test_the_correction_lands_exactly_at_the_requested_slot(slot):
    case = generate(TOK, slot=slot)
    contested = [s for s in case.segments if s.id not in case.pinned]
    correction = _seg(case, lambda s: s.provenance.revision == 2)

    # the stale chunk sits at index 0, ahead of the shuffled turns
    assert contested.index(correction) - 1 == slot


def test_by_asker_sets_the_correction_authorship_and_nothing_else_moves():
    a = generate(TOK, fact=1, slot=3, by_asker=False, noise_seed=1)
    b = generate(TOK, fact=1, slot=3, by_asker=True, noise_seed=1)

    def author(case):
        return _seg(case, lambda s: s.provenance.revision == 2).provenance.author

    asker = _seg(a, lambda s: s.id == a.question_id).provenance.author
    assert author(a) != asker and author(b) == asker
    assert [s.provenance.author for s in a.segments
            if s.provenance.revision != 2] == [
        s.provenance.author for s in b.segments if s.provenance.revision != 2]


def test_the_noise_does_not_move_when_the_slot_is_swept():
    def noise(case):
        return [s.text for s in case.segments
                if s.provenance.revision == 0 and not s.provenance.derived_from
                and s.id not in case.pinned]

    assert noise(generate(TOK, slot=1, noise_seed=4)) == noise(
        generate(TOK, slot=6, noise_seed=4))


def test_noise_is_drawn_without_replacement():
    assert len(_NOISE) >= 12
    assert len(set(_NOISE)) == len(_NOISE)
    for seed in range(20):
        case = generate(TOK, noise_seed=seed)
        texts = [s.text for s in case.segments]
        assert len(texts) == len(set(texts))


def test_the_grid_covers_every_slot_and_both_authorships_for_each_fact():
    grid = list(cases(TOK, n_facts=3, n_distractors=6))

    assert len(grid) == 3 * 8 * 2
    assert {c.slot for c in grid} == set(range(8))
    assert {c.by_asker for c in grid} == {True, False}
    assert {c.fact for c in grid} == {0, 1, 2}
    # noise is held per fact, so it is not confounded with the swept factors
    for f in range(3):
        assert len({c.noise_seed for c in grid if c.fact == f}) == 1


def test_the_stale_chunk_is_older_earlier_and_the_reply_derives_from_it():
    for slot in range(8):
        case = generate(TOK, slot=slot)
        ids = [s.id for s in case.segments]
        stale = _seg(case, lambda s: s.provenance.revision == 1)
        correction = _seg(case, lambda s: s.provenance.revision == 2)

        assert ids.index(stale.id) < ids.index(correction.id)
        assert any(stale.id in s.provenance.derived_from for s in case.segments)


def test_headroom_exists_when_outcomes_differ():
    assert has_headroom({"recency": True, "provenance": False}) is True


def test_no_headroom_when_every_policy_agrees():
    assert has_headroom({"recency": True, "provenance": True}) is False


def test_require_headroom_raises_rather_than_returning_a_flat_table():
    with pytest.raises(HeadroomError, match="no headroom"):
        require_headroom({"recency": True, "provenance": True})


class _Engine:
    def __init__(self, text):
        self.text = text

    def generate(self, tokens, params):
        assert isinstance(params, GenParams) and params.temperature == 0.0
        for ch in self.text:
            yield SimpleNamespace(token_id=ord(ch))


@pytest.mark.parametrize("said,verdict", [
    ("The password is QUILLON.", "wrong"),
    ("the password is heliotrope", "right"),
    ("I cannot tell.", "neither"),
    ("HELIOTROPE, not QUILLON", "wrong"),
    ("HELIOTROPEX", "neither"),
])
def test_scored_is_a_trichotomy_on_word_boundaries(said, verdict):
    got = scored(_Engine(said), TOK, [1], "HELIOTROPE", "QUILLON", n=40)

    assert got == verdict
