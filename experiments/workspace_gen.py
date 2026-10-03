"""Generate multi-writer histories with a known gold answer.

Run: uv run python -m experiments.workspace_gen --fact 1 --slot 3

The claim under test is that structural provenance ranks better than attention
mass and recency. Testing it needs contexts where those rankers DISAGREE about
what to keep and where one of them is right -- which is exactly what several
earlier bench designs failed to produce.

So each case is built around a superseded pair: one source document in two
revisions, the older one stating a wrong fact, the newer the right one, with an
assistant turn derived from the older. Recency keeps the tail and may lose the
correction; attention mass tracks word overlap and the sink; provenance can see
that one chunk supersedes another and that a generation hangs off the stale
one. Distractors are written by other authors so authorship carries signal.

The model is shown a conversation: every segment is one whole rendered chat
turn, followed by a pinned instruction turn, the question, and the assistant
header to complete. The only thing varied between cases is named, never drawn
from one shared seed -- see `generate`."""
from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass

from workbench.engine.engine import GenParams
from workbench.context.model import Editor, Provenance, Segment, SegmentKind
from workbench.server.framing import generation_prompt_text

AUTHORS = ("mira", "hao", "tomas", "wren", "dev")

# (topic, right, wrong). Both answers are invented, distinctive words: neither
# can collide with the other on a prefix, and neither can be guessed.
_FACTS = [
    ("archive password", "HELIOTROPE", "QUILLON"),
    ("day crew badge", "SERAPHIM", "MARLOWE"),
    ("logistics desk password", "TAMARIND", "FENWICK"),
    ("night courier codeword", "OBSIDIAN", "PELLUCID"),
    ("west gate passphrase", "VERMILION", "CASTELLAN"),
    ("relay station callsign", "PERIWINKLE", "GRIMALKIN"),
]

_NOISE = [
    "moving the lunch thread to another channel",
    "does anyone have the runbook link handy",
    "checking the metrics now, one moment",
    "re-running the job with the old config",
    "the dashboard is still loading for me",
    "adding this to the agenda for Thursday",
    "who owns the weekly status doc these days",
    "the build is green again after the rebase",
    "can someone swap my on-call shift on Friday",
    "the vendor call moved to ten tomorrow",
    "I pushed the formatting fixes to the branch",
    "reminder that the office is closed on Monday",
    "the staging box ran out of disk overnight",
    "taking the afternoon off, back on Wednesday",
]

SYS_TEXT = "Answer from the material above. Reply with one word."

# Asker and the authors of the stale chunk and (when not by the asker) the
# correction are fixed, so `by_asker` is the only authorship factor that moves.
_ASKER, _STALE_AUTHOR, _OTHER_AUTHOR = AUTHORS[0], AUTHORS[1], AUTHORS[2]


@dataclass
class Case:
    segments: list[Segment]
    gold_text: str
    answer_word: str
    wrong_word: str
    question: str
    question_id: str
    sys_text: str
    pinned: tuple[str, ...]
    fact: int = 0
    slot: int = 0
    by_asker: bool = False
    noise_seed: int = 0


class HeadroomError(RuntimeError):
    """Raised when no cell of a whole sweep has headroom.

    A sweep where every policy reaches the same outcome in every cell produces
    a table of identical numbers that looks like a null result and is not one.
    Fail loudly instead. A single flat cell is not this: it is skipped."""


def _turn(tokenizer, role: str, author: str, text: str) -> str:
    """One rendered chat turn, as a single string.

    The evictable unit is the whole turn: one segment = one turn = one
    independently removable thing that leaves a valid token stream. Framing
    the parts separately would make the template scaffolding evictable in its
    own right, and pinning all of it would reserve most of the budget.

    The author is rendered INTO the text because the Qwen3 template has no
    speaker field. Left as metadata, authorship is something the provenance
    ranker can see and the attention-mass baseline cannot, and the contest for
    that one signal would not be symmetric."""
    body = text if role == "system" else f"{author}: {text}"
    return tokenizer.apply_chat_template(
        [{"role": role, "content": body}], tokenize=False)


def generate(tokenizer, fact: int = 0, slot: int = 0, by_asker: bool = False,
             noise_seed: int = 0, n_distractors: int = 6) -> Case:
    """One case, with every factor named rather than drawn from one seed.

    `slot` is the correction's position among the contested turns (after the
    stale chunk), `by_asker` whether its author is the questioner,
    `noise_seed` fixes the distractor set so it can be held constant while
    another factor is swept."""
    if not 0 <= slot <= n_distractors + 1:
        raise ValueError(f"slot {slot} outside 0..{n_distractors + 1}")
    topic, right, wrong = _FACTS[fact % len(_FACTS)]
    rng = random.Random(noise_seed)
    source = f"runbook-{fact}.md"

    segs: list[Segment] = []

    def add(kind, role, author, text, editable=Editor.BOTH, **prov):
        sid = f"s{len(segs):02d}"
        segs.append(Segment(id=sid, kind=kind,
                            text=_turn(tokenizer, role, author, text),
                            editable_by=editable,
                            provenance=Provenance(author=author, **prov)))
        return sid

    stale = add(SegmentKind.DOC_CHUNK, "user", _STALE_AUTHOR,
                f"The {topic} is {wrong}.", source=source, revision=1)

    # Noise authors are free draws so authorship is no proxy for "is noise";
    # the texts are sampled WITHOUT replacement so no two turns are identical.
    noise = [("noise", t, rng.choice(AUTHORS))
             for t in rng.sample(_NOISE, n_distractors)]
    # The reply sits at a fixed place among the noise; the correction is then
    # inserted at exactly `slot`, so slot is the one thing that moves.
    later = list(noise)
    later.insert(len(later) // 2, ("reply", None, "model"))
    correction_author = _ASKER if by_asker else _OTHER_AUTHOR
    later.insert(slot, ("correction", f"The {topic} is {right}.",
                        correction_author))

    for what, text, author in later:
        if what == "noise":
            add(SegmentKind.USER_MSG, "user", author, text)
        elif what == "reply":
            # A reply conditioned on the stale chunk: the taint edge the
            # ranker can see and neither recency nor attention mass can.
            add(SegmentKind.ASSISTANT_MSG, "assistant", author,
                f"Going by the runbook, the {topic} is {wrong}.",
                derived_from=(stale,))
        else:
            add(SegmentKind.DOC_CHUNK, "user", author, text,
                source=source, revision=2)

    # The instruction is its own pinned turn, not part of the question.
    sys_id = add(SegmentKind.SYSTEM, "system", "system", SYS_TEXT,
                 editable=Editor.NONE)
    question = f"What is the {topic}?"
    q = add(SegmentKind.USER_MSG, "user", _ASKER, question)
    header = f"s{len(segs):02d}"
    segs.append(Segment(id=header, kind=SegmentKind.SCRATCH,
                        text=generation_prompt_text(tokenizer),
                        editable_by=Editor.NONE,
                        provenance=Provenance(author="framing")))

    return Case(segments=segs, gold_text=right, answer_word=right,
                wrong_word=wrong, question=question, question_id=q,
                sys_text=SYS_TEXT, pinned=(sys_id, q, header), fact=fact,
                slot=slot, by_asker=by_asker, noise_seed=noise_seed)


def cases(tokenizer, n_facts: int = 3, n_distractors: int = 6):
    """The factorial grid: every (fact, slot, by_asker), noise fixed per fact."""
    for f in range(n_facts):
        for slot in range(n_distractors + 2):
            for by_asker in (False, True):
                yield generate(tokenizer, fact=f, slot=slot, by_asker=by_asker,
                               noise_seed=f, n_distractors=n_distractors)


CONTINUATION_TOKENS = 14  # a scoring parameter, per lab-notes finding 31 --
# at 7 tokens a full context scored 5/6 and at 12 it scored 6/6, because the
# model answers in sentence form about half the time. Too small a window
# penalises a FORMAT and reports it as an error of CORRECTNESS. Record it with
# every result; never let it be an unrecorded default.


def scored(engine, tokenizer, tokens, right: str, wrong: str,
           n: int = CONTINUATION_TOKENS) -> str:
    """"right", "wrong" or "neither" from an n-token greedy continuation.

    A trichotomy, not a boolean: "said the poison" and "said nothing" are
    different failures and the headroom check needs to tell them apart.
    Matched on WORD BOUNDARIES -- a substring test would let a continuation
    reading "120" contain "12", and would not separate shared prefixes."""
    out = [ev.token_id for ev in engine.generate(
        tokens, GenParams(max_tokens=n, temperature=0.0))]
    text = tokenizer.decode(out)

    def has(w):
        return re.search(rf"\b{re.escape(w)}\b", text, re.I) is not None

    if has(wrong):
        return "wrong"
    return "right" if has(right) else "neither"


def has_headroom(outcomes: dict[str, bool]) -> bool:
    """True when the policies do not all reach the same outcome.

    The precondition every sweep needs: if every keep-set of a given size
    produces the same answer, the sweep cannot order the policies and the
    table it prints is not evidence about them."""
    return len(set(outcomes.values())) > 1


def require_headroom(outcomes: dict[str, bool]) -> None:
    if not has_headroom(outcomes):
        agreed = next(iter(outcomes.values()), None)
        raise HeadroomError(
            f"no headroom: every policy reached {agreed!r}; "
            "this case cannot order them, so the sweep is not evidence")


def main():
    from workbench.engine.loader import TEST_MODEL, load_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--fact", type=int, default=0)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--by-asker", action="store_true")
    ap.add_argument("--distractors", type=int, default=6)
    args = ap.parse_args()

    _, tok = load_model(TEST_MODEL)
    case = generate(tok, args.fact, args.slot, args.by_asker,
                    noise_seed=args.fact, n_distractors=args.distractors)
    print(f"gold: {case.answer_word} (poison {case.wrong_word}) "
          f"| question: {case.question}")
    print("".join(s.text for s in case.segments))


if __name__ == "__main__":
    main()
