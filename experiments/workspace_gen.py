"""Generate multi-writer histories with a known gold answer.

Run: uv run python -m experiments.workspace_gen --seed 3

The claim under test is that structural provenance ranks better than attention
mass and recency. Testing it needs contexts where those rankers DISAGREE about
what to keep and where one of them is right -- which is exactly what several
earlier bench designs failed to produce.

So each case is built around a superseded pair: one source document in two
revisions, the older one stating a wrong fact, the newer the right one, with an
assistant turn derived from the older. Recency keeps the tail and may lose the
correction; attention mass tracks word overlap and the sink; provenance can see
that one chunk supersedes another and that a generation hangs off the stale
one. Distractors are written by other authors so authorship carries signal."""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

from workbench.context.model import Provenance, Segment, SegmentKind

AUTHORS = ("mira", "hao", "tomas", "wren", "dev")

_FACTS = [
    ("the retry budget is scoped per-host", "per-host", "per-request"),
    ("the writer timeout is 340 ms", "340", "900"),
    ("the canary version is 2.9", "2.9", "2.4"),
    ("the pool ceiling is 64 connections", "64", "16"),
]

_NOISE = [
    "moving the lunch thread to another channel",
    "does anyone have the runbook link handy",
    "checking the metrics now, one moment",
    "re-running the job with the old config",
    "the dashboard is still loading for me",
    "adding this to the agenda for Thursday",
]


@dataclass
class Case:
    segments: list[Segment]
    gold_text: str
    question: str
    pinned: tuple[str, ...]


class HeadroomError(RuntimeError):
    """Raised when no policy's outcome differs from any other's.

    A sweep over a case with no headroom produces a table of identical numbers
    that looks like a null result and is not one. Fail loudly instead."""


def generate(seed: int, n_authors: int = 3, n_distractors: int = 6) -> Case:
    rng = random.Random(seed)
    claim, right, wrong = rng.choice(_FACTS)
    authors = list(AUTHORS[:max(2, n_authors)])
    rng.shuffle(authors)
    asker = authors[0]
    source = f"runbook-{seed}.md"

    segs: list[Segment] = []

    def add(kind, text, author, **prov):
        sid = f"s{len(segs):02d}"
        segs.append(Segment(id=sid, kind=kind, text=text,
                            provenance=Provenance(author=author, **prov)))
        return sid

    stale = add(SegmentKind.DOC_CHUNK,
                f"{claim.replace(right, wrong)}.",
                authors[1 % len(authors)], source=source, revision=1)

    for i in range(n_distractors // 2):
        add(SegmentKind.USER_MSG, rng.choice(_NOISE),
            authors[(i + 1) % len(authors)])

    # A reply conditioned on the stale chunk: the taint edge the ranker can see
    # and neither recency nor attention mass can.
    add(SegmentKind.ASSISTANT_MSG, f"Going by the runbook, {wrong}.",
        "model", derived_from=(stale,))

    add(SegmentKind.DOC_CHUNK, f"{claim}.",
        authors[2 % len(authors)], source=source, revision=2)

    for i in range(n_distractors - n_distractors // 2):
        add(SegmentKind.USER_MSG, rng.choice(_NOISE),
            authors[(i + 2) % len(authors)])

    question = f"In one word, is {claim.split(' is ')[0]} {right} or {wrong}?"
    q = add(SegmentKind.USER_MSG, question, asker)

    return Case(segments=segs, gold_text=right, question=question, pinned=(q,))


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--authors", type=int, default=3)
    ap.add_argument("--distractors", type=int, default=6)
    args = ap.parse_args()

    case = generate(args.seed, args.authors, args.distractors)
    print(f"gold: {case.gold_text} | question: {case.question}")
    for s in case.segments:
        p = s.provenance
        print(f"{s.id} {p.author:<7} src={p.source} rev={p.revision} "
              f"from={p.derived_from} {s.text}")


if __name__ == "__main__":
    main()
