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
    ("the cache ttl is 45 seconds", "45", "120"),
    ("the shard count is 12", "12", "8"),
    ("the backoff cap is 30 seconds", "30", "60"),
    ("the queue depth limit is 500", "500", "200"),
    ("the rollout wave is 3", "3", "5"),
    ("the heartbeat interval is 7 seconds", "7", "15"),
    ("the leader lease is 20 seconds", "20", "50"),
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

    # Everything after the stale chunk is shuffled into place, so the
    # correction's slot and author are drawn from the seed rather than fixed:
    # a fixed layout would make a sweep measure the layout, not the ranker.
    # The correction may be written by the asker or by anyone else, and noise
    # authors are drawn freely so authorship is no proxy for "is noise".
    later = [("noise", rng.choice(_NOISE), rng.choice(authors))
             for _ in range(n_distractors)]
    later.insert(rng.randint(0, len(later)), ("reply", None, "model"))
    later.insert(rng.randint(0, len(later)),
                 ("correction", f"{claim}.", rng.choice(authors)))

    for what, text, author in later:
        if what == "noise":
            add(SegmentKind.USER_MSG, text, author)
        elif what == "reply":
            # A reply conditioned on the stale chunk: the taint edge the
            # ranker can see and neither recency nor attention mass can.
            add(SegmentKind.ASSISTANT_MSG, f"Going by the runbook, {wrong}.",
                author, derived_from=(stale,))
        else:
            add(SegmentKind.DOC_CHUNK, text, author,
                source=source, revision=2)

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
