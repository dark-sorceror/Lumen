"""Relevance signals read off a segment's provenance rather than the model.

The project's measured position is that attention mass is a weak ranker: it
outranks the best distractor by 0.90-1.14x at six of seven contested slots and
is inverted at three of them, and the one strong case is the attention sink.
The internal candidates are spent. These are the external ones.

Every signal here is a STRUCTURAL fact -- who wrote a segment, which revision
of a source it is, what a generation was conditioned on. None of them is
inferred from text similarity, and none of them needs a label, which is the
whole point: gold scoring needs a known-correct token and KL is anti-correlated
with quality on exactly the poisoned contexts that matter.

Each signal is (segments, target) -> float in 0..1, returns 0.0 for an unknown
id rather than raising, and is independently ablatable."""
from __future__ import annotations

from collections.abc import Callable

from workbench.context.model import Segment


def _by_id(segments: list[Segment]) -> dict[str, Segment]:
    return {s.id: s for s in segments}


def superseded(segments: list[Segment], target: str) -> float:
    """1.0 when a later segment carries the same source at a higher revision.

    Structural supersession only -- two chunks of the same document, one newer.
    Deciding that two sentences disagree is a much larger problem and is
    deliberately out of scope."""
    seg = _by_id(segments).get(target)
    if seg is None or not seg.provenance.source:
        return 0.0
    for other in segments:
        p = other.provenance
        if (other.id != target
                and p.source == seg.provenance.source
                and p.revision > seg.provenance.revision):
            return 1.0
    return 0.0


def tainted(segments: list[Segment], target: str) -> float:
    """1.0 when this segment was generated from something now superseded.

    Whether this beats chance is itself a thing to measure. It is recorded as
    a signal, not assumed to be a good one."""
    seg = _by_id(segments).get(target)
    if seg is None:
        return 0.0
    return 1.0 if any(superseded(segments, src) == 1.0
                      for src in seg.provenance.derived_from) else 0.0


def referenced(segments: list[Segment], target: str) -> float:
    """1.0 when a later generation was conditioned on this segment."""
    if target not in _by_id(segments):
        return 0.0
    return 1.0 if any(target in s.provenance.derived_from
                      for s in segments if s.id != target) else 0.0


def authored_by_asker(segments: list[Segment], target: str) -> float:
    """1.0 when this segment shares an author with the newest segment.

    In a shared context the live question's author is the one whose earlier
    material is most likely load-bearing for it."""
    index = _by_id(segments)
    seg = index.get(target)
    if seg is None or not segments:
        return 0.0
    return 1.0 if seg.provenance.author == segments[-1].provenance.author else 0.0


def depth(segments: list[Segment], target: str) -> float:
    """Position, newest at 1.0 and oldest at 0.0.

    The recency baseline as a component, so the composite can be ablated
    against it rather than merely compared with it."""
    ids = [s.id for s in segments]
    if target not in ids or len(ids) < 2:
        return 0.0 if target not in ids else 1.0
    return ids.index(target) / (len(ids) - 1)


SIGNALS: dict[str, Callable[[list[Segment], str], float]] = {
    "superseded": superseded,
    "tainted": tainted,
    "referenced": referenced,
    "authored_by_asker": authored_by_asker,
    "depth": depth,
}

# Negative where the signal is evidence AGAINST keeping a segment. These are a
# starting point to be ablated, not a tuned result: the point of the bench is
# to find out which of them carry the weight.
DEFAULT_WEIGHTS: dict[str, float] = {
    "superseded": -2.0,
    "tainted": -1.0,
    "referenced": 1.5,
    "authored_by_asker": 0.5,
    "depth": 1.0,
}


def score(segments: list[Segment], target: str,
          weights: dict[str, float] | None = None) -> float:
    w = DEFAULT_WEIGHTS if weights is None else weights
    return sum(w.get(name, 0.0) * fn(segments, target)
               for name, fn in SIGNALS.items())


def score_all(segments: list[Segment],
              weights: dict[str, float] | None = None) -> dict[str, float]:
    return {s.id: score(segments, s.id, weights) for s in segments}
