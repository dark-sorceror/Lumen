"""Policies for choosing what to drop from a context under a token budget.

Deliberately pure: a policy sees spans, optionally measured attention mass, and
a budget -- never a model. That keeps the *decision* separable from the
*evaluation*, so a policy can be unit tested for what it selects and then
scored, independently, by what the selection costs at the readout.

Both policies here fill the budget greedily under a different ordering. Greedy
is not optimal -- choosing a maximum-value subset under a size budget is
knapsack -- but a policy that has to solve knapsack per turn is not a policy
you can run in a decode loop, and the baseline it is being compared against
(recency) is greedy too."""


def _lengths(spans: dict[str, tuple[int, int]]) -> dict[str, int]:
    return {sid: end - start for sid, (start, end) in spans.items()}


def _fill(order: list[str], spans: dict[str, tuple[int, int]], budget: int) -> set[str]:
    """Walk `order`, taking any segment that still fits.

    Skips over an oversized segment rather than stopping at it: a single long
    low-value segment should not block every shorter one behind it."""
    lengths = _lengths(spans)
    kept: set[str] = set()
    used = 0
    for sid in order:
        if used + lengths[sid] <= budget:
            kept.add(sid)
            used += lengths[sid]
    return kept


def keep_by_recency(spans: dict[str, tuple[int, int]], budget: int) -> set[str]:
    """Baseline: keep the newest segments that fit.

    This is what almost every production context manager does, and it is a
    genuinely strong baseline -- recent tokens really are the most predictive
    of the next one. Any measurement-driven policy has to beat it, not merely
    differ from it."""
    newest_first = sorted(spans, key=lambda sid: spans[sid][0], reverse=True)
    return _fill(newest_first, spans, budget)


def keep_by_attention(
    spans: dict[str, tuple[int, int]],
    mass: dict[str, float],
    budget: int,
) -> set[str]:
    """Keep the segments the model actually looked at.

    Ranks on total mass, not mass-per-token, so a long well-attended segment
    outranks a short one -- the question is how much of the model's attention
    the segment holds, not how efficiently it holds it. Segments with no
    measurement sort last rather than raising."""
    richest_first = sorted(spans, key=lambda sid: mass.get(sid, 0.0), reverse=True)
    return _fill(richest_first, spans, budget)


def rebuild_tokens(
    tokens: list[int],
    spans: dict[str, tuple[int, int]],
    kept: set[str],
) -> list[int]:
    """Splice the kept segments back into one token list, in POSITIONAL order.

    A policy ranks by value; the model reads in order. Emitting the kept
    segments in the policy's ranking would hand the model scrambled text and
    the divergence measured afterwards would be reporting the scramble, not
    the eviction."""
    out: list[int] = []
    for sid in sorted(kept & set(spans), key=lambda s: spans[s][0]):
        start, end = spans[sid]
        out.extend(tokens[start:end])
    return out
