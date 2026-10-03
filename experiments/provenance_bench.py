"""Does provenance rank better than attention mass and recency?

Phase 2 is the gate for the whole multiplayer direction: if provenance does not
beat the incumbents here, phases 3 and 4 are not worth building and the result
is a lab-note entry saying so.

Run: uv run python -m experiments.provenance_bench            (0.6B, plumbing only)
     uv run python -m experiments.provenance_bench \
         --model mlx-community/Qwen3-8B-8bit                   (real result)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from workbench.context.manager import ContextManager
from workbench.context.model import ContextObject, append_event
from workbench.context.provenance import DEFAULT_WEIGHTS, SIGNALS, score_all
from workbench.engine.engine import Engine
from workbench.engine.loader import TEST_MODEL, load_model
from experiments.workspace_gen import (
    CONTINUATION_TOKENS,
    HeadroomError,
    cases,
    has_headroom,
    scored,
)

# Chosen from `--band` on the 0.6B over the 48-case grid (see the Task 7 report),
# not inherited: the old (0.4, 0.55, 0.7) was picked against an unframed
# ~100-token context. The framing, the sys turn, the question and the header
# are reserved off the top, so below ~0.55 the correction survives under no
# policy (<=38%) and every cell is uniform, while from 0.60 the policies
# separate on whether the poison is dropped. Above ~0.90 everything is kept.
# Re-run `--band` if the corpus or the pinned turns change.
BUDGET_FRACTIONS = (0.6, 0.7, 0.8)

SMOKE_BANNER = "PLUMBING ONLY — this run makes no claim about any ranker."


def _rankings(segments) -> dict[str, dict[str, float]]:
    """The composite, one ranking per signal ablated out, and one with the
    two position-confounded signals ablated together.

    A composite that wins tells you nothing about which signal did the work,
    so each one is also scored with that signal's weight set to zero.

    The joint ablation is not symmetry for its own sake. `derived_from`
    records everything that was in the projection, not what the model drew
    on, so `referenced` is close to "older than the most recent assistant
    turn" -- a positional fact, strongly collinear with `depth` and opposite
    in sign. Ablating either one alone leaves the other carrying the
    position signal, so a per-signal sweep cannot see the confound at all.
    Without this cell, a composite that beats recency might BE recency."""
    out = {"provenance": score_all(segments)}
    for name in SIGNALS:
        weights = dict(DEFAULT_WEIGHTS, **{name: 0.0})
        out[f"provenance_no_{name}"] = score_all(segments, weights)
    out["provenance_no_position"] = score_all(
        segments, dict(DEFAULT_WEIGHTS, depth=0.0, referenced=0.0))
    return out


def collinearity(segments, pinned=()) -> float:
    """Pearson r between the `referenced` and `depth` signals over `segments`.

    Pinned segments are excluded: they are never ranked, so they say nothing
    about whether the ranked signals are one signal.

    Reported beside the table because it decides how the table may be read:
    at high r the two signals are one signal, and a composite win is not
    evidence for provenance over recency."""
    import statistics

    ranked = [s for s in segments if s.id not in set(pinned)]
    xs = [SIGNALS["referenced"](segments, s.id) for s in ranked]
    ys = [SIGNALS["depth"](segments, s.id) for s in ranked]
    if len(xs) < 2 or statistics.pstdev(xs) == 0 or statistics.pstdev(ys) == 0:
        return 0.0
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)
    return cov / (statistics.pstdev(xs) * statistics.pstdev(ys))


def _tokenize(case, tokenizer):
    ctx = ContextObject()
    for seg in case.segments:
        ctx.apply(append_event(seg))
    return ContextManager(ctx, tokenizer).to_tokens()


def _pruned(tokenized, kept) -> list[int]:
    return [t for sid in sorted(kept, key=lambda s: tokenized.spans[s][0])
            for t in tokenized.tokens[slice(*tokenized.spans[sid])]]


def _wrong_ids(case) -> set[str]:
    """Segments that carry the poison: the stale chunk and the reply."""
    return {s.id for s in case.segments
            if case.wrong_word.lower() in s.text.lower()}


def _correction_id(case) -> str:
    return next(s.id for s in case.segments if s.provenance.revision == 2)


def band_table(engine, tokenizer, grid, fractions) -> list[dict]:
    """For each candidate budget fraction and policy: how often the poison is
    entirely dropped, how often the correction survives, and how often both.

    The decisive band is where BOTH hold for some policies and not for others.
    Framing and the pinned turns are reserved off the top, so fractions chosen
    against an unframed context do not sit there; this measures where it is
    instead of assuming. It runs the model only for attention mass."""
    cells: dict[tuple, list[tuple[bool, bool]]] = {}
    for case in grid:
        tokenized = _tokenize(case, tokenizer)
        poison, fix = _wrong_ids(case), _correction_id(case)
        for frac in fractions:
            res = engine.evaluate_eviction(
                tokenized.tokens, tokenized.spans,
                budget=int(len(tokenized.tokens) * frac),
                pinned=case.pinned, rankings=_rankings(case.segments))
            for name, r in res.items():
                cells.setdefault((frac, name), []).append(
                    (not (poison & r["kept"]), fix in r["kept"]))
    out = []
    for (frac, name), v in cells.items():
        n = len(v)
        out.append({"frac": frac, "policy": name, "n": n,
                    "poison_dropped": sum(d for d, _ in v) / n,
                    "correction_kept": sum(k for _, k in v) / n,
                    "both": sum(d and k for d, k in v) / n})
    return out


def run(engine, tokenizer, grid, fractions=BUDGET_FRACTIONS,
        smoke: bool = False, n_tokens: int = CONTINUATION_TOKENS):
    """Returns (rows, stats).

    `smoke` skips the per-policy scoring and the headroom precondition, for a
    model that cannot answer the question, and does not skip cases that the
    full context fails. Budget parity is still asserted.

    A case the full context does not answer carries no information about any
    policy, so outside smoke it is skipped and counted. A flat cell (every
    policy the same) is skipped and counted too; only a sweep with no
    headroom anywhere raises."""
    rows: list[dict] = []
    stats = {"cases": 0, "skipped": 0, "cells": 0, "flat": 0,
             "full_right": 0, "continuation_tokens": n_tokens}
    for case in grid:
        stats["cases"] += 1
        tokenized = _tokenize(case, tokenizer)
        full = scored(engine, tokenizer, tokenized.tokens, case.answer_word,
                      case.wrong_word, n_tokens)
        stats["full_right"] += full == "right"
        if full != "right" and not smoke:
            stats["skipped"] += 1
            continue
        factors = {"fact": case.fact, "slot": case.slot,
                   "by_asker": case.by_asker, "noise_seed": case.noise_seed,
                   "continuation_tokens": n_tokens}

        for frac in fractions:
            budget = int(len(tokenized.tokens) * frac)
            res = engine.evaluate_eviction(
                tokenized.tokens, tokenized.spans, budget=budget,
                pinned=case.pinned, rankings=_rankings(case.segments))

            outcomes: dict[str, bool] = {}
            cell: list[dict] = []
            for name, result in res.items():
                row = {**factors, "budget_frac": frac, "policy": name,
                       "n_tokens": result["n_tokens"], "kl": result["kl"],
                       "kept": sorted(result["kept"])}
                if not smoke:
                    verdict = scored(engine, tokenizer,
                                     _pruned(tokenized, result["kept"]),
                                     case.answer_word, case.wrong_word,
                                     n_tokens)
                    outcomes[name] = verdict == "right"
                    row.update(verdict=verdict, correct=outcomes[name],
                               full_verdict=full)
                cell.append(row)
            rows.extend(cell)

            # Budgets must be comparable or the table compares sizes, not
            # rankings. Assert it rather than trusting the fill.
            # A greedy fill stops when the next item will not fit, so two
            # fills over the same budget can legitimately differ by up to one
            # CONTESTED segment: pinned spans are reserved off the top and
            # never contested, and the question is usually the longest span,
            # so including them would loosen the invariant this states.
            widest = max(e - s for sid, (s, e) in tokenized.spans.items()
                         if sid not in case.pinned)
            held = {r["n_tokens"] for r in cell}
            assert max(held) - min(held) <= widest, (
                f"policies held {sorted(held)} tokens at budget {budget} "
                f"(widest contested segment {widest}); that is a comparison "
                "of sizes, not of rankings")

            if not smoke:
                stats["cells"] += 1
                if not has_headroom(outcomes):
                    stats["flat"] += 1
    if not smoke and stats["cells"] and stats["flat"] == stats["cells"]:
        raise HeadroomError(
            f"no headroom: all {stats['cells']} cells were flat; "
            "the sweep cannot order the policies and is not evidence")
    return rows, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=TEST_MODEL)
    ap.add_argument("--facts", type=int, default=3,
                    help="facts in the factorial grid (each x 8 slots x 2 "
                         "authorships)")
    ap.add_argument("--out", default="experiments/results/provenance_bench.json",
                    help="per-cell rows, kept so a later entry can re-verify them")
    ap.add_argument("--smoke", action="store_true",
                    help="plumbing check: skip per-policy scoring and the "
                         "headroom precondition, write no results file")
    ap.add_argument("--band", action="store_true",
                    help="print how often the poison is dropped and the "
                         "correction survives per budget fraction, and exit")
    args = ap.parse_args()

    if args.model == TEST_MODEL:
        print("NOTE: running on the 0.6B test model. Per docs/roadmap.md, the")
        print("small end of the band checks that the plumbing runs and nothing")
        print("else -- a construction that moves nothing at 0.6B steers")
        print("reliably at 8B on the same prompts. A CLAIM THAT THIS RANKER")
        print("WORKS HAS TO BE MADE AT 8B:")
        print("  uv run python -m experiments.provenance_bench \\")
        print("      --model mlx-community/Qwen3-8B-8bit")
        print()
    if args.smoke:
        print(SMOKE_BANNER)
        print()

    model, tokenizer = load_model(args.model)
    engine = Engine(model, tokenizer)
    grid = list(cases(tokenizer, n_facts=args.facts))

    if args.band:
        fracs = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
                 0.80, 0.90)
        table = band_table(engine, tokenizer, grid, fracs)
        print(f"{len(grid)} cases; per budget fraction and policy")
        print(f"{'frac':>5} {'policy':<26} {'poison dropped':>15} "
              f"{'correction kept':>16} {'both':>6}")
        for r in sorted(table, key=lambda r: (r["frac"], r["policy"])):
            print(f"{r['frac']:>5.2f} {r['policy']:<26} "
                  f"{r['poison_dropped']:>15.0%} {r['correction_kept']:>16.0%} "
                  f"{r['both']:>6.0%}")
        return

    rows, stats = run(engine, tokenizer, grid, smoke=args.smoke)

    if not args.smoke:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=1))

    print(f"full-context precondition: {stats['full_right']} of "
          f"{stats['cases']} cases answered 'right' unpruned "
          f"({stats['continuation_tokens']}-token continuation)")
    if not args.smoke:
        print(f"{stats['skipped']} of {stats['cases']} cases skipped: the full "
              "context did not answer")
        print(f"{stats['flat']} of {stats['cells']} cells were flat (every "
              "policy the same) and carry no ordering")
    print()

    print("signal firing (share of ranked segments non-zero, over all cases):")
    for name, fn in SIGNALS.items():
        vals = [fn(c.segments, s.id) != 0 for c in grid
                for s in c.segments if s.id not in c.pinned]
        print(f"  {name:<14} {sum(vals) / len(vals):>6.1%}")
    print()

    rs = [collinearity(c.segments, c.pinned) for c in grid]
    mean_r = sum(rs) / len(rs) if rs else 0.0
    print(f"referenced-vs-depth collinearity: r = {mean_r:+.2f}")
    if abs(mean_r) > 0.7:
        print("  WARNING: those two signals are nearly one signal here.")
        print("  Read provenance_no_position, not provenance, as the result:")
        print("  a composite win at this r may be recency under another name.")
    print()

    if args.smoke:
        print("per cell: distinct keep-sets among policies, held-token spread")
        cells: dict[tuple, list[dict]] = {}
        for r in rows:
            cells.setdefault((r["fact"], r["slot"], r["by_asker"],
                              r["budget_frac"]), []).append(r)
        for frac in BUDGET_FRACTIONS:
            cs = [c for k, c in cells.items() if k[-1] == frac]
            distinct = [len({tuple(r["kept"]) for r in c}) for c in cs]
            spread = [max(r["n_tokens"] for r in c) - min(r["n_tokens"] for r in c)
                      for c in cs]
            print(f"  budget {frac:.2f}: distinct keep-sets per cell "
                  f"min {min(distinct)} mean {sum(distinct) / len(distinct):.1f} "
                  f"max {max(distinct)} of {len(cs[0])} policies; "
                  f"held-token spread max {max(spread)}")
        print()
        print(SMOKE_BANNER)
        return

    by_policy: dict[str, list[bool]] = {}
    for r in rows:
        by_policy.setdefault(r["policy"], []).append(r["correct"])
    print(f"{'policy':<28} {'correct':>8}  {'n':>4}")
    for policy, hits in sorted(by_policy.items(),
                               key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"{policy:<28} {sum(hits) / len(hits):>7.1%}  {len(hits):>4}")


if __name__ == "__main__":
    main()
