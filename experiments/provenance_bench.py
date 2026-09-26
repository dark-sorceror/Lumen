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
from experiments.workspace_gen import generate, require_headroom

# Deliberately tight. At a generous budget every policy keeps everything and
# there is nothing to compare -- the failure mode of the earlier runs.
BUDGET_FRACTIONS = (0.4, 0.55, 0.7)


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


def collinearity(segments) -> float:
    """Pearson r between the `referenced` and `depth` signals over `segments`.

    Reported beside the table because it decides how the table may be read:
    at high r the two signals are one signal, and a composite win is not
    evidence for provenance over recency."""
    import statistics

    xs = [SIGNALS["referenced"](segments, s.id) for s in segments]
    ys = [SIGNALS["depth"](segments, s.id) for s in segments]
    if len(xs) < 2 or statistics.pstdev(xs) == 0 or statistics.pstdev(ys) == 0:
        return 0.0
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)
    return cov / (statistics.pstdev(xs) * statistics.pstdev(ys))


def run(engine, tokenizer, seeds, fractions=BUDGET_FRACTIONS) -> list[dict]:
    rows: list[dict] = []
    for seed in seeds:
        case = generate(seed=seed)
        ctx = ContextObject()
        for seg in case.segments:
            ctx.apply(append_event(seg))
        tokenized = ContextManager(ctx, tokenizer).to_tokens()
        gold = tokenizer.encode(case.gold_text)[0]

        for frac in fractions:
            budget = int(len(tokenized.tokens) * frac)
            scored = engine.evaluate_eviction(
                tokenized.tokens,
                tokenized.spans,
                budget=budget,
                pinned=case.pinned,
                rankings=_rankings(case.segments),
            )

            outcomes: dict[str, bool] = {}
            for name, result in scored.items():
                pruned = [t for sid in sorted(result["kept"],
                                              key=lambda s: tokenized.spans[s][0])
                          for t in tokenized.tokens[slice(*tokenized.spans[sid])]]
                verdict = engine.assess_edit(tokenized.tokens, pruned, gold)
                outcomes[name] = verdict["gold_after"]["rank"] == 1
                rows.append({"seed": seed, "budget_frac": frac, "policy": name,
                             "n_tokens": result["n_tokens"],
                             "kl": result["kl"],
                             "gold_rank": verdict["gold_after"]["rank"],
                             "full_rank": verdict["gold_before"]["rank"],
                             "correct": outcomes[name]})

            # Budgets must be comparable or the table compares sizes, not
            # rankings. Assert it rather than trusting the fill.
            held = {r["n_tokens"] for r in rows
                    if r["seed"] == seed and r["budget_frac"] == frac}
            assert max(held) - min(held) <= budget * 0.1, (
                f"policies held {sorted(held)} tokens at budget {budget}; "
                "that is a comparison of sizes, not of rankings")

            require_headroom(outcomes)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=TEST_MODEL)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--out", default="experiments/results/provenance_bench.json",
                    help="per-cell rows, kept so a later entry can re-verify them")
    args = ap.parse_args()

    model, tokenizer = load_model(args.model)
    engine = Engine(model, tokenizer)

    rows = run(engine, tokenizer, seeds=range(args.seeds))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))

    print("signal firing (share of segments non-zero, over all seeds):")
    for name, fn in SIGNALS.items():
        vals = [fn(c.segments, s.id) != 0
                for c in (generate(seed=i) for i in range(args.seeds))
                for s in c.segments]
        print(f"  {name:<14} {sum(vals) / len(vals):>6.1%}")
    print()

    by_policy: dict[str, list[bool]] = {}
    for r in rows:
        by_policy.setdefault(r["policy"], []).append(r["correct"])

    rs = [collinearity(generate(seed=s).segments) for s in range(args.seeds)]
    mean_r = sum(rs) / len(rs) if rs else 0.0
    print(f"referenced-vs-depth collinearity: r = {mean_r:+.2f}")
    if abs(mean_r) > 0.7:
        print("  WARNING: those two signals are nearly one signal here.")
        print("  Read provenance_no_position, not provenance, as the result:")
        print("  a composite win at this r may be recency under another name.")
    print()

    print(f"{'policy':<28} {'correct':>8}  {'n':>4}")
    for policy, hits in sorted(by_policy.items(),
                               key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"{policy:<28} {sum(hits) / len(hits):>7.1%}  {len(hits):>4}")


if __name__ == "__main__":
    main()
