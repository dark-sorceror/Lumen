"""Parity check: our hand-owned loop must be token-identical to stock mlx_lm at T=0.

Run: uv run python -m experiments.parity            (tiny test model)
     uv run python -m experiments.parity --model mlx-community/Qwen3-8B-8bit
"""
import argparse
import json
from pathlib import Path

from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from workbench.engine.engine import Engine, GenParams
from workbench.engine.loader import TEST_MODEL, load_model

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Q: What is 17 + 25?\nA:",
    "Once upon a time, in a small village by the sea,",
]
MAX_TOKENS = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=TEST_MODEL)
    args = ap.parse_args()

    model, tokenizer = load_model(args.model)
    engine = Engine(model, tokenizer)
    results = []
    for prompt in PROMPTS:
        tokens = tokenizer.encode(prompt)
        ours = [e.token_id for e in engine.generate(tokens, GenParams(max_tokens=MAX_TOKENS))]
        theirs = [r.token for r in stream_generate(
            model, tokenizer, tokens, max_tokens=MAX_TOKENS,
            sampler=make_sampler(temp=0.0))]
        ok = ours == theirs
        results.append({"prompt": prompt, "match": ok,
                        "ours": ours, "theirs": theirs})
        print(f"{'PASS' if ok else 'FAIL'}  {prompt[:40]!r}")

    Path("results").mkdir(exist_ok=True)
    Path("results/parity.json").write_text(json.dumps(
        {"model": args.model, "results": results}, indent=2))
    if not all(r["match"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
