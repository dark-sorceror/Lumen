"""Run one of the page's pure modules under node, for a test of its logic.

The frontend has no test runner, so a test transpiles a module that imports
nothing (frontend/src/lib/...) with the frontend's own TypeScript, calls it,
and reads back what the call returned as JSON. Skipped where node or the
frontend's TypeScript is not installed."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
NODE = shutil.which("node")
TYPESCRIPT = FRONTEND / "node_modules" / "typescript"

needs_node = pytest.mark.skipif(NODE is None or not TYPESCRIPT.is_dir(),
                                reason="needs node and the frontend's TypeScript")

# argv: the TypeScript package, the module, the function body, the source
# root (frontend/src). stdin: the body's `input`, as JSON. The body is given
# `m` (the module's exports). A module's own imports ("./x", "../x", "@/x")
# are loaded the same way, from .ts, .tsx or an index.ts; anything else goes
# to node's require.
_RUN = r"""
const fs = require("fs");
const path = require("path");
const ts = require(process.argv[1]);
const SRC = process.argv[4];
const cache = new Map();
function resolve(spec, from) {
  const base = spec.startsWith("@/") ? path.join(SRC, spec.slice(2)) : path.resolve(path.dirname(from), spec);
  for (const tail of [".ts", ".tsx", "/index.ts"]) {
    if (fs.existsSync(base + tail)) return base + tail;
  }
  throw new Error("cannot resolve " + spec + " from " + from);
}
function load(file) {
  if (cache.has(file)) return cache.get(file).exports;
  const source = fs.readFileSync(file, "utf8");
  const js = ts.transpileModule(source, { compilerOptions: {
    module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
  const mod = { exports: {} };
  cache.set(file, mod);
  const own = (spec) => spec.startsWith("./") || spec.startsWith("../") || spec.startsWith("@/");
  new Function("module", "exports", "require", js)(mod, mod.exports,
    (spec) => (own(spec) ? load(resolve(spec, file)) : require(spec)));
  return mod.exports;
}
const input = JSON.parse(fs.readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify(new Function("m", "input", process.argv[3])(load(process.argv[2]), input)));
"""


def run_ts(module: str, body: str, data: Any = None, timeout: float = 60) -> Any:
    """What `body` returns, given `m` (the exports of frontend/src/`module`)
    and `input` (`data`). A run past `timeout` seconds fails the test."""
    try:
        done = subprocess.run([NODE, "-e", _RUN, str(TYPESCRIPT), str(FRONTEND / "src" / module), body,
                               str(FRONTEND / "src")],
                              input=json.dumps(data), capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        pytest.fail(f"{module} ran past {timeout} s")
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)
