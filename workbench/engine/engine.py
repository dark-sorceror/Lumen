"""The controllable inference loop: a hand-owned token loop over direct model calls.

Deliberately NOT mlx_lm.stream_generate — owning this loop is what makes
logits processors, control-queue interrupts, taps, and cache
surgery possible. The parity harness (experiments/parity.py) proves equivalence at T=0.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Callable, Iterator

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from workbench.context.manager import _common_prefix_len
from workbench.context.eviction import (keep_by_attention, keep_by_recency,
                                        rebuild_tokens)
from workbench.engine.taps import (apply_steering, attention_mass_by_segment,
                                   capture_attention, capture_layer_outputs,
                                   layer_count, logit_lens, top_k_logprobs)


@dataclass
class GenParams:
    max_tokens: int = 512
    temperature: float = 0.0
    top_k_logprobs: int = 0
    # Transformer block indices whose output hidden state should be captured.
    # Empty (the default) leaves the forward pass completely untouched.
    hidden_layers: tuple[int, ...] = ()
    # layer index -> (direction, strength), added to that block's output for
    # every forward pass of this generation.
    steering: dict = field(default_factory=dict)


@dataclass
class TokenEvent:
    token_id: int
    text: str
    top_logprobs: dict[int, float] = field(default_factory=dict)
    # layer index -> hidden state [d_model] leaving that block for this token.
    hidden: dict[int, "mx.array"] = field(default_factory=dict)
    finish_reason: str | None = None


class Engine:
    # Mirrors mlx-lm's `prefill_step_size` default: chunk long prompts so we
    # never materialize [1, n, vocab] logits for the whole prompt at once —
    # only the final chunk's last-position logits are kept.
    _PREFILL_CHUNK_SIZE = 2048

    def __init__(self, model, tokenizer, logits_processors=None):
        self.model = model
        self.tokenizer = tokenizer
        self.logits_processors: list[Callable] = logits_processors or []
        self._cache = None
        self._cached_tokens: list[int] = []
        # Set synchronously at the top of generate_with_cache, before any
        # token is yielded: how many tokens of the incoming prompt were
        # already present in the KV cache (the common-prefix length between
        # what was cached and what's being asked for now). The server reads
        # this right after generation starts to report prompt/cache stats
        # over the wire (workbench/server/app.py's gen_stats message).
        self.last_cache_reuse: int = 0
        # Hidden states captured by the most recent _forward, if requested.
        self._last_hidden: dict[int, mx.array] = {}

    # -- cache-owning API -----------------

    def start_session(self) -> None:
        """Fresh empty cache; forgets any prior session state."""
        self._cache = make_prompt_cache(self.model)
        self._cached_tokens = []

    def trim_to(self, n_tokens: int) -> None:
        """Drop cache content after position `n_tokens`. Used when an edit
        arrives with no generation (generate_with_cache does this itself too,
        via the common-prefix check, before prefilling the new suffix)."""
        excess = len(self._cached_tokens) - n_tokens
        if excess > 0:
            trimmed = trim_prompt_cache(self._cache, excess)
            # trim_prompt_cache returns the number of tokens ACTUALLY
            # trimmed -- 0 for non-trimmable cache types (e.g. caches that
            # don't support `.trim()`). An empty cache (`[]`, as used by
            # cache-blind fakes and any model with no real KV state) also
            # trivially returns 0 regardless of `excess`; that's fine and
            # not a desync, since there's no real state to fall out of
            # sync in the first place. Only raise when the cache holds
            # real state but didn't shrink by the amount we asked for --
            # truncating `_cached_tokens` anyway would silently desync it
            # from what the cache actually contains.
            if self._cache and trimmed != excess:
                raise RuntimeError(
                    f"trim_to: cache only trimmed {trimmed}/{excess} tokens "
                    "(non-trimmable cache type?) -- refusing to desync "
                    "_cached_tokens from the actual cache content"
                )
            self._cached_tokens = self._cached_tokens[:n_tokens]

    def _recover_from_prefill_failure(self) -> None:
        """A prefill that raises partway through a chunked run (see
        `_prefill`) may leave the cache holding KV state for chunks that
        were never recorded into `_cached_tokens` -- the cache ends up
        AHEAD of the tracked token list (the mirror image of the
        early-generator-abandonment desync, where the list can end up
        ahead of the cache). Only called for tracked (`track=True`)
        sessions, since untracked one-shot calls (`generate`) never
        persist `self._cache` past their own `finally` block anyway.

        Best case: the cache exposes a real offset, so we trim it back
        down to exactly `len(self._cached_tokens)` -- the session is
        restored to its pre-call state, no data lost beyond the failed
        call itself.

        If we can't determine or restore that exact state (a cache type
        with no usable offset, an empty/no-op fake cache, or a trim that
        doesn't remove what we asked for), we discard the session outright
        via `start_session()` rather than risk silently resuming from a
        desynced cache -- the next `generate_with_cache` call simply
        re-prefills from scratch instead of reusing anything suspect."""
        expected = len(self._cached_tokens)
        try:
            offset = self._cache[0].offset
        except (IndexError, AttributeError, TypeError):
            self.start_session()
            return

        excess = offset - expected
        if excess <= 0:
            return

        try:
            trimmed = trim_prompt_cache(self._cache, excess)
        except Exception:
            trimmed = 0

        if trimmed != excess:
            self.start_session()

    def generate_with_cache(self, full_tokens, params, control=None) -> Iterator[TokenEvent]:
        """Prefills only the un-cached suffix of `full_tokens` (tracking what
        the cache has seen), generates, and extends the record with the
        newly generated tokens."""
        if self._cache is None:
            self.start_session()

        full_tokens = list(full_tokens)
        keep = _common_prefix_len(self._cached_tokens, full_tokens)
        # Set before any trimming/prefilling/yielding, so a caller reading
        # this attribute as soon as generation has *started* (e.g. right
        # after the first token/event is observed) sees an accurate value.
        self.last_cache_reuse = keep
        self.trim_to(keep)
        suffix = full_tokens[keep:]

        if not suffix:
            if not self._cached_tokens:
                raise ValueError(
                    "generate_with_cache: nothing to prefill and no cached "
                    "context to resume decoding from"
                )
            # full_tokens is an exact (possibly-shorter) prefix of what's
            # already cached: there is nothing new to feed the model, but we
            # still need fresh next-token logits to keep decoding. Re-derive
            # them by reprocessing the last cached token against a cache
            # trimmed back by one -- deterministic, so it reconstructs a
            # bit-identical cache state while handing us the logits we need.
            # Routed through trim_to so this shares the same
            # trim-return-value consistency check as any other trim.
            self.trim_to(len(self._cached_tokens) - 1)
            suffix = full_tokens[-1:]

        yield from self._run(suffix, params, control, track=True)

    def generate(self, prompt_tokens, params, control=None) -> Iterator[TokenEvent]:
        """Stateless one-shot generation (fresh throwaway cache). the parity harness uses this."""
        saved = (self._cache, self._cached_tokens)
        self._cache, self._cached_tokens = make_prompt_cache(self.model), []
        try:
            yield from self._run(list(prompt_tokens), params, control, track=False)
        finally:
            self._cache, self._cached_tokens = saved

    def _measure(self, tokens: list[int], layers: tuple, top_k: int,
                 steering: dict | None = None) -> dict:
        """One forward pass with taps, optionally under an intervention.

        Runs on a THROWAWAY cache: measuring must never disturb the session's
        KV state, or inspecting would silently change what a later edit costs."""
        cache = make_prompt_cache(self.model)
        with ExitStack() as stack:
            if steering:
                stack.enter_context(apply_steering(self.model, steering))
            attention = stack.enter_context(capture_attention(self.model, layers))
            hidden = stack.enter_context(capture_layer_outputs(self.model, layers))
            self.model(mx.array(tokens)[None], cache=cache)
        lens = logit_lens(self.model, hidden, k=top_k)
        return {
            layer: {"hidden": hidden.get(layer),
                    "lens": lens.get(layer, {}),
                    "attention": attention.get(layer)}
            for layer in layers
        }

    def inspect(self, tokens: list[int], layers, top_k: int = 5) -> dict:
        """Measure one forward pass over `tokens` without generating.

        Returns, per requested layer, what that depth would predict (`lens`)
        and where its attention went (`attention`, a row over key positions).
        The raw hidden state is deliberately dropped -- it is d_model floats
        per layer and no consumer of this call needs it."""
        layers = tuple(layers)
        measured = self._measure(tokens, layers, top_k)
        return {
            layer: {"lens": m["lens"], "attention": m["attention"]}
            for layer, m in measured.items()
        }

    def compare(self, tokens: list[int], layers, steering: dict,
                top_k: int = 5) -> dict:
        """Run the same prompt with and without an intervention, and report how
        far the model moved at each depth.

        `l2` and `cosine` are activation drift: how far, and in what direction,
        the residual stream shifted. `top_token_changed` says whether that was
        enough to alter what the layer would actually predict -- drift that
        never reaches the readout is not a behavioural change."""
        layers = tuple(layers)
        base = self._measure(tokens, layers, top_k)
        other = self._measure(tokens, layers, top_k, steering=steering)

        out = {}
        for layer in layers:
            # Promote to float32 before any reduction: these arrive in the
            # model's compute precision, and accumulating a dot product there
            # can put a cosine slightly above 1 -- a number this must never
            # report.
            a = base[layer]["hidden"].astype(mx.float32)
            b = other[layer]["hidden"].astype(mx.float32)
            delta = b - a
            norm_a = float(mx.linalg.norm(a).item())
            norm_b = float(mx.linalg.norm(b).item())
            denom = norm_a * norm_b
            cosine = float((a * b).sum().item()) / denom if denom else 0.0
            cosine = max(-1.0, min(1.0, cosine))
            top_a = next(iter(base[layer]["lens"]), None)
            top_b = next(iter(other[layer]["lens"]), None)
            out[layer] = {
                "l2": float(mx.linalg.norm(delta).item()),
                "cosine": cosine,
                "relative": (float(mx.linalg.norm(delta).item()) / norm_a) if norm_a else 0.0,
                "top_token_changed": top_a != top_b,
                "lens_before": base[layer]["lens"],
                "lens_after": other[layer]["lens"],
            }
        return out

    def _mean_activation(self, prompts: list[list[int]], layer: int) -> mx.array:
        """Average hidden state leaving `layer` across a set of prompts."""
        total = None
        for tokens in prompts:
            cache = make_prompt_cache(self.model)
            with capture_layer_outputs(self.model, (layer,)) as captured:
                self.model(mx.array(tokens)[None], cache=cache)
            h = captured[layer]
            total = h if total is None else total + h
        if total is None:
            raise ValueError("need at least one prompt")
        return total / len(prompts)

    def steering_vector(self, positive: list[list[int]], negative: list[list[int]],
                        layer: int) -> mx.array:
        """A direction in activation space, built by contrast.

        mean(positive) - mean(negative) at `layer`: whatever the two prompt
        sets differ in, expressed as a vector that can be added back into the
        residual stream. Minutes to build and nothing is trained -- which is
        why it is the cheapest intervention worth measuring."""
        return (self._mean_activation(positive, layer)
                - self._mean_activation(negative, layer))

    def compare_over(self, prompts: list[list[int]], layers, steering: dict,
                     top_k: int = 5) -> dict:
        """Aggregate `compare` over a set of prompts.

        One prompt is an anecdote: whether a single prediction flipped says
        little, because it may have been a near-tie the intervention nudged by
        accident. Over a set, `flip_rate` becomes a rate and `std_relative`
        says whether the drift was consistent or driven by one outlier -- which
        is the difference between a demonstration and a measurement."""
        prompts = list(prompts)
        if not prompts:
            raise ValueError("need at least one prompt to compare over")
        layers = tuple(layers)
        per_prompt = [self.compare(p, layers, steering, top_k) for p in prompts]

        def _mean(xs):
            return sum(xs) / len(xs)

        def _std(xs):
            if len(xs) < 2:
                return 0.0
            m = _mean(xs)
            return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5

        out = {}
        for layer in layers:
            rel = [d[layer]["relative"] for d in per_prompt]
            l2 = [d[layer]["l2"] for d in per_prompt]
            cos = [d[layer]["cosine"] for d in per_prompt]
            flips = [1.0 if d[layer]["top_token_changed"] else 0.0 for d in per_prompt]
            out[layer] = {
                "n": len(per_prompt),
                "mean_l2": _mean(l2),
                "mean_relative": _mean(rel),
                "std_relative": _std(rel),
                "mean_cosine": _mean(cos),
                "flip_rate": _mean(flips),
            }
        return out

    def activation_norm(self, tokens: list[int], layer: int) -> float:
        """||h_l|| at the last position of `tokens`.

        The scale any intervention must be sized against. A dose quoted as a
        bare constant is meaningless, because this grows with depth."""
        measured = self._measure(tokens, (layer,), top_k=1)
        return float(mx.linalg.norm(measured[layer]["hidden"]).item())

    def sweep_injection(self, probes: list[list[int]], positive: list[list[int]],
                        negative: list[list[int]], layers, rho: float = 0.15,
                        top_k: int = 5) -> dict:
        """Ask where an intervention is worth injecting.

        For each candidate depth: derive a contrastive direction THERE, dose it
        at `rho * ||h_l||` measured at that same depth, and report what reached
        the readout -- drift and flip rate at the final block.

        Holding `rho` constant rather than `alpha` is the whole point. The
        stream's magnitude grows with depth, so a fixed alpha is a shrinking
        perturbation as you go deeper, and a sweep over it would measure the
        dose schedule rather than the depth."""
        probes = list(probes)
        if not probes:
            raise ValueError("need at least one probe prompt")
        final = layer_count(self.model) - 1
        out = {}
        for layer in tuple(layers):
            direction = self.steering_vector(positive, negative, layer=layer)
            norm = float(mx.linalg.norm(direction).item())
            if norm > 0:
                direction = direction / norm
            # Dose from the FIRST probe's stream at this depth; probes are drawn
            # from one distribution, so this is a stable scale rather than a
            # per-probe moving target.
            alpha = rho * self.activation_norm(probes[0], layer)
            agg = self.compare_over(probes, layers=(final,),
                                    steering={layer: (direction, alpha)}, top_k=top_k)
            out[layer] = {
                "rho": rho,
                "alpha": alpha,
                "n": agg[final]["n"],
                "final_relative": agg[final]["mean_relative"],
                "final_std": agg[final]["std_relative"],
                "final_flip_rate": agg[final]["flip_rate"],
            }
        return out

    def next_logprobs(self, tokens: list[int]) -> mx.array:
        """Full next-token log-distribution for `tokens`, on a throwaway cache."""
        cache = make_prompt_cache(self.model)
        logits = self._forward(list(tokens), cache).astype(mx.float32)
        return (logits - mx.logsumexp(logits, axis=-1, keepdims=True))[0]

    def divergence(self, tokens_a: list[int], tokens_b: list[int]) -> dict:
        """How much the prediction moves between two contexts.

        `kl` is KL(A || B) in nats -- how surprised context A's distribution is
        by context B's. Deliberately asymmetric, and the argument order is the
        useful one: pass the FULL context first and the pruned one second, so
        the number answers "what did dropping that cost", not the reverse.

        `top1_agrees` is the coarse behavioural question underneath it. A
        policy can move probability mass around considerably without changing
        what the model actually says next, and only the second one is a
        behaviour change."""
        # clamp inside _divergence_from: exact equality can land at -0.0
        return self._divergence_from(self.next_logprobs(tokens_a),
                                     self.next_logprobs(tokens_b))

    def attention_mass(
        self,
        tokens: list[int],
        spans: dict[str, tuple[int, int]],
        layers: list[int] | None = None,
        exclude_sink: bool = False,
    ) -> dict[str, float]:
        """Share of the final position's attention landing on each segment,
        averaged over `layers`.

        `exclude_sink` zeroes position 0 and renormalises. Real models park a
        large, roughly content-independent share of attention on the first
        token -- an attention sink, which is a pressure valve for heads with
        nothing to attend to, not evidence that token 0 matters. Left in, it
        inflates whichever segment happens to start the context."""
        if layers is None:
            n = layer_count(self.model)
            layers = [n // 4, n // 2, (3 * n) // 4, n - 1]
        cache = make_prompt_cache(self.model)
        with capture_attention(self.model, layers) as captured:
            self.model(mx.array(tokens)[None], cache=cache)

        ids = list(spans)
        span_list = [spans[sid] for sid in ids]
        totals = {sid: 0.0 for sid in ids}
        measured = [captured[i] for i in layers if i in captured]
        for weights in measured:
            if exclude_sink:
                weights = mx.concatenate([mx.zeros((1,)), weights[1:]])
                weights = weights / weights.sum()
            for sid, m in zip(ids, attention_mass_by_segment(weights, span_list)):
                totals[sid] += m / len(measured)
        return totals

    def evaluate_eviction(
        self,
        tokens: list[int],
        spans: dict[str, tuple[int, int]],
        budget: int,
        layers: list[int] | None = None,
        pinned: tuple[str, ...] = (),
    ) -> dict[str, dict]:
        """Score each eviction policy by what its drop costs at the readout.

        Every policy is compared against the SAME full-context distribution, so
        the numbers are commensurable. `pinned` segments are always kept and
        still charged against the budget -- typically the live query, which no
        sane policy evicts and whose removal would swamp every other effect."""
        raw = self.attention_mass(tokens, spans, layers)
        nosink = self.attention_mass(tokens, spans, layers, exclude_sink=True)

        # Pinned segments are RESERVED off the top: removed from the ranking and
        # their tokens subtracted from the budget, rather than unioned in after
        # the fill. Unioning afterwards lets the result exceed the budget, and
        # then the policies are being compared while holding different numbers
        # of tokens -- which is not a comparison of their rankings at all.
        pinned = set(pinned) & set(spans)
        reserved = sum(spans[sid][1] - spans[sid][0] for sid in pinned)
        contested = {sid: sp for sid, sp in spans.items() if sid not in pinned}
        room = max(0, budget - reserved)

        candidates = {
            "recency": keep_by_recency(contested, room),
            "attention": keep_by_attention(contested, raw, room),
            "attention_nosink": keep_by_attention(contested, nosink, room),
        }
        out: dict[str, dict] = {}
        for name, kept in candidates.items():
            kept = kept | pinned
            pruned = rebuild_tokens(tokens, spans, kept)
            scored = self.divergence(tokens, pruned)
            out[name] = {
                "kept": kept,
                "n_tokens": len(pruned),
                "kl": scored["kl"],
                "top1_agrees": scored["top1_agrees"],
                "mass": {"raw": raw, "nosink": nosink}[
                    "nosink" if name.endswith("nosink") else "raw"
                ],
            }
        return out

    def gold_score(self, tokens: list[int], gold: int) -> dict:
        """Score a context against a KNOWN-correct answer token.

        The complement to `divergence`. Where KL asks "how far did this move
        from the full context", this asks "did it get the answer right" -- and
        those are different questions the moment the full context is itself
        wrong."""
        return self._gold_from(self.next_logprobs(tokens), gold)

    @staticmethod
    def _gold_from(lp: mx.array, gold: int) -> dict:
        return {"logprob": float(lp[gold].item()),
                "rank": int((lp > lp[gold]).sum().item()) + 1}

    @staticmethod
    def _divergence_from(lp_a: mx.array, lp_b: mx.array) -> dict:
        p_a = mx.exp(lp_a)
        kl = float((p_a * (lp_a - lp_b)).sum().item())
        return {"kl": max(0.0, kl),
                "top1_agrees": int(mx.argmax(lp_a).item()) == int(mx.argmax(lp_b).item())}

    def assess_edit(self, full: list[int], pruned: list[int], gold: int) -> dict:
        """Judge an edit on two coordinates, because one is not enough.

        `KL(full || pruned)` measures distance from the unedited model, and
        treats the full context as ground truth. When the full context is
        POISONED -- a segment in it drives the model to the wrong answer -- the
        one edit that repairs the answer is also the one that moves furthest
        from the full distribution, so KL ranks the repair as the most damaging
        edit available (measured at 7.5x the cost of deleting the true answer;
        see docs). KL alone is therefore anti-correlated with quality in exactly
        the case where editing has the most to offer.

        So the verdict comes from the gold score, which tracks the thing we
        actually care about, and KL is reported alongside as the magnitude of
        distributional movement. A large `kl` together with `verdict ==
        "repaired"` is the signature of a poisoned context -- and neither number
        can report that on its own."""
        # Both coordinates are functions of the SAME two logprob vectors, so
        # they are computed once each. Calling gold_score twice and then
        # divergence would run four forward passes for two distributions.
        lp_full = self.next_logprobs(full)
        lp_pruned = self.next_logprobs(pruned)
        before = self._gold_from(lp_full, gold)
        after = self._gold_from(lp_pruned, gold)
        delta = after["logprob"] - before["logprob"]
        moved = self._divergence_from(lp_full, lp_pruned)

        # The verdict keys on the GOLD ANSWER'S STANDING, not on the top-1.
        # Keying it on the top-1 was wrong in the case that matters most: on a
        # poisoned context the top-1 is already the wrong token, so an edit can
        # destroy the answer completely without disturbing it. Measured on the
        # real lure context, deleting the true answer moved gold from rank 5 to
        # rank 6732 (-11.5 nats) while the top-1 sat unchanged -- a top-1 rule
        # calls that "preserved".
        #
        # The verdict is RANK ONLY. An unchanged rank is "preserved" however the
        # probability wobbles: across 42 clean-context cells, dropping
        # distractors nudged the gold logprob by +0.005 to +0.34 nats with the
        # answer at rank 1 throughout, and letting the delta decide labelled 20
        # of those 42 "repaired" when nothing had been repaired. It also removes
        # the epsilon that decision needed -- a constant with no measurement
        # behind it. The magnitudes travel in `kl` and `gold_delta` for anyone
        # who needs to rank edits within a verdict.
        if after["rank"] < before["rank"]:
            verdict = "repaired"
        elif after["rank"] > before["rank"]:
            verdict = "damaged"
        else:
            verdict = "preserved"

        return {
            "kl": moved["kl"],
            "top1_agrees": moved["top1_agrees"],
            "gold_before": before,
            "gold_after": after,
            "gold_delta": delta,
            "verdict": verdict,
        }

    # -- shared loop -----------------------------------------------------

    def _forward(self, tokens: list[int], cache, hidden_layers: tuple[int, ...] = ()) -> mx.array:
        """Run the model over `tokens`, return last-position logits [1, V].

        When `hidden_layers` is non-empty the hidden state leaving each of those
        blocks is recorded into `self._last_hidden` for this pass."""
        if not hidden_layers:
            self._last_hidden = {}
            logits = self.model(mx.array(tokens)[None], cache=cache)
            return logits[:, -1, :]
        with capture_layer_outputs(self.model, hidden_layers) as captured:
            logits = self.model(mx.array(tokens)[None], cache=cache)
            self._last_hidden = dict(captured)
        return logits[:, -1, :]

    def _prefill(self, tokens: list[int], cache, hidden_layers: tuple[int, ...] = ()) -> mx.array:
        """Prefill `tokens` into `cache`, returning next-token logits.

        Mirrors mlx-lm's generate_step exactly: all but the LAST token are
        processed in chunks whose logits are discarded (never sliced/kept),
        with cache state eagerly evaluated between chunks so peak memory
        stays bounded by one chunk; the final token is then run as a
        single-token step whose logits seed the decode loop.

        The split point is not just a memory choice — it is required for
        T=0 token parity with mlx-lm. On quantized models the batched
        prompt matmul and the single-token decode matmul use different
        kernels that round differently, so including the last prompt token
        in the batch produces epsilon-different logits (and KV state for
        that position) than mlx-lm computes, which flips argmax at
        near-ties a few tokens downstream."""
        n = len(tokens)
        if n == 0:
            return None
        pos = 0
        while n - pos > 1:
            end = min(pos + self._PREFILL_CHUNK_SIZE, n - 1)
            chunk = tokens[pos:end]
            self.model(mx.array(chunk)[None], cache=cache)
            mx.eval([c.state for c in cache])
            mx.clear_cache()
            pos = end
        return self._forward(tokens[pos:], cache, hidden_layers)

    def _sample(self, logits: mx.array, params: GenParams) -> int:
        if params.temperature == 0.0:
            # Match mlx-lm: apply argmax to log-probabilities
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            return int(mx.argmax(logprobs, axis=-1).item())
        scaled = logits / params.temperature
        return int(mx.random.categorical(scaled).item())

    def _run(self, tokens_to_prefill: list[int], params: GenParams, control, track: bool) -> Iterator[TokenEvent]:
        """Run the loop, with any steering directions installed for its whole
        duration -- prefill and every decode step alike."""
        if not params.steering:
            yield from self._run_loop(tokens_to_prefill, params, control, track)
            return
        with apply_steering(self.model, params.steering):
            yield from self._run_loop(tokens_to_prefill, params, control, track)

    def _run_loop(self, tokens_to_prefill: list[int], params: GenParams, control, track: bool) -> Iterator[TokenEvent]:
        """The shared loop body: prefills `tokens_to_prefill`
        into `self._cache`, then decodes. When `track` is True, both the
        prefilled and generated tokens are recorded into `self._cached_tokens`
        so a later `generate_with_cache` call can reuse this cache state."""
        cache = self._cache
        detok = self.tokenizer.detokenizer
        detok.reset()

        try:
            logits = self._prefill(tokens_to_prefill, cache, params.hidden_layers)  # prefill
        except Exception:
            # A chunked prefill (see _prefill) may have already fed some
            # chunks into the cache before raising -- the cache can be
            # AHEAD of `_cached_tokens` (which we haven't extended yet).
            # Repair or invalidate the session so the failure can't cause
            # a later generate_with_cache call to silently desync.
            if track:
                self._recover_from_prefill_failure()
            raise
        if track:
            self._cached_tokens.extend(tokens_to_prefill)

        generated: list[int] = []

        for i in range(params.max_tokens):
            if control is not None:
                verdict = control.checkpoint()  # blocks while paused
                if verdict == "abort":
                    # No token has been sampled this iteration, so the cache
                    # and `_cached_tokens` are already consistent -- nothing
                    # to feed or roll back. Note: any text the detokenizer
                    # was withholding pending more bytes (an unresolved
                    # multibyte/emoji tail from a prior token) is
                    # intentionally dropped here rather than flushed --
                    # same as EOS's text="" treatment, we don't finalize()
                    # on abort.
                    yield TokenEvent(-1, "", finish_reason="aborted")
                    return

            for proc in self.logits_processors:
                logits = proc(generated, logits)

            token = self._sample(logits, params)
            generated.append(token)

            # Snapshot now: the trailing _forward for the NEXT token overwrites it.
            hidden = self._last_hidden
            top = {}
            if params.top_k_logprobs > 0:
                top = top_k_logprobs(logits, params.top_k_logprobs)

            is_eos = token in self.tokenizer.eos_token_ids
            if is_eos:
                # Do not feed the EOS token to the detokenizer: its decoded text
                # (e.g. "<|im_end|>") must never leak into the UI or chat history,
                # where apply_chat_template would otherwise double up terminators.
                # EOS is also never forwarded through the cache or appended to
                # `_cached_tokens`: it's a terminal marker, not resumable context.
                yield TokenEvent(
                    token_id=token,
                    text="",
                    top_logprobs=top,
                    hidden=hidden,
                    finish_reason="stop",
                )
                return

            detok.add_token(token)
            is_last = i == params.max_tokens - 1
            text = detok.last_segment
            if is_last:
                # Flush any text the detokenizer withheld pending more bytes
                # (e.g. a generation cut off mid multibyte/emoji sequence) so
                # the terminal event doesn't silently drop trailing characters.
                detok.finalize()
                text += detok.last_segment

            # Feed this token's KV into the cache BEFORE yielding (not
            # after). This keeps `len(self._cached_tokens) == tokens
            # actually present in the cache` true at every yield point,
            # even if the consumer never resumes the generator past this
            # yield (break on finish_reason, itertools.islice, task
            # cancellation, ...) -- there is no longer a window where a
            # token is recorded as cached but its KV was never fed. The
            # resulting logits become the next iteration's `logits`,
            # exactly as before: for any consumer that fully drains the
            # generator this is a pure reorder, not a behavior change --
            # the same forward calls happen in the same sequence.
            next_logits = self._forward([token], cache, params.hidden_layers)
            if track:
                self._cached_tokens.append(token)

            yield TokenEvent(
                token_id=token,
                text=text,
                top_logprobs=top,
                hidden=hidden,
                finish_reason="length" if is_last else None,
            )
            logits = next_logits
