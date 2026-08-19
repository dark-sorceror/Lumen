"""The controllable inference loop: a hand-owned token loop over direct model calls.

Deliberately NOT mlx_lm.stream_generate — owning this loop is what makes
logits processors, control-queue interrupts, taps, and cache
surgery possible. The parity harness (experiments/parity.py) proves equivalence at T=0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from workbench.context.manager import _common_prefix_len
from workbench.engine.taps import top_k_logprobs


@dataclass
class GenParams:
    max_tokens: int = 512
    temperature: float = 0.0
    top_k_logprobs: int = 0


@dataclass
class TokenEvent:
    token_id: int
    text: str
    top_logprobs: dict[int, float] = field(default_factory=dict)
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

    # -- shared loop -----------------------------------------------------

    def _forward(self, tokens: list[int], cache) -> mx.array:
        """Run the model over `tokens`, return last-position logits [1, V]."""
        logits = self.model(mx.array(tokens)[None], cache=cache)
        return logits[:, -1, :]

    def _prefill(self, tokens: list[int], cache) -> mx.array:
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
        return self._forward(tokens[pos:], cache)

    def _sample(self, logits: mx.array, params: GenParams) -> int:
        if params.temperature == 0.0:
            # Match mlx-lm: apply argmax to log-probabilities
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            return int(mx.argmax(logprobs, axis=-1).item())
        scaled = logits / params.temperature
        return int(mx.random.categorical(scaled).item())

    def _run(self, tokens_to_prefill: list[int], params: GenParams, control, track: bool) -> Iterator[TokenEvent]:
        """The shared loop body: prefills `tokens_to_prefill`
        into `self._cache`, then decodes. When `track` is True, both the
        prefilled and generated tokens are recorded into `self._cached_tokens`
        so a later `generate_with_cache` call can reuse this cache state."""
        cache = self._cache
        detok = self.tokenizer.detokenizer
        detok.reset()

        try:
            logits = self._prefill(tokens_to_prefill, cache)  # prefill
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
            next_logits = self._forward([token], cache)
            if track:
                self._cached_tokens.append(token)

            yield TokenEvent(
                token_id=token,
                text=text,
                top_logprobs=top,
                finish_reason="length" if is_last else None,
            )
            logits = next_logits
