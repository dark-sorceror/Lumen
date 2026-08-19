import mlx.core as mx

from workbench.engine.engine import Engine, GenParams


def collect(engine, prompt, params):
    return list(engine.generate(prompt, params))


def test_greedy_generation_follows_fake_model(fake_model, fake_tokenizer):
    engine = Engine(fake_model, fake_tokenizer)
    events = collect(engine, [1, 2, 3], GenParams(max_tokens=5))
    # FakeModel counts up from the last prompt token: 4, 5, 6, 7, 8
    assert [e.token_id for e in events] == [4, 5, 6, 7, 8]
    assert events[-1].finish_reason == "length"
    assert events[0].text == "<4>"


def test_stops_at_eos(fake_model, fake_tokenizer):
    engine = Engine(fake_model, fake_tokenizer)
    events = collect(engine, [248], GenParams(max_tokens=10))
    # 249, then 250 which is EOS -> included with finish_reason "stop"
    assert [e.token_id for e in events] == [249, 250]
    assert events[-1].finish_reason == "stop"
    assert events[-1].text == ""


def test_logits_processor_chain_is_applied(fake_model, fake_tokenizer):
    import mlx.core as mx

    def ban_even(generated, logits):
        mask = mx.array([[-1e9 if i % 2 == 0 else 0.0 for i in range(fake_model.vocab_size)]])
        return logits + mask

    engine = Engine(fake_model, fake_tokenizer, logits_processors=[ban_even])
    events = collect(engine, [1, 2, 3], GenParams(max_tokens=3))
    assert all(e.token_id % 2 == 1 for e in events)


def test_terminal_event_includes_finalized_detokenizer_remainder(fake_model, fake_tokenizer):
    """Amendment 3: on the terminal (finish_reason='length') event, the engine
    must call detok.finalize() and fold the flushed remainder into that
    event's text. FakeDetokenizer withholds a "~" marker until finalize()."""
    engine = Engine(fake_model, fake_tokenizer)
    events = collect(engine, [1, 2, 3], GenParams(max_tokens=3))
    assert events[-1].finish_reason == "length"
    assert events[-1].text == "<6>~"          # normal segment + flushed remainder
    # Non-terminal events are unaffected: no flush happens mid-stream.
    assert events[0].text == "<4>"
    assert events[1].text == "<5>"


def test_eos_terminal_event_still_has_empty_text(fake_model, fake_tokenizer):
    """Amendment/requirement 4: EOS keeps text="" and is never fed to the
    detokenizer, even though it is also a terminal event."""
    events = collect(engine=Engine(fake_model, fake_tokenizer), prompt=[248],
                      params=GenParams(max_tokens=10))
    assert events[-1].token_id == 250
    assert events[-1].finish_reason == "stop"
    assert events[-1].text == ""


def test_generate_with_cache_only_prefills_suffix(fake_tokenizer):
    import mlx.core as mx

    class CountingModel:
        """FakeModel that records how many positions each call processes."""
        vocab_size = 16
        def __init__(self):
            self.calls = []
        def __call__(self, inputs, cache=None):
            self.calls.append(inputs.shape[1])
            last = int(inputs[0, -1].item())
            row = [0.0] * 16
            row[(last + 1) % 16] = 10.0
            return mx.broadcast_to(mx.array(row), (1, inputs.shape[1], 16))
        def make_cache(self):
            return []

    from workbench.engine.engine import Engine, GenParams
    model = CountingModel()
    engine = Engine(model, fake_tokenizer)
    engine.start_session()

    list(engine.generate_with_cache([1, 2, 3], GenParams(max_tokens=2)))
    # Prefill mirrors mlx-lm: first n-1 tokens chunked, last token as a
    # single-token step; then one decode call per generated token. What
    # matters here is the TOTAL positions processed: exactly the 3-token
    # prompt plus the 2 generated tokens -- nothing reprocessed.
    assert list(model.calls) == [2, 1, 1, 1]
    assert sum(model.calls) == 3 + 2

    model.calls.clear()
    # same history + generated tokens [4, 5] + two new tokens appended:
    # ONLY the new suffix [8, 9] is prefilled (as [1]+[1]), never the
    # already-cached [1, 2, 3, 4, 5].
    list(engine.generate_with_cache([1, 2, 3, 4, 5, 8, 9], GenParams(max_tokens=1)))
    assert list(model.calls) == [1, 1, 1]
    assert sum(model.calls) == 2 + 1


def test_generate_with_cache_exposes_last_cache_reuse(fake_tokenizer):
    """last_cache_reuse must reflect exactly how many tokens of the
    incoming prompt were already present in the KV cache (the common-prefix
    length), updated fresh on every generate_with_cache call -- this is what
    the server reports to the UI as 'cached tokens' in the gen_stats
    message."""
    class CountingModel:
        vocab_size = 16

        def __call__(self, inputs, cache=None):
            last = int(inputs[0, -1].item())
            row = [0.0] * 16
            row[(last + 1) % 16] = 10.0
            return mx.broadcast_to(mx.array(row), (1, inputs.shape[1], 16))

        def make_cache(self):
            return []

    from workbench.engine.engine import Engine, GenParams
    engine = Engine(CountingModel(), fake_tokenizer)
    engine.start_session()

    # A brand-new session: nothing cached yet, so nothing can be reused.
    list(engine.generate_with_cache([1, 2, 3], GenParams(max_tokens=2)))
    assert engine.last_cache_reuse == 0

    # Second call shares the [1, 2, 3, 4, 5] prefix with what's now cached
    # (prompt [1,2,3] + generated [4,5]) -- that whole 5-token prefix is
    # reused; only the new suffix [8, 9] is fresh.
    list(engine.generate_with_cache([1, 2, 3, 4, 5, 8, 9], GenParams(max_tokens=1)))
    assert engine.last_cache_reuse == 5


def test_generate_with_cache_resumes_with_no_new_tokens(fake_model, fake_tokenizer):
    """Calling generate_with_cache again with the exact same full_tokens (no
    edit, no new turn) must still be able to keep generating from the cache
    tip -- there's no new suffix to prefill, so the engine must re-derive
    fresh decode logits without corrupting the cache."""
    engine = Engine(fake_model, fake_tokenizer)
    engine.start_session()

    first = [e.token_id for e in engine.generate_with_cache([1, 2, 3], GenParams(max_tokens=2))]
    assert first == [4, 5]
    assert engine._cached_tokens == [1, 2, 3, 4, 5]

    more = [e.token_id for e in engine.generate_with_cache([1, 2, 3, 4, 5], GenParams(max_tokens=2))]
    assert more == [6, 7]
    assert engine._cached_tokens == [1, 2, 3, 4, 5, 6, 7]


# -- Finding 1: cache/list desync on early generator abandonment ----------


class FakeCacheState:
    """Tracks a real running offset (unlike the `[]` no-op caches above), so
    a fake model's predictions can depend on how many tokens were actually
    forwarded through the cache -- making a cache/`_cached_tokens` desync
    observable via generated token VALUES, not just call shapes."""

    def __init__(self):
        self.offset = 0

    @property
    def state(self):
        # Real mlx-lm caches expose `.state` (their KV arrays) for
        # `mx.eval([c.state for c in cache])` during chunked prefill; a
        # fake with no arrays has nothing to evaluate.
        return []

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(n, self.offset)
        self.offset -= n
        return n


class PositionCountingModel:
    """Predicts next = (total tokens ever forwarded through the cache) %
    vocab_size. Unlike the FakeModel/CountingModel above (which predict
    from the literal last input token, ignoring cache), this model's
    output depends on the cache's real offset -- so a phantom
    cache/_cached_tokens desync (a token counted as cached but never
    actually forwarded, or vice versa) changes the predicted token VALUE,
    making the bug observable end-to-end rather than just in call counts."""

    vocab_size = 32

    def __call__(self, inputs, cache=None):
        n = inputs.shape[1]
        state = cache[0]
        state.offset += n
        row = [0.0] * self.vocab_size
        row[state.offset % self.vocab_size] = 10.0
        return mx.broadcast_to(mx.array(row), (1, n, self.vocab_size))

    def make_cache(self):
        return [FakeCacheState()]


def test_early_generator_abandonment_keeps_cache_consistent(fake_tokenizer):
    """Finding 1: closing the generator mid-stream (break/close at a yield,
    e.g. on finish_reason, itertools.islice, or task cancellation) must not
    leave `_cached_tokens` ahead of what was actually fed into the cache.
    If it does, the next generate_with_cache silently reuses a phantom
    cached position and produces wrong continuations."""
    from workbench.engine.engine import Engine, GenParams

    base = [1, 2, 3]

    engine = Engine(PositionCountingModel(), fake_tokenizer)
    engine.start_session()
    gen = engine.generate_with_cache(base, GenParams(max_tokens=5))
    received = [next(gen).token_id for _ in range(2)]
    gen.close()  # abandon mid-stream, right after a yield

    # Continue the abandoned session using exactly what it believes is
    # cached (base + the 2 tokens we actually received).
    resumed = [
        e.token_id
        for e in engine.generate_with_cache(base + received, GenParams(max_tokens=3))
    ]

    # Reference: an uninterrupted, never-abandoned engine asked to produce
    # a correct continuation from the same full_tokens.
    ref_engine = Engine(PositionCountingModel(), fake_tokenizer)
    ref_engine.start_session()
    reference = [
        e.token_id
        for e in ref_engine.generate_with_cache(base + received, GenParams(max_tokens=3))
    ]

    assert resumed == reference


class RaisingAfterNCallsModel:
    """FakeModel that raises on its Nth call, simulating a prefill failure
    partway through a chunked prefill (earlier chunks already fed into the
    cache before the error)."""

    vocab_size = 16

    def __init__(self, fail_on_call):
        self.fail_on_call = fail_on_call
        self.calls = 0

    def __call__(self, inputs, cache=None):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated prefill failure")
        n = inputs.shape[1]
        last = int(inputs[0, -1].item())
        row = [0.0] * self.vocab_size
        row[(last + 1) % self.vocab_size] = 10.0
        return mx.broadcast_to(mx.array(row), (1, n, self.vocab_size))

    def make_cache(self):
        return []


def test_prefill_failure_does_not_silently_desync_cache(fake_tokenizer):
    """Finding 1: if `_prefill` raises partway through a chunked prefill,
    the engine must not end up with a cache silently ahead of
    `_cached_tokens`. Next use must either raise a clear session-invalid
    error, or recover to a state where the tracked tokens and what was fed
    to the cache agree -- never a silent desync."""
    import pytest as _pytest

    from workbench.engine.engine import Engine, GenParams

    model = RaisingAfterNCallsModel(fail_on_call=2)
    engine = Engine(model, fake_tokenizer)
    engine._PREFILL_CHUNK_SIZE = 2  # force multiple chunks: [1,2] then [3,4] then [5]
    engine.start_session()

    with _pytest.raises(RuntimeError, match="simulated prefill failure"):
        list(engine.generate_with_cache([1, 2, 3, 4, 5], GenParams(max_tokens=1)))

    try:
        list(engine.generate_with_cache([1, 2, 3, 4, 5], GenParams(max_tokens=2)))
    except Exception:
        pass  # a raised error on next use satisfies "no silent desync"
    else:
        # Recovered cleanly: tracked tokens must reflect exactly what was
        # (re-)fed, with no phantom entries.
        assert engine._cached_tokens[:5] == [1, 2, 3, 4, 5]


# -- Finding 3: trim_to ignoring trim_prompt_cache's return ----------------


class NonTrimmableCacheState:
    def is_trimmable(self):
        return False


def test_trim_to_raises_when_cache_cannot_actually_be_trimmed(fake_tokenizer):
    """Finding 3: trim_prompt_cache returns the number of tokens ACTUALLY
    trimmed (0 for non-trimmable cache types). trim_to must not blindly
    truncate `_cached_tokens` when the cache didn't actually shrink."""
    import pytest as _pytest

    from workbench.engine.engine import Engine

    class NonTrimmableModel:
        vocab_size = 8

        def __call__(self, inputs, cache=None):
            return mx.zeros((1, inputs.shape[1], self.vocab_size))

        def make_cache(self):
            return [NonTrimmableCacheState()]

    engine = Engine(NonTrimmableModel(), fake_tokenizer)
    engine.start_session()
    engine._cached_tokens = [1, 2, 3, 4, 5]

    with _pytest.raises(RuntimeError, match="trim_to"):
        engine.trim_to(2)
