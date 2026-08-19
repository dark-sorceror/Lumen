"""Trim-and-continue must be EXACT: generating with an edited context via cache trim
must produce byte-identical output to generating from scratch. Greedy (T=0) makes
this deterministic."""
import pytest

pytestmark = pytest.mark.slow


def _greedy_ids(engine, tokens, n):
    from workbench.engine.engine import GenParams
    return [e.token_id for e in engine.generate_with_cache(tokens, GenParams(max_tokens=n))]


def test_trim_and_continue_matches_from_scratch():
    from workbench.engine.engine import Engine, GenParams
    from workbench.engine.loader import TEST_MODEL, load_model

    model, tokenizer = load_model(TEST_MODEL)

    base = tokenizer.encode("The capital of France is Paris. The capital of Germany is")
    edited = tokenizer.encode("The capital of France is Paris. The capital of Italy is")

    # From-scratch reference for the edited prompt
    ref_engine = Engine(model, tokenizer)
    ref_engine.start_session()
    ref = _greedy_ids(ref_engine, edited, 8)

    # Session engine: generate on base, then EDIT (shared prefix) and continue
    engine = Engine(model, tokenizer)
    engine.start_session()
    _ = _greedy_ids(engine, base, 8)
    got = _greedy_ids(engine, edited, 8)

    assert got == ref


@pytest.mark.parametrize("chunk_size", [3, 4, 5])
def test_chunked_prefill_matches_single_chunk_prefill(chunk_size):
    """Self-review concern: chunk-boundary correctness against a REAL model's
    attention/KV cache (the fakes in test_engine.py are semantics-blind to
    chunking since they predict purely from the input's last token, ignoring
    cache content). The 12-token prompt below is an exact multiple of
    chunk_size=4 and a non-multiple for 3 and 5, covering both boundary
    shapes; chunked prefill must be byte-identical to single-shot prefill."""
    from workbench.engine.engine import Engine
    from workbench.engine.loader import TEST_MODEL, load_model

    model, tokenizer = load_model(TEST_MODEL)
    prompt = tokenizer.encode("The capital of France is Paris. The capital of Germany is")
    assert len(prompt) == 12

    baseline_engine = Engine(model, tokenizer)  # default chunk size (2048) -> single chunk
    baseline_engine.start_session()
    baseline = _greedy_ids(baseline_engine, prompt, 8)

    chunked_engine = Engine(model, tokenizer)
    chunked_engine._PREFILL_CHUNK_SIZE = chunk_size
    chunked_engine.start_session()
    chunked = _greedy_ids(chunked_engine, prompt, 8)

    assert chunked == baseline


def test_empty_suffix_resume_matches_from_scratch_on_real_model():
    """Finding 2: the empty-suffix resume path in generate_with_cache
    (calling it again with full_tokens == exactly what's already cached,
    trimming the cache back by one and reprocessing the last token to
    re-derive fresh decode logits) is only exercised by cache-blind fakes
    elsewhere. Those fakes can't detect real KV/attention corruption. Prove
    it against the real model: generate N tokens, resume with the exact
    same (prompt + N generated) as full_tokens (empty suffix), generate M
    more -- the combined sequence must be byte-identical to a from-scratch
    generate_with_cache run of the same total length."""
    from workbench.engine.engine import Engine, GenParams
    from workbench.engine.loader import TEST_MODEL, load_model

    model, tokenizer = load_model(TEST_MODEL)
    prompt = tokenizer.encode("The capital of France is Paris. The capital of Germany is")

    engine = Engine(model, tokenizer)
    engine.start_session()
    first = _greedy_ids(engine, prompt, 4)

    # Resume with full_tokens == exactly what's already cached: empty
    # suffix, must hit the trim-by-one-and-reprocess path.
    second = _greedy_ids(engine, prompt + first, 4)

    got = first + second

    ref_engine = Engine(model, tokenizer)
    ref_engine.start_session()
    ref = _greedy_ids(ref_engine, prompt, 8)

    assert got == ref
