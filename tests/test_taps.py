import math

import mlx.core as mx

from workbench.engine.engine import Engine, GenParams
from workbench.engine.taps import top_k_logprobs


def test_top_k_logprobs_normalized():
    logits = mx.array([[0.0, 1.0, 2.0, 3.0]])
    result = top_k_logprobs(logits, k=2)
    assert list(result.keys()) == [3, 2]                      # sorted desc
    assert math.isclose(sum(math.exp(v) for v in top_k_logprobs(logits, k=4).values()),
                        1.0, rel_tol=1e-5)                    # full k sums to 1
    assert result[3] > result[2]


def test_engine_emits_logprobs_when_requested(fake_model, fake_tokenizer):
    engine = Engine(fake_model, fake_tokenizer)
    events = list(engine.generate([1], GenParams(max_tokens=2, top_k_logprobs=3)))
    for e in events:
        assert len(e.top_logprobs) == 3
        assert e.token_id == max(e.top_logprobs, key=e.top_logprobs.get)


def test_engine_emits_no_logprobs_by_default(fake_model, fake_tokenizer):
    engine = Engine(fake_model, fake_tokenizer)
    events = list(engine.generate([1], GenParams(max_tokens=2)))
    assert all(e.top_logprobs == {} for e in events)
