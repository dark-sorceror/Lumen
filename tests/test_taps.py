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


def test_engine_captures_hidden_states_for_requested_layers(fake_layered_model, fake_tokenizer):
    engine = Engine(fake_layered_model, fake_tokenizer)
    events = list(engine.generate([1], GenParams(max_tokens=2, hidden_layers=(0, 2))))

    assert events, "expected at least one token event"
    for e in events:
        assert set(e.hidden) == {0, 2}
        assert e.hidden[0].shape == (fake_layered_model.hidden_dim,)


def test_hidden_capture_is_off_by_default(fake_layered_model, fake_tokenizer):
    """The default path must not wrap layers or carry activations."""
    engine = Engine(fake_layered_model, fake_tokenizer)
    events = list(engine.generate([1], GenParams(max_tokens=2)))
    assert all(e.hidden == {} for e in events)


def test_captured_hidden_is_that_layer_s_output(fake_layered_model, fake_tokenizer):
    """Each FakeLayer adds (index + 1) to a hidden state seeded with the last
    token id, so the captured value pins WHICH block's output was recorded --
    not merely that something was recorded."""
    engine = Engine(fake_layered_model, fake_tokenizer)
    first = next(iter(engine.generate([1], GenParams(max_tokens=1, hidden_layers=(0, 2)))))
    # seed 1.0 -> +1 (layer 0) = 2.0 -> +2 = 4.0 -> +3 (layer 2) = 7.0
    assert float(first.hidden[0][0].item()) == 2.0
    assert float(first.hidden[2][0].item()) == 7.0


def test_layer_list_is_restored_after_capture(fake_layered_model, fake_tokenizer):
    """A leaked wrapper would silently tap every later generation."""
    originals = list(fake_layered_model.layers)
    engine = Engine(fake_layered_model, fake_tokenizer)
    list(engine.generate([1], GenParams(max_tokens=2, hidden_layers=(0, 1, 2))))
    assert fake_layered_model.layers == originals
