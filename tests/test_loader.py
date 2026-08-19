import pytest

pytestmark = pytest.mark.slow


def test_load_and_stream_tiny_model():
    from mlx_lm import stream_generate
    from workbench.engine.loader import TEST_MODEL, load_model

    model, tokenizer = load_model(TEST_MODEL)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Say hi."}],
        add_generation_prompt=True,
    )
    chunks = [r.text for r in stream_generate(model, tokenizer, prompt, max_tokens=16)]
    assert len(chunks) > 0 and any(c.strip() for c in chunks)
