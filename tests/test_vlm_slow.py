"""Real round-trip: VlmImageProcessor against the ACTUAL mlx-vlm
model (workbench.engine.loader.VLM_MODEL) -- no mocks. Marked slow (excluded
from the default `-m "not slow"` suite, per pyproject.toml's marker and
tests/test_loader.py's precedent): first run may download several GB from
Hugging Face and take minutes. Skips (rather than fails) if mlx-vlm isn't
importable on this machine, mirroring the OCR_AVAILABLE-gated slow test in
test_attachments.py and the always-slow-marked test_loader.py."""
from __future__ import annotations

import io

import pytest

pytestmark = pytest.mark.slow


def _make_test_png() -> bytes:
    """A small, recognizable image: a solid red square with a black outline
    on a white background, plus a short text label -- gives the VLM
    something concrete (shape, color, and text) to describe."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (400, 300), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((60, 60, 220, 220), fill="red", outline="black", width=4)
    draw.text((60, 240), "WORKBENCH TEST IMAGE", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_vlm_image_processor_real_round_trip(monkeypatch):
    from workbench.attachments.processors import VLM_AVAILABLE, VlmImageProcessor

    if not VLM_AVAILABLE:
        pytest.skip("mlx-vlm not importable on this machine")

    # Independent of the test-suite-wide default (tests/conftest.py disables
    # the VLM by default so unrelated tests never trigger a real download) --
    # this test explicitly wants it enabled.
    monkeypatch.delenv("WORKBENCH_DISABLE_VLM", raising=False)

    proc = VlmImageProcessor()
    assert proc.can_handle("image/png", "test.png")

    data = _make_test_png()
    chunks = proc.process(data, "test.png", "image/png")

    assert chunks
    description = chunks[0]
    assert isinstance(description, str)
    assert len(description.strip()) > 0
    print("\n--- VLM description of test image ---\n" + description + "\n")
