"""Attachment processing: AttachmentProcessor implementations, the
dispatch registry, and the transient AttachmentStore. Model-free (no engine,
no MAIN_MODEL) -- these exercise text extraction / OCR / storage in
isolation, and the VLM processor with mlx-vlm's load/generate MOCKED (no
real model, no download)."""
from __future__ import annotations

import pytest

from workbench.attachments import processors as processors_mod
from workbench.attachments.processors import (
    MAX_TOTAL_CHARS,
    _PROCESSORS,
    FallbackProcessor,
    ImageOcrProcessor,
    OCR_AVAILABLE,
    PdfProcessor,
    TextProcessor,
    process_attachment)
from workbench.attachments.store import AttachmentStore, kind_for_mime


# --------------------------------------------------------------- TextProcessor

def test_text_processor_can_handle():
    proc = TextProcessor()
    assert proc.can_handle("text/plain", "notes.txt")
    assert proc.can_handle("text/markdown", "readme.md")
    assert proc.can_handle("application/octet-stream", "data.csv")
    assert proc.can_handle("application/octet-stream", "data.json")
    assert not proc.can_handle("application/pdf", "doc.pdf")
    assert not proc.can_handle("image/png", "pic.png")


def test_text_processor_chunks_multi_paragraph_text():
    proc = TextProcessor()
    para = "x" * 1000
    text = "\n\n".join([para] * 4)  # 4000+ chars, should split into >1 chunk
    chunks = proc.process(text.encode("utf-8"), "notes.txt", "text/plain")
    assert len(chunks) > 1
    # Every paragraph's content must survive somewhere in the chunks.
    joined = "\n\n".join(chunks)
    assert joined.count("x" * 1000) == 4
    for chunk in chunks:
        assert len(chunk) <= 1600  # target ~1500 with some slack


def test_text_processor_decodes_invalid_utf8_with_replace():
    proc = TextProcessor()
    data = b"hello \xff\xfe world"
    chunks = proc.process(data, "bad.txt", "text/plain")
    assert chunks  # doesn't raise
    assert "hello" in chunks[0]


def test_text_processor_single_short_chunk():
    proc = TextProcessor()
    chunks = proc.process(b"hello world", "a.txt", "text/plain")
    assert chunks == ["hello world"]


# ---------------------------------------------------------------- PdfProcessor

def _make_pdf_bytes(pages: list[str]) -> bytes:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_pdf_processor_can_handle():
    proc = PdfProcessor()
    assert proc.can_handle("application/pdf", "doc.pdf")
    assert proc.can_handle("application/octet-stream", "doc.pdf")
    assert not proc.can_handle("text/plain", "notes.txt")


def test_pdf_processor_extracts_text_per_page():
    data = _make_pdf_bytes(["Hello from page one", "Second page content here"])
    proc = PdfProcessor()
    chunks = proc.process(data, "doc.pdf", "application/pdf")
    joined = "\n".join(chunks)
    assert "Hello from page one" in joined
    assert "Second page content here" in joined


def test_pdf_processor_notes_empty_page():
    data = _make_pdf_bytes(["Only page has text", ""])
    proc = PdfProcessor()
    chunks = proc.process(data, "doc.pdf", "application/pdf")
    joined = "\n".join(chunks)
    assert "Only page has text" in joined
    assert any("no extractable text" in c for c in chunks)


# ----------------------------------------------------------- ImageOcrProcessor

def test_image_ocr_processor_can_handle():
    proc = ImageOcrProcessor()
    assert proc.can_handle("image/png", "pic.png")
    assert proc.can_handle("image/jpeg", "pic.jpg")
    assert not proc.can_handle("application/pdf", "doc.pdf")


def _make_png_with_text(text: str) -> bytes:
    PIL = pytest.importorskip("PIL")
    from PIL import Image, ImageDraw
    import io
    img = Image.new("RGB", (400, 100), color="white")
    d = ImageDraw.Draw(img)
    d.text((10, 40), text, fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_image_ocr_processor_recognizes_or_gracefully_unavailable():
    proc = ImageOcrProcessor()
    data = _make_png_with_text("Hello Workbench")
    chunks = proc.process(data, "pic.png", "image/png")
    assert chunks
    if OCR_AVAILABLE:
        joined = " ".join(chunks)
        assert "Hello" in joined or "Workbench" in joined
    else:
        assert chunks == ["[image pic.png: OCR unavailable]"]


def test_image_ocr_processor_no_text_found_note_when_available():
    if not OCR_AVAILABLE:
        pytest.skip("ocrmac not available on this machine")
    PIL = pytest.importorskip("PIL")
    from PIL import Image
    import io
    img = Image.new("RGB", (50, 50), color="white")  # blank, no text
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    proc = ImageOcrProcessor()
    chunks = proc.process(buf.getvalue(), "blank.png", "image/png")
    assert chunks == ["[image blank.png: no text found]"]


# --------------------------------------------------------------- Fallback + dispatch

def test_fallback_processor_handles_anything():
    proc = FallbackProcessor()
    assert proc.can_handle("application/x-weird", "thing.bin")
    chunks = proc.process(b"...", "thing.bin", "application/x-weird")
    assert chunks == ["[attachment thing.bin: unsupported type application/x-weird]"]


def test_dispatch_picks_text_processor():
    chunks = process_attachment(b"hello", "notes.txt", "text/plain")
    assert chunks == ["hello"]


def test_dispatch_picks_pdf_processor():
    data = _make_pdf_bytes(["PDF dispatch check"])
    chunks = process_attachment(data, "doc.pdf", "application/pdf")
    assert any("PDF dispatch check" in c for c in chunks)


def test_dispatch_picks_fallback_for_unknown_mime():
    chunks = process_attachment(b"\x00\x01", "weird.bin", "application/x-weird")
    assert chunks == ["[attachment weird.bin: unsupported type application/x-weird]"]


def test_dispatch_size_cap_truncates_with_note():
    huge = "y" * (MAX_TOTAL_CHARS + 5000)
    chunks = process_attachment(huge.encode("utf-8"), "big.txt", "text/plain")
    total_chars = sum(len(c) for c in chunks)
    assert total_chars <= MAX_TOTAL_CHARS + 200  # truncation note itself is small
    assert any("truncated" in c for c in chunks)


# ------------------------------------------------------------------ AttachmentStore

def test_store_put_get_roundtrip():
    store = AttachmentStore()
    aid = store.put("notes.txt", "file", ["chunk one", "chunk two"])
    record = store.get(aid)
    assert record["name"] == "notes.txt"
    assert record["kind"] == "file"
    assert record["chunks"] == ["chunk one", "chunk two"]


def test_store_get_missing_returns_none():
    store = AttachmentStore()
    assert store.get("nonexistent") is None


def test_store_pop_removes_entry():
    store = AttachmentStore()
    aid = store.put("a.txt", "file", ["x"])
    popped = store.pop(aid)
    assert popped["name"] == "a.txt"
    assert store.get(aid) is None


def test_store_ids_are_unique():
    store = AttachmentStore()
    ids = {store.put("a.txt", "file", ["x"]) for _ in range(20)}
    assert len(ids) == 20


def test_kind_for_mime():
    assert kind_for_mime("image/png") == "image"
    assert kind_for_mime("application/pdf") == "pdf"
    assert kind_for_mime("text/plain") == "file"
    assert kind_for_mime("application/octet-stream") == "file"


# --------------------------------------------------------------- VlmImageProcessor
#
# All tests here
# MOCK mlx-vlm's load()/generate() -- no real model, no download, fast. The
# real round-trip lives in tests/test_vlm_slow.py (@pytest.mark.slow).


class _FakeGenerationResult:
    def __init__(self, text):
        self.text = text


def _tiny_png_bytes() -> bytes:
    from PIL import Image
    import io

    img = Image.new("RGB", (20, 20), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


