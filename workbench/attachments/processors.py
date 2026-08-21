"""Attachment processors: turn uploaded file bytes into extracted TEXT
CHUNKS.

`AttachmentProcessor` is the seam that kept a real VLM from being
a rewrite: every processor's contract is `bytes -> list[str]` (plain text
chunks), and `process_attachment()` dispatches through one ORDERED registry
(`_PROCESSORS`). `VlmImageProcessor` (image -> a text *description* via
mlx-vlm) registers in that same list ahead of
`ImageOcrProcessor` -- same interface, same return type, same downstream
append-as-doc_chunk path in `workbench/server/app.py`. No caller changes.
When the VLM is disabled (`WORKBENCH_DISABLE_VLM=1`) or mlx-vlm isn't
importable, dispatch falls through to `ImageOcrProcessor` (text extraction:
text-in-image only, via Apple Vision/`ocrmac`).
"""
from __future__ import annotations

import io
import os
import tempfile
import threading
from abc import ABC, abstractmethod

# Token-budget safety net: cap the TOTAL
# extracted text per attachment and make any truncation visible in the
# context rather than silently dropping content.
MAX_TOTAL_CHARS = 40_000
_TRUNCATION_NOTE = "[attachment truncated: extracted text exceeded {limit} chars]"

# Target chunk size in characters -- this chars-based target is just a chunking
# heuristic to keep segments individually inspectable/editable).
_CHUNK_TARGET = 1500


class AttachmentProcessor(ABC):
    """One attachment kind -> list of text chunks.

    The Text/Pdf/ImageOcr processors all extract or transcribe text
    verbatim. `VlmImageProcessor` instead returns a text
    *description* of the image's content -- same `list[str]` contract, so
    the server never needs to know "how" an attachment became text."""

    @abstractmethod
    def can_handle(self, mime_type: str, filename: str) -> bool:
        ...

    @abstractmethod
    def process(self, data: bytes, filename: str, mime_type: str) -> list[str]:
        ...


def _chunk_text(text: str, target: int = _CHUNK_TARGET) -> list[str]:
    """Chunk `text` into pieces of ~`target` chars, preferring paragraph
    boundaries, then line boundaries, then a hard split as a last resort for
    a single pathologically long line."""
    text = text.strip()
    if not text:
        return []
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""

    def flush():
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    for para in paragraphs:
        para = para.strip("\n")
        if not para:
            continue
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= target:
            current = candidate
            continue
        flush()
        if len(para) <= target:
            current = para
            continue
        # Paragraph itself exceeds target: break on lines.
        for line in para.split("\n"):
            candidate = f"{current}\n{line}" if current else line
            if len(candidate) <= target:
                current = candidate
                continue
            flush()
            if len(line) <= target:
                current = line
            else:
                # Single line still too long: hard-split.
                for i in range(0, len(line), target):
                    chunks.append(line[i:i + target])
    flush()
    return chunks


class TextProcessor(AttachmentProcessor):
    """text/plain, .txt/.md/.csv/.json -- decode as UTF-8 (best-effort) and
    chunk on paragraph/line boundaries."""

    _EXTS = (".txt", ".md", ".csv", ".json")

    def can_handle(self, mime_type: str, filename: str) -> bool:
        if mime_type.startswith("text/"):
            return True
        return filename.lower().endswith(self._EXTS)

    def process(self, data: bytes, filename: str, mime_type: str) -> list[str]:
        text = data.decode("utf-8", errors="replace")
        return _chunk_text(text)


class PdfProcessor(AttachmentProcessor):
    """application/pdf -- extract text with PyMuPDF (`fitz`), one chunk (or
    more, if a page is long) per page. A page with no extractable text (e.g.
    a scanned image page) gets a visible note rather than silently
    vanishing; routing such pages through OCR is a
    documented future enhancement, not built yet."""

    def can_handle(self, mime_type: str, filename: str) -> bool:
        return mime_type == "application/pdf" or filename.lower().endswith(".pdf")

    def process(self, data: bytes, filename: str, mime_type: str) -> list[str]:
        import fitz

        chunks: list[str] = []
        doc = fitz.open(stream=data, filetype="pdf")
        try:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text().strip()
                if text:
                    chunks.extend(_chunk_text(text))
                else:
                    chunks.append(f"[page {page_num}: no extractable text]")
        finally:
            doc.close()
        return chunks


# `mlx-vlm` is a real (if heavy, ~5-8 GB download) dependency -- the import
# is optional so the whole package still works on a machine that hasn't
# pulled it in, or before its first-use download completes. If unavailable,
# VlmImageProcessor.can_handle() returns False and dispatch falls through to
# ImageOcrProcessor (see _PROCESSORS ordering below).
try:
    import mlx_vlm  # noqa: F401 (presence check only; used via lazy imports below)

    VLM_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only where mlx-vlm is absent
    VLM_AVAILABLE = False

# Opt-out for users who don't want the multi-GB VLM download/load at all
#: set WORKBENCH_DISABLE_VLM=1 to
# force images through OcrImageProcessor (text-extraction behavior) even when
# mlx-vlm is installed and importable.
_DISABLE_VLM_ENV_VAR = "WORKBENCH_DISABLE_VLM"

_VLM_DESCRIBE_PROMPT = (
    "Describe this image in detail, including any visible text, charts, or diagrams."
)
_VLM_MAX_TOKENS = 384

# Lazy-loaded singleton (module-level, not per-processor-instance) so the
# ~5-8 GB VLM is loaded at most once per server process, on first image
# upload, and reused across every subsequent image -- never reloaded per
# call. Guarded by a lock because attachment processing runs in FastAPI's
# executor (a thread pool -- see workbench/server/app.py's
# `run_in_executor(None, process_attachment, ...)`), so two images uploaded
# close together could race on first-load, and the loaded MLX model/Metal
# command queue is not safe to drive from two threads at once regardless
# (the lock below is held for the full load-and-generate critical section,
# not just the load).
#
# GPU CONTENTION (known, accepted, documented -- not solved here): the VLM
# runs at upload time (POST /attachments, in the executor thread) and
# shares the same Metal device/GPU as the main text engine's generation
# loop (workbench/engine/engine.py), which is itself serialized by a
# per-app `gen_lock` (workbench/server/app.py) that this processor has no
# handle on. A VLM describe-call that lands *during* an in-flight text
# generation will contend for the GPU rather than queue behind it via a
# shared lock. For a single-user, occasional-attachment workbench this is
# acceptable today (MLX/Metal will interleave/serialize the work at the
# driver level rather than corrupt anything -- it's a latency/throughput
# hit, not a correctness bug), but a future refactor should have
# `process_attachment` accept (or the app pass in) the same `gen_lock` so
# VLM and text generation are explicitly serialized end-to-end.
_vlm_lock = threading.Lock()
_vlm_model = None
_vlm_processor = None


def _vlm_disabled_by_env() -> bool:
    return os.environ.get(_DISABLE_VLM_ENV_VAR, "") == "1"


def _get_vlm():
    """Return (model, processor), loading the singleton on first call. Must
    be called while holding `_vlm_lock`."""
    global _vlm_model, _vlm_processor
    if _vlm_model is None:
        from mlx_vlm import load

        from workbench.engine.loader import VLM_MODEL

        _vlm_model, _vlm_processor = load(VLM_MODEL)
    return _vlm_model, _vlm_processor


class VlmImageProcessor(AttachmentProcessor):
    """image/* -- genuine image *understanding* via a vision-language model
    (mlx-vlm). Unlike ImageOcrProcessor (which
    only transcribes text-in-image), this describes the image's actual
    visual content -- photos, charts, diagrams, UI screenshots.

    This is the "VLM-describe-to-text" sidecar: the VLM's output is plain
    text, returned as a single-element `list[str]` exactly like every other
    processor, so it becomes a `doc_chunk` segment through the same
    unmodified append path. It is NOT true multimodal token interleaving --
    the main text engine (workbench/engine/) never sees pixels, only the
    VLM's textual description of them."""

    def can_handle(self, mime_type: str, filename: str) -> bool:
        if not mime_type.startswith("image/"):
            return False
        if _vlm_disabled_by_env():
            return False
        return VLM_AVAILABLE

    def process(self, data: bytes, filename: str, mime_type: str) -> list[str]:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        suffix = os.path.splitext(filename)[1] or ".png"
        fd, tmp_path = tempfile.mkstemp(prefix="workbench-vlm-", suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)

            with _vlm_lock:
                model, processor = _get_vlm()
                prompt = apply_chat_template(
                    processor, model.config, _VLM_DESCRIBE_PROMPT, num_images=1)
                result = generate(
                    model, processor, prompt,
                    image=[tmp_path], max_tokens=_VLM_MAX_TOKENS, verbose=False)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        text = (result.text or "").strip()
        if not text:
            return [f"[image {filename}: VLM returned no description]"]
        return [text]


# `ocrmac` (pyobjc + Apple Vision) is macOS-only and may not build/import on
# every machine -- the import is optional so the whole package (and every
# other processor) still works without it. If unavailable at runtime,
# ImageOcrProcessor.process() degrades to a single graceful notice chunk
# instead of raising.
try:
    from ocrmac import ocrmac as _ocrmac_module

    OCR_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only where ocrmac is absent
    _ocrmac_module = None
    OCR_AVAILABLE = False


class ImageOcrProcessor(AttachmentProcessor):
    """image/* -- OCR via `ocrmac` (Apple Vision). Reads TEXT IN images; it
    does not describe non-text image content (that is the VLM's
    job)."""

    def can_handle(self, mime_type: str, filename: str) -> bool:
        return mime_type.startswith("image/")

    def process(self, data: bytes, filename: str, mime_type: str) -> list[str]:
        if not OCR_AVAILABLE:
            return [f"[image {filename}: OCR unavailable]"]
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        img.load()
        results = _ocrmac_module.OCR(img).recognize()
        texts = [text for text, _confidence, _box in results if text and text.strip()]
        if not texts:
            return [f"[image {filename}: no text found]"]
        return _chunk_text("\n".join(texts))


class FallbackProcessor(AttachmentProcessor):
    """Anything not matched by a more specific processor -- always
    `can_handle`s (last in the registry), so dispatch never falls through
    with nothing to do."""

    def can_handle(self, mime_type: str, filename: str) -> bool:
        return True

    def process(self, data: bytes, filename: str, mime_type: str) -> list[str]:
        return [f"[attachment {filename}: unsupported type {mime_type}]"]


# Ordered registry -- THE extensibility seam (see this module's docstring). `VlmImageProcessor` sits AHEAD of
# `ImageOcrProcessor`: images go to the VLM (real understanding) when it's
# enabled and importable; when disabled (WORKBENCH_DISABLE_VLM=1) or
# mlx-vlm isn't installed, `VlmImageProcessor.can_handle()` returns False
# and dispatch falls through to `ImageOcrProcessor` (text extraction: text-in-image
# only). No change to `process_attachment()`'s signature or to any caller.
_PROCESSORS: list[AttachmentProcessor] = [
    TextProcessor(),
    PdfProcessor(),
    VlmImageProcessor(),
    ImageOcrProcessor(),
    FallbackProcessor(),
]


def _apply_size_guard(chunks: list[str]) -> list[str]:
    """Cap the total extracted text across all chunks at MAX_TOTAL_CHARS,
    appending a visible truncation note rather than silently dropping
    content."""
    total = 0
    kept: list[str] = []
    for chunk in chunks:
        if total >= MAX_TOTAL_CHARS:
            kept.append(_TRUNCATION_NOTE.format(limit=MAX_TOTAL_CHARS))
            return kept
        remaining = MAX_TOTAL_CHARS - total
        if len(chunk) > remaining:
            kept.append(chunk[:remaining])
            kept.append(_TRUNCATION_NOTE.format(limit=MAX_TOTAL_CHARS))
            return kept
        kept.append(chunk)
        total += len(chunk)
    return kept


def process_attachment(data: bytes, filename: str, mime_type: str) -> list[str]:
    """Dispatch to the first processor (in registry order) whose
    `can_handle()` matches, run it, and enforce the total-size guard."""
    for processor in _PROCESSORS:
        if processor.can_handle(mime_type, filename):
            return _apply_size_guard(processor.process(data, filename, mime_type))
    # Unreachable: FallbackProcessor.can_handle() is always True.
    raise AssertionError("no processor matched (FallbackProcessor should always match)")
