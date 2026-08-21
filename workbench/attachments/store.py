"""Transient in-memory attachment store.

Single-user localhost tool: a module-level singleton keyed by a generated
`attachment_id` is sufficient. Nothing is persisted to disk; records live
only for the process lifetime (or until explicitly popped)."""
from __future__ import annotations

import threading
import uuid


class AttachmentStore:
    """id -> {"name", "kind", "chunks"}. Thread-safe (uploads happen on the
    FastAPI event loop's executor thread; reads happen on the WS handler's
    task) via a simple lock around dict access."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, dict] = {}

    def put(self, name: str, kind: str, chunks: list[str]) -> str:
        attachment_id = uuid.uuid4().hex
        record = {"name": name, "kind": kind, "chunks": list(chunks)}
        with self._lock:
            self._records[attachment_id] = record
        return attachment_id

    def get(self, attachment_id: str) -> dict | None:
        with self._lock:
            record = self._records.get(attachment_id)
            return dict(record) if record is not None else None

    def pop(self, attachment_id: str) -> dict | None:
        with self._lock:
            record = self._records.pop(attachment_id, None)
            return dict(record) if record is not None else None


def kind_for_mime(mime_type: str) -> str:
    """Coarse attachment kind derived from mime, for the upload response and
    stored record ("image" | "pdf" | "file")."""
    if mime_type.startswith("image/"):
        return "image"
    if mime_type == "application/pdf":
        return "pdf"
    return "file"


# Module-level singleton: fine for a single-user localhost tool (see
# docstring above). `workbench.server.app` imports and uses this directly.
store = AttachmentStore()
