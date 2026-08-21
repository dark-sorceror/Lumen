"""Attachment handling: file -> extracted text -> doc_chunk segments.

See `workbench.attachments.processors` for the `AttachmentProcessor`
extensibility seam (the VLM slots in as one more processor)."""
from __future__ import annotations

from workbench.attachments.processors import AttachmentProcessor, process_attachment
from workbench.attachments.store import AttachmentStore, kind_for_mime, store

__all__ = [
    "AttachmentProcessor",
    "process_attachment",
    "AttachmentStore",
    "kind_for_mime",
    "store",
]
