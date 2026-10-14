"""End-to-end ingestion: files -> chunks -> vector store."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from docrag.config import Settings, get_settings
from docrag.ingest.chunking import chunk_documents
from docrag.ingest.loaders import load_path
from docrag.store import VectorStore

log = logging.getLogger(__name__)


@dataclass
class IngestStats:
    files: int
    pages_or_sections: int
    chunks: int
    skipped: int
    failed: int = 0
    failed_files: list[tuple[str, str]] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


def ingest_path(
    path: Path, store: VectorStore, settings: Settings | None = None, reset: bool = False
) -> IngestStats:
    s = settings or get_settings()
    if reset:
        store.reset()
    docs, skipped, failed = load_path(path)
    chunks, ids = chunk_documents(docs, s.chunk_size, s.chunk_overlap)
    sources = sorted({d.metadata["source_file"] for d in docs})
    store.add_documents(chunks, ids)  # add first: a failed add must not lose existing data
    if not reset:  # then drop only stale chunks, so edited files leave nothing behind
        for source in sources:
            keep = [
                i for c, i in zip(chunks, ids, strict=True) if c.metadata["source_file"] == source
            ]
            store.delete_by_source(source, keep_ids=keep)
    stats = IngestStats(
        files=len({d.metadata["source_file"] for d in docs}),
        pages_or_sections=len(docs),
        chunks=len(chunks),
        skipped=len(skipped),
        failed=len(failed),
        failed_files=failed,
        sources=sources,
    )
    log.info(
        "Ingested files=%d pages/sections=%d chunks=%d skipped=%d failed=%d",
        stats.files,
        stats.pages_or_sections,
        stats.chunks,
        stats.skipped,
        stats.failed,
    )
    return stats
