"""Vector store interface with a Chroma implementation (swap for Qdrant/pgvector later)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from docrag.config import Settings, get_settings


class VectorStore(Protocol):
    def add_documents(self, docs: list[Document], ids: list[str]) -> None: ...
    def search(self, query: str, k: int, mmr: bool = False) -> list[Document]: ...
    def count(self) -> int: ...
    def delete_by_source(self, source_file: str, keep_ids: Sequence[str] = ()) -> None: ...
    def list_sources(self) -> list[str]: ...
    def count_by_source(self) -> dict[str, int]: ...
    def reset(self) -> None: ...


class ChromaStore:
    def __init__(self, embeddings: Embeddings, settings: Settings | None = None) -> None:
        self._s = settings or get_settings()
        self._embeddings = embeddings
        self._db = self._open()
        self._stamp = self._read_stamp()

    def _read_stamp(self) -> tuple[int, int] | None:
        # Chroma's vector index is cached in memory per process, so a second process writing to
        # the same directory (e.g. `python -m docrag.ingest` while the API runs) is invisible
        # until the client is reopened; the sqlite file changes on every write, so we watch it.
        try:
            st = (Path(self._s.chroma_dir) / "chroma.sqlite3").stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def _refresh(self) -> None:
        """Reopen the index if another process changed it since this store last touched it."""
        stamp = self._read_stamp()
        if stamp != self._stamp:
            from chromadb.api.shared_system_client import SharedSystemClient

            SharedSystemClient.clear_system_cache()
            self._db = self._open()
            self._stamp = self._read_stamp()

    def _open(self) -> Any:
        from langchain_chroma import Chroma

        return Chroma(
            collection_name=self._s.collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(self._s.chroma_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, docs: list[Document], ids: list[str]) -> None:
        # Deterministic ids + upsert semantics make re-ingestion idempotent.
        if docs:
            self._refresh()
            self._db.add_documents(docs, ids=ids)
            self._stamp = self._read_stamp()

    def search(self, query: str, k: int, mmr: bool = False) -> list[Document]:
        self._refresh()
        if mmr:
            return list(self._db.max_marginal_relevance_search(query, k=k, fetch_k=max(20, k * 3)))
        return list(self._db.similarity_search(query, k=k))

    def count(self) -> int:
        self._refresh()
        return int(self._db._collection.count())

    def delete_by_source(self, source_file: str, keep_ids: Sequence[str] = ()) -> None:
        """Delete this file's chunks except those whose ids are in `keep_ids`."""
        self._refresh()
        existing = self._db._collection.get(where={"source_file": source_file}, include=[])["ids"]
        stale = [i for i in existing if i not in set(keep_ids)]
        if stale:
            self._db._collection.delete(ids=stale)
            self._stamp = self._read_stamp()

    def list_sources(self) -> list[str]:
        self._refresh()
        metas = self._db._collection.get(include=["metadatas"])["metadatas"] or []
        return sorted({str(m["source_file"]) for m in metas if m and "source_file" in m})

    def count_by_source(self) -> dict[str, int]:
        self._refresh()
        metas = self._db._collection.get(include=["metadatas"])["metadatas"] or []
        counts: dict[str, int] = {}
        for m in metas:
            if m and "source_file" in m:
                name = str(m["source_file"])
                counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items()))

    def reset(self) -> None:
        self._db.delete_collection()
        self._db = self._open()
        self._stamp = self._read_stamp()
