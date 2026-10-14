"""ingest -> query on the committed sample documents, fully offline (hash embeddings)."""

from __future__ import annotations

from pathlib import Path

import pytest

from docrag.config import Settings
from docrag.ingest import ingest_path
from docrag.ingest.loaders import load_path
from docrag.store import ChromaStore
from tests.fakes import HashEmbeddings, make_settings

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"


@pytest.fixture
def store(tmp_path: Path) -> tuple[ChromaStore, Settings]:
    s = make_settings(chroma_dir=tmp_path / "chroma", collection_name="test_docs")
    return ChromaStore(HashEmbeddings(), s), s


def test_ingest_is_idempotent(store: tuple[ChromaStore, Settings]) -> None:
    st, s = store
    first = ingest_path(SAMPLES, st, s, reset=True)
    n = st.count()
    assert first.files == 3 and first.chunks == n > 10
    ingest_path(SAMPLES, st, s)  # re-run: no duplicates
    assert st.count() == n


def test_query_returns_metadata(store: tuple[ChromaStore, Settings]) -> None:
    st, s = store
    ingest_path(SAMPLES, st, s, reset=True)
    hits = st.search("Wie lange dauert die Ladezeit der Batterie", k=3)
    assert any(h.metadata["source_file"] == "forklift_charger_manual_de.pdf" for h in hits)
    for h in hits:
        assert {"source_file", "chunk_id"} <= h.metadata.keys()
        assert "page" in h.metadata or "section" in h.metadata


def test_docx_sections_and_unsupported_skipped(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("ignore me")
    docs, skipped, failed = load_path(tmp_path)
    assert docs == [] and [p.name for p in skipped] == ["notes.txt"]
    docs, _, _ = load_path(SAMPLES / "air_purifier_manual_de.docx")
    assert {d.metadata["section"] for d in docs} >= {"Filterwechsel", "Betriebsarten"}
