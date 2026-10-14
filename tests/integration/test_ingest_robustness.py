"""Ingestion robustness: per-file isolation, path-based ids, replace-on-reingest, PDF loader."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from docrag.config import Settings
from docrag.ingest import ingest_path
from docrag.ingest.loaders import load_path, load_pdf
from docrag.store import ChromaStore
from tests.fakes import HashEmbeddings, make_settings
from tests.helpers import make_docx, make_pdf

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"
DATASET = Path(__file__).resolve().parents[2] / "eval" / "dataset.jsonl"


@pytest.fixture
def store(tmp_path: Path) -> tuple[ChromaStore, Settings]:
    s = make_settings(chroma_dir=tmp_path / "chroma", collection_name="robust")
    return ChromaStore(HashEmbeddings(), s), s


def test_bad_files_do_not_abort_ingest(tmp_path: Path, store: tuple[ChromaStore, Settings]) -> None:
    st, s = store
    root = tmp_path / "in"
    root.mkdir()
    (root / "corrupt.pdf").write_bytes(b"this is not a pdf")
    (root / "empty.docx").write_bytes(b"")
    (root / "README.md").write_text("docs about the folder")
    (root / ".hidden").write_text("x")
    shutil.copy(SAMPLES / "espresso_machine_manual_en.pdf", root / "good.pdf")
    stats = ingest_path(root, st, s)
    assert stats.files == 1 and stats.chunks > 0 and st.count() == stats.chunks
    assert stats.failed == 2
    assert {n for n, _ in stats.failed_files} == {"corrupt.pdf", "empty.docx"}
    assert stats.skipped == 2  # README.md and the dotfile are reported, not silent


def test_skipped_files_are_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / "README.md").write_text("x")
    (tmp_path / ".DS_Store").write_text("x")
    with caplog.at_level("WARNING"):
        _, skipped, _ = load_path(tmp_path)
    assert len(skipped) == 2
    assert "README.md" in caplog.text and ".DS_Store" in caplog.text


def test_same_named_files_in_different_dirs_both_indexed(
    tmp_path: Path, store: tuple[ChromaStore, Settings]
) -> None:
    st, s = store
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    make_pdf(tmp_path / "a" / "manual.pdf", ["Alpha text"])
    make_pdf(tmp_path / "b" / "manual.pdf", ["Alpha text"])  # identical content, other folder
    stats = ingest_path(tmp_path, st, s)
    assert stats.sources == ["a/manual.pdf", "b/manual.pdf"]
    assert st.count() == 2 and st.list_sources() == ["a/manual.pdf", "b/manual.pdf"]


def test_edited_file_leaves_no_stale_chunks(
    tmp_path: Path, store: tuple[ChromaStore, Settings]
) -> None:
    st, s = store
    f = tmp_path / "m.pdf"
    make_pdf(f, ["First page about kettles", "Second page about toasters", "Third page"])
    ingest_path(tmp_path, st, s)
    assert st.count() == 3
    make_pdf(f, ["Only one page now about blenders"])
    ingest_path(tmp_path, st, s)
    assert st.count() == 1
    assert "blenders" in st.search("blenders", k=3)[0].page_content


def test_rerun_adds_no_duplicates_and_other_files_untouched(
    tmp_path: Path, store: tuple[ChromaStore, Settings]
) -> None:
    st, s = store
    ingest_path(SAMPLES, st, s, reset=True)
    n = st.count()
    ingest_path(SAMPLES, st, s)
    ingest_path(SAMPLES / "espresso_machine_manual_en.pdf", st, s)  # single-file re-ingest
    assert st.count() == n
    assert "espresso_machine_manual_en.pdf" in st.list_sources()


def test_eval_expected_sources_match_ingested_source_files() -> None:
    docs, _, failed = load_path(SAMPLES)
    assert failed == []
    known = {d.metadata["source_file"] for d in docs}
    expected = {
        e["source_file"]
        for line in DATASET.read_text().splitlines()
        if line.strip()
        for e in json.loads(line)["expected_sources"]
    }
    assert expected <= known, f"dataset references unknown files: {expected - known}"


def test_pdf_loader_pages_and_empty_pages(tmp_path: Path) -> None:
    f = make_pdf(tmp_path / "p.pdf", ["Page one text", "", "Page three text"])
    docs = load_pdf(f)
    assert [d.metadata["page"] for d in docs] == [1, 3]  # blank page skipped, numbering kept
    assert all(d.metadata["source_file"] == "p.pdf" for d in docs)
    assert "Page three" in docs[1].page_content


def test_docx_sections_via_load_path(tmp_path: Path) -> None:
    make_docx(tmp_path / "m.docx", {"Reinigung": "Täglich reinigen.", "Wartung": "Jährlich."})
    docs, _, failed = load_path(tmp_path)
    assert failed == [] and {d.metadata["section"] for d in docs} == {"Reinigung", "Wartung"}


def test_failed_add_keeps_existing_chunks(
    tmp_path: Path, store: tuple[ChromaStore, Settings]
) -> None:
    st, s = store
    f = tmp_path / "m.pdf"
    make_pdf(f, ["Original page about kettles"])
    ingest_path(tmp_path, st, s)
    before = st.count()

    def boom(docs: list, ids: list) -> None:  # type: ignore[type-arg]
        raise RuntimeError("embedding failed")

    make_pdf(f, ["Edited page about toasters"])
    st.add_documents = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        ingest_path(tmp_path, st, s)
    del st.add_documents  # restore the real method
    assert st.count() == before
    assert "kettles" in st.search("kettles", k=1)[0].page_content
