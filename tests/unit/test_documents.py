from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from docrag.api import create_app
from docrag.graph import Deps
from tests.fakes import FakeStore, make_fake_llm, make_settings


def _docs(name: str, n: int) -> list[Document]:
    return [
        Document(page_content=f"text {i}", metadata={"source_file": name, "chunk_id": f"{name}{i}"})
        for i in range(n)
    ]


def _setup(tmp_path: Path) -> tuple[TestClient, FakeStore, Path, Path]:
    raw, samples = tmp_path / "raw", tmp_path / "samples"
    raw.mkdir()
    samples.mkdir()
    (raw / "cv.pdf").write_bytes(b"x")
    (samples / "m.pdf").write_bytes(b"x")
    (tmp_path / "outside.pdf").write_bytes(b"x")
    store = FakeStore(_docs("m.pdf", 3) + _docs("cv.pdf", 2) + _docs("a.docx", 1))
    settings = make_settings(docs_dir=raw, samples_dir=samples)
    deps = Deps(make_fake_llm({}), store, settings)
    return TestClient(create_app(lambda: deps)), store, raw, samples


def test_list_documents_sorted(tmp_path: Path) -> None:
    c, *_ = _setup(tmp_path)
    assert c.get("/documents").json() == {
        "documents": [
            {"source_file": "a.docx", "chunks": 1, "is_sample": False},
            {"source_file": "cv.pdf", "chunks": 2, "is_sample": False},
            {"source_file": "m.pdf", "chunks": 3, "is_sample": True},
        ]
    }


def test_is_sample_flags_match_delete_file_removed(tmp_path: Path) -> None:
    c, _, raw, samples = _setup(tmp_path)
    flags = {d["source_file"]: d["is_sample"] for d in c.get("/documents").json()["documents"]}
    assert flags == {"a.docx": False, "cv.pdf": False, "m.pdf": True}
    r = c.delete("/documents/m.pdf").json()
    assert r["file_removed"] is False and (samples / "m.pdf").exists()  # sample: stays on disk
    r = c.delete("/documents/cv.pdf").json()
    assert r["file_removed"] is True and not (raw / "cv.pdf").exists()  # upload: deleted


def test_is_sample_false_when_upload_shadows_sample(tmp_path: Path) -> None:
    c, _, raw, _ = _setup(tmp_path)
    (raw / "m.pdf").write_bytes(b"x")
    flags = {d["source_file"]: d["is_sample"] for d in c.get("/documents").json()["documents"]}
    assert flags["m.pdf"] is False
    assert c.delete("/documents/m.pdf").json()["file_removed"] is True


def test_delete_removes_index_and_uploaded_copy(tmp_path: Path) -> None:
    c, store, raw, _ = _setup(tmp_path)
    assert c.get("/status").json()["chunks"] == 6
    r = c.delete("/documents/cv.pdf")
    assert r.status_code == 200
    assert r.json() == {"deleted": "cv.pdf", "chunks": 2, "file_removed": True}
    assert "cv.pdf" not in store.list_sources()
    assert not (raw / "cv.pdf").exists()
    status = c.get("/status").json()
    assert status["chunks"] == 4 and "cv.pdf" not in status["sources"]


def test_delete_unknown_is_404_json(tmp_path: Path) -> None:
    c, store, raw, _ = _setup(tmp_path)
    r = c.delete("/documents/nope.pdf")
    assert r.status_code == 404 and r.json() == {"detail": "Unknown document"}
    assert store.count() == 6 and (raw / "cv.pdf").exists()


def test_delete_traversal_rejected(tmp_path: Path) -> None:
    c, store, _, _ = _setup(tmp_path)
    for name in ("../outside.pdf", "..%2Foutside.pdf", "%2e%2e/outside.pdf", "/etc/passwd", ""):
        assert c.delete(f"/documents/{name}").status_code in (404, 405)
    assert (tmp_path / "outside.pdf").exists() and store.count() == 6


def test_delete_indexed_traversal_name_never_leaves_docs_dir(tmp_path: Path) -> None:
    # Even if a hostile name somehow is in the index, only files inside docs_dir may go.
    c, store, raw, _ = _setup(tmp_path)
    store.add_documents(_docs("../outside.pdf", 1), ids=["x"])
    r = c.delete("/documents/../outside.pdf")
    assert r.status_code in (200, 404)
    assert (tmp_path / "outside.pdf").exists()
    assert (raw / "cv.pdf").exists()


def test_delete_sample_removes_index_but_keeps_file(tmp_path: Path) -> None:
    c, store, _, samples = _setup(tmp_path)
    r = c.delete("/documents/m.pdf")
    assert r.status_code == 200 and r.json()["file_removed"] is False
    assert "m.pdf" not in store.list_sources()
    assert (samples / "m.pdf").exists()


def test_delete_never_touches_samples_even_if_dirs_overlap(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "m.pdf").write_bytes(b"x")
    deps = Deps(
        make_fake_llm({}),
        FakeStore(_docs("m.pdf", 1)),
        make_settings(docs_dir=shared, samples_dir=shared),
    )
    r = TestClient(create_app(lambda: deps)).delete("/documents/m.pdf")
    assert r.status_code == 200 and r.json()["file_removed"] is False
    assert (shared / "m.pdf").exists()


def test_delete_store_error_is_generic(tmp_path: Path) -> None:
    c, store, _, _ = _setup(tmp_path)

    def boom(*a: object, **k: object) -> None:
        raise RuntimeError("secret internals")

    store.delete_by_source = boom  # type: ignore[method-assign]
    r = c.delete("/documents/cv.pdf")
    assert r.status_code == 502 and "secret" not in r.text
