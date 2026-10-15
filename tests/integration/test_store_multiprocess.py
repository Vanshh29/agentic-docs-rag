"""A long-running store must see chunks written by another process (e.g. the ingest CLI)."""

import subprocess
import sys
import textwrap
from pathlib import Path

from langchain_core.documents import Document

from docrag.store import ChromaStore
from tests.fakes import HashEmbeddings, make_settings

ADD_B = (
    's.add_documents([Document(page_content="beta", metadata={"source_file": "b.pdf"})], ["b1"])'
)
DOC_A = Document(page_content="alpha", metadata={"source_file": "a.pdf"})
ROOT = Path(__file__).resolve().parents[2]


def _store(tmp_path: Path) -> ChromaStore:
    return ChromaStore(HashEmbeddings(), make_settings(chroma_dir=tmp_path))


def _other_process(tmp_path: Path, body: str) -> None:
    code = textwrap.dedent(
        f"""
        from pathlib import Path
        from langchain_core.documents import Document
        from docrag.store import ChromaStore
        from tests.fakes import HashEmbeddings, make_settings
        s = ChromaStore(HashEmbeddings(), make_settings(chroma_dir=Path({str(tmp_path)!r})))
        {body}
        """
    )
    subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, check=True, env={"PYTHONPATH": f"{ROOT}/src:{ROOT}"}
    )


def test_reader_sees_documents_added_by_another_process(tmp_path: Path) -> None:
    reader = _store(tmp_path)
    reader.add_documents(
        [Document(page_content="alpha", metadata={"source_file": "a.pdf"})], ["a1"]
    )
    assert reader.count() == 1

    _other_process(
        tmp_path,
        "s.add_documents([Document(page_content='beta', metadata={'source_file': 'b.pdf'})], "
        "['b1'])",
    )

    assert {d.metadata["source_file"] for d in reader.search("beta", k=5)} == {"a.pdf", "b.pdf"}
    assert reader.count_by_source() == {"a.pdf": 1, "b.pdf": 1}


def test_reader_sees_documents_deleted_by_another_process(tmp_path: Path) -> None:
    reader = _store(tmp_path)
    reader.add_documents(
        [Document(page_content="alpha", metadata={"source_file": "a.pdf"})], ["a1"]
    )
    _other_process(tmp_path, 's.delete_by_source("a.pdf")')
    assert reader.search("alpha", k=5) == []
