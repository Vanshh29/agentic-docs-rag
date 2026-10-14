from langchain_core.documents import Document

from docrag.ingest.chunking import chunk_documents


def test_long_text_splits_into_many_chunks_with_unique_ids() -> None:
    text = "\n\n".join(
        f"Absatz {i}: Größenänderung über Öl-Prüfung für Wärmetauscher. " * 4 for i in range(20)
    )
    doc = Document(page_content=text, metadata={"source_file": "a.pdf", "page": 1})
    chunks, ids = chunk_documents([doc], 300, 50)
    assert len(chunks) > 1 and len(set(ids)) == len(ids)
    assert all(len(c.page_content) <= 300 for c in chunks)
    assert any(
        "Größenänderung" in c.page_content and "Wärmetauscher" in c.page_content for c in chunks
    )
    assert all(c.metadata["chunk_id"] == i for c, i in zip(chunks, ids, strict=True))


def test_ids_are_deterministic_and_path_sensitive() -> None:
    def ids_for(name: str) -> list[str]:
        d = Document(page_content="Text", metadata={"source_file": name, "page": 1})
        return chunk_documents([d])[1]

    assert ids_for("a/m.pdf") == ids_for("a/m.pdf")
    assert ids_for("a/m.pdf") != ids_for("b/m.pdf")
