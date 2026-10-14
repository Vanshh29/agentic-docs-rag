"""Chunking with deterministic IDs (hash of file + position) for idempotent indexing."""

from __future__ import annotations

import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def detect_language(text: str) -> str | None:
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return str(detect(text))
    except Exception:
        return None


def chunk_documents(
    docs: list[Document], chunk_size: int = 800, chunk_overlap: int = 120
) -> tuple[list[Document], list[str]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # prefer paragraph, then line boundaries
    )
    chunks: list[Document] = []
    ids: list[str] = []
    for doc in docs:
        loc = doc.metadata.get("page", doc.metadata.get("section", ""))
        lang = detect_language(doc.page_content)
        for pos, text in enumerate(splitter.split_text(doc.page_content)):
            key = f"{doc.metadata['source_file']}|{loc}|{pos}"
            chunk_id = hashlib.sha1(key.encode()).hexdigest()[:16]
            meta = {**doc.metadata, "chunk_id": chunk_id}
            if lang:
                meta["language"] = lang
            chunks.append(Document(page_content=text, metadata=meta))
            ids.append(chunk_id)
    return chunks, ids
