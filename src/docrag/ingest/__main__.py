"""CLI: python -m docrag.ingest --path data/raw --reset"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from docrag.config import get_settings
from docrag.ingest.pipeline import ingest_path
from docrag.llm import make_embeddings
from docrag.store import ChromaStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDF/DOCX files into the vector store.")
    parser.add_argument("--path", type=Path, default=Path("data/raw"))
    parser.add_argument("--reset", action="store_true", help="Clear the index first.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = get_settings()
    store = ChromaStore(make_embeddings(settings), settings)
    stats = ingest_path(args.path, store, settings, reset=args.reset)
    print(
        f"files={stats.files} pages/sections={stats.pages_or_sections} "
        f"chunks={stats.chunks} skipped={stats.skipped} failed={stats.failed} "
        f"index_size={store.count()}"
    )
    for name, err in stats.failed_files:
        print(f"FAILED {name}: {err}")


if __name__ == "__main__":
    main()
