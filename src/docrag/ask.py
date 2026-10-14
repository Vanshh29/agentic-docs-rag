"""CLI: python -m docrag.ask "question" [--baseline]"""

from __future__ import annotations

import argparse
import asyncio
import logging

from docrag.config import get_settings
from docrag.graph import Deps, build_baseline_graph, build_graph, run_question, to_response
from docrag.llm import make_embeddings, make_llm
from docrag.store import ChromaStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about the indexed documentation.")
    parser.add_argument("question")
    parser.add_argument("--baseline", action="store_true", help="Use naive RAG instead of agentic.")
    parser.add_argument("--trace", action="store_true", help="Print the step trace.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    settings = get_settings()
    deps = Deps(make_llm(settings), ChromaStore(make_embeddings(settings), settings), settings)
    graph = build_baseline_graph(deps) if args.baseline else build_graph(deps)
    resp = to_response(asyncio.run(run_question(graph, args.question)))

    print(resp.answer)
    for c in resp.citations:
        print(f"  [{c.id}] {c.source_file}, {c.page}: {c.snippet[:90]!r}")
    if args.trace:
        for step in resp.trace:
            print(f"  - {step.node}: {step.detail}")


if __name__ == "__main__":
    main()
