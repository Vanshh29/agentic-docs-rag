"""Record real answers for the public replay demo (data/demo_answers.json).

Runs the full pipeline once per question, against an index built from data/samples only, using
whatever LLM_PROVIDER is configured locally, and saves each answer with its citations and steps.
The public demo (DEMO_REPLAY=true) replays this file and makes no model calls.

Usage: PYTHONPATH=src python scripts/record_demo.py

Honesty notes: each answer is a single real run. If a question that should be answerable is
refused (the pipeline is not perfectly stable), it is re-run, up to 3 attempts, and the number of
attempts is printed. Questions that should be refused are recorded as refused.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "demo_answers.json"
MAX_ATTEMPTS = 3

# (question, should_be_answered). Order is the order of the example cards in the demo.
QUESTIONS: list[tuple[str, bool]] = [
    ("How long does the machine need to heat up?", True),
    ("My espresso tastes watery. How can I fix it?", True),
    ("Wie lange dauert das Laden einer Batterie mit 500 Ah?", True),
    ("Wie oft muss der HEPA-Filter H13 gewechselt werden?", True),
    ("How often should the espresso machine be descaled?", True),
    ("What does error F2 mean on the forklift charger?", True),
    ("What is the price of the espresso machine?", False),
    ("Who won the football world cup in 2018?", False),
]

BLOCKLIST = ROOT / ".demo-blocklist"  # local and gitignored: one term per line, # for comments


def blocked_terms() -> list[str]:
    """Words that must never appear in the public demo's answers (read from a local file)."""
    if not BLOCKLIST.exists():
        return []
    lines = [ln.strip() for ln in BLOCKLIST.read_text().splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def find_blocked_terms(text: str) -> list[str]:
    low = text.lower()
    return sorted({term for term in blocked_terms() if term.lower() in low})


async def record() -> dict[str, object]:
    from docrag.config import get_settings
    from docrag.graph import Deps, build_graph, to_response
    from docrag.graph.builder import run_question
    from docrag.ingest import ingest_path
    from docrag.llm import make_embeddings, make_llm
    from docrag.store import ChromaStore

    with tempfile.TemporaryDirectory() as tmp:
        settings = get_settings().model_copy(update={"chroma_dir": Path(tmp)})
        store = ChromaStore(make_embeddings(settings), settings)
        ingest_path(ROOT / "data" / "samples", store, settings)
        graph = build_graph(Deps(make_llm(settings), store, settings))
        sem = asyncio.Semaphore(2)

        async def one(question: str, should_answer: bool) -> dict[str, object]:
            async with sem:
                for attempt in range(1, MAX_ATTEMPTS + 1):
                    state = await run_question(graph, question)
                    resp = to_response(state, settings.min_relevant_docs)
                    if resp.declined != should_answer:  # got what we expect
                        break
                    print(f"  retry {attempt}: {question!r} came out wrong way round")
                else:
                    raise SystemExit(f"{question!r} never behaved as expected; not saving")
                names = find_blocked_terms(
                    json.dumps(resp.model_dump(mode="json"), ensure_ascii=False)
                )
                if names:
                    raise SystemExit(
                        f"{question!r}: answer contains blocked term(s) {names}; not saving"
                    )
                print(f"  ok ({attempt} attempt{'s' if attempt > 1 else ''}): {question}")
                return {"question": question, "response": resp.model_dump(mode="json")}

        answers = await asyncio.gather(*(one(q, a) for q, a in QUESTIONS))
    return {"recorded_at": datetime.now(UTC).strftime("%Y-%m-%d"), "answers": list(answers)}


def main() -> None:
    if not BLOCKLIST.exists():
        print("note: no .demo-blocklist file, so answers are not screened for blocked terms")
    data = asyncio.run(record())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(data['answers'])} answers to {OUT}")  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
