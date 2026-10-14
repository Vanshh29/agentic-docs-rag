"""Baseline (naive RAG) vs agentic graph on eval/dataset.jsonl.

Usage: python -m eval.run_eval [--limit N] [--concurrency 4] [--systems baseline agentic]

Writes eval/results/<timestamp>.json and eval/results/latest.md. Only reports measured numbers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document

from docrag import prompts
from docrag.config import Settings, get_settings
from docrag.graph import Deps, build_baseline_graph, build_graph, run_question
from docrag.graph.nodes import format_context
from docrag.llm import make_embeddings, make_judge_llm, make_llm, structured
from docrag.schemas import CorrectnessGrade, HallucinationGrade
from docrag.store import ChromaStore

ROOT = Path(__file__).resolve().parent
UNANSWERABLE = {"unanswerable", "out_of_scope"}


class CallCounter(BaseCallbackHandler):
    """Counts chat-model invocations (one per LLM call, including each abatch element)."""

    def __init__(self) -> None:
        self.calls = 0

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        self.calls += 1


@dataclass
class ItemResult:
    question: str
    category: str
    language: str
    status: str = "error"
    answer: str = ""
    retrieval_hit: bool | None = None
    hit_at_1: bool | None = None
    hit_at_3: bool | None = None
    correct: bool | None = None
    grounded: bool | None = None
    citation_precision: float | None = None
    refused: bool | None = None
    latency_s: float = 0.0
    llm_calls: int = 0
    retries: int = 0
    regenerations: int = 0
    cited: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""


def _matches(meta: dict[str, Any], expected: dict[str, Any]) -> bool:
    if meta.get("source_file") != expected["source_file"]:
        return False
    if "page" in expected:
        return bool(meta.get("page") == expected["page"])
    return bool(meta.get("section") == expected.get("section"))


def hit_at_k(docs: list[Document], expected: list[dict[str, Any]], k: int) -> bool:
    """True if any expected source matches one of the first k retrieved chunks (rank order)."""
    return any(_matches(d.metadata, e) for d in docs[:k] for e in expected)


async def judge_item(
    judge: Any, item: dict[str, Any], state: dict[str, Any], res: ItemResult
) -> None:
    expected = item["expected_sources"]
    answerable = item["category"] not in UNANSWERABLE
    answered = res.status == "answered"
    docs: list[Document] = state.get("documents", [])
    used: list[Document] = state.get("relevant_documents", [])

    if answerable:
        res.retrieval_hit = hit_at_k(docs, expected, len(docs))
        res.hit_at_1 = hit_at_k(docs, expected, 1)
        res.hit_at_3 = hit_at_k(docs, expected, 3)
        res.cited = [c.model_dump() for c in state.get("citations", [])]
        if answered and res.cited:
            ok = sum(
                any(
                    _matches(
                        {"source_file": c["source_file"], "page": c["page"], "section": c["page"]},
                        e,
                    )
                    for e in expected
                )
                for c in res.cited
            )
            res.citation_precision = ok / len(res.cited)
        else:
            res.citation_precision = 0.0
        if answered:
            verdict = await structured(judge, CorrectnessGrade).ainvoke(
                [
                    ("system", prompts.JUDGE_CORRECTNESS_SYSTEM),
                    (
                        "human",
                        prompts.JUDGE_CORRECTNESS_HUMAN.format(
                            question=item["question"],
                            reference=item["reference_answer"],
                            candidate=res.answer,
                        ),
                    ),
                ]
            )
            res.correct = verdict.correct
        else:
            res.correct = False
    else:
        res.refused = not answered

    if answered and used:
        g = await structured(judge, HallucinationGrade).ainvoke(
            [
                ("system", prompts.JUDGE_GROUNDED_SYSTEM),
                (
                    "human",
                    prompts.JUDGE_GROUNDED_HUMAN.format(
                        context=format_context(used), answer=res.answer
                    ),
                ),
            ]
        )
        res.grounded = g.grounded


async def eval_one(
    graph: Any, judge: Any, item: dict[str, Any], sem: asyncio.Semaphore
) -> ItemResult:
    res = ItemResult(item["question"], item["category"], item["language"])
    async with sem:
        counter = CallCounter()
        t0 = time.perf_counter()
        try:
            for attempt in range(5):  # OpenRouter 402 "in-flight budget" is transient
                try:
                    counter.calls = 0
                    state = await run_question(graph, item["question"], callbacks=[counter])
                    break
                except Exception as e:
                    if "402" not in str(e) or attempt == 4:
                        raise
                    await asyncio.sleep(5 * (attempt + 1))
            res.latency_s = time.perf_counter() - t0
            res.llm_calls = counter.calls
            res.status = state.get("status", "error")
            res.answer = state.get("generation", "")
            res.retries = state.get("retry_count", 0)
            res.regenerations = state.get("regeneration_count", 0)
            await judge_item(judge, item, dict(state), res)
        except Exception as e:  # keep going; errors are reported, not hidden
            res.error = f"{type(e).__name__}: {e}"
            res.latency_s = time.perf_counter() - t0
    return res


def _rate(vals: list[bool | None]) -> float | None:
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None


def aggregate(results: list[ItemResult]) -> dict[str, Any]:
    ok = [r for r in results if not r.error]
    answerable = [r for r in ok if r.category not in UNANSWERABLE]
    cp = [r.citation_precision for r in answerable if r.citation_precision is not None]
    by_cat: dict[str, list[bool | None]] = defaultdict(list)
    for r in answerable:
        by_cat[r.category].append(r.correct)
    return {
        "n_items": len(results),
        "n_errors": len(results) - len(ok),
        "hit_rate_at_k": _rate([r.retrieval_hit for r in answerable]),
        "hit_rate_at_1": _rate([r.hit_at_1 for r in answerable]),
        "hit_rate_at_3": _rate([r.hit_at_3 for r in answerable]),
        "answer_correctness": _rate([r.correct for r in answerable]),
        "groundedness": _rate([r.grounded for r in ok]),
        "citation_accuracy": sum(cp) / len(cp) if cp else None,
        "correct_refusal_rate": _rate([r.refused for r in ok if r.category in UNANSWERABLE]),
        "false_refusal_rate": _rate([r.status != "answered" for r in answerable]),
        "avg_latency_s": sum(r.latency_s for r in ok) / len(ok) if ok else None,
        "avg_llm_calls": sum(r.llm_calls for r in ok) / len(ok) if ok else None,
        "correctness_by_category": {k: _rate(v) for k, v in by_cat.items()},
    }


def _fmt(v: float | None, pct: bool = True, nd: int = 1) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%" if pct else f"{v:.{nd}f}"


def render_markdown(meta: dict[str, Any], agg: dict[str, dict[str, Any]]) -> str:
    names = list(agg)
    rows = [
        ("Retrieval hit rate@1", "hit_rate_at_1", True),
        ("Retrieval hit rate@3", "hit_rate_at_3", True),
        ("Retrieval hit rate@k", "hit_rate_at_k", True),
        ("Answer correctness (LLM judge)", "answer_correctness", True),
        ("Groundedness (LLM judge)", "groundedness", True),
        ("Citation accuracy", "citation_accuracy", True),
        ("Correct refusal rate (unanswerable)", "correct_refusal_rate", True),
        ("False refusal rate (answerable)", "false_refusal_rate", True),
        ("Avg latency (s)", "avg_latency_s", False),
        ("Avg LLM calls / question", "avg_llm_calls", False),
        ("Errors", "n_errors", False),
    ]
    out = [
        "# Evaluation results: naive RAG vs. agentic RAG",
        "",
        f"- Run: `{meta['timestamp']}` | dataset items: {meta['n_items']} "
        f"({meta['n_unanswerable']} unanswerable/out-of-scope)",
        f"- LLM: `{meta['llm_model']}` ({meta['provider']}) | judge: `{meta['judge_model']}` "
        f"(temperature 0, prompt {meta['judge_prompt_version']}) | embeddings: "
        f"`{meta['embedding_model']}` | k={meta['top_k']}",
        "- **The dataset was drafted by an LLM-assisted process and needs human review before "
        "these numbers are trusted.** Judge and generator may share a model family.",
        "",
        "| Metric | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for label, key, pct in rows:
        cells = []
        for n in names:
            v = agg[n].get(key)  # runs carried over with --reuse may predate newer metrics
            if key == "n_errors":
                cells.append("n/a" if v is None else str(v))
            else:
                cells.append(_fmt(v, pct, 2))
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    cats = sorted({c for n in names for c in agg[n]["correctness_by_category"]})
    out += [
        "",
        "## Answer correctness by question category",
        "",
        "| Category | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for c in cats:
        out.append(
            f"| {c} | "
            + " | ".join(_fmt(agg[n]["correctness_by_category"].get(c)) for n in names)
            + " |"
        )
    out += [
        "",
        "Definitions: hit rate@k = an expected (file, page/section) is among the final retrieved "
        "chunks (hit@1 / hit@3: within the first 1 / 3 chunks in retrieval order); "
        "citation accuracy = mean share of returned citations that match an expected "
        "source (0 when the answer has none); groundedness = share of answered questions whose "
        "answer an independent judge finds fully supported by the chunks used; refusal = the "
        "system declined or routed out of scope.",
    ]
    return "\n".join(out) + "\n"


async def main_async(args: argparse.Namespace) -> None:
    settings: Settings = get_settings()
    items = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    if args.limit:
        items = items[: args.limit]
    deps = Deps(make_llm(settings), ChromaStore(make_embeddings(settings), settings), settings)
    graphs = {"baseline": build_baseline_graph(deps), "agentic": build_graph(deps)}
    judge = make_judge_llm(settings)
    sem = asyncio.Semaphore(args.concurrency)

    def label(system: str) -> str:
        return (
            f"agentic [MIN_RELEVANT_DOCS={settings.min_relevant_docs}]"
            if system == "agentic"
            else system
        )

    all_results: dict[str, list[ItemResult]] = {}
    agg: dict[str, dict[str, Any]] = {}
    items_out: dict[str, list[dict[str, Any]]] = {}
    if (
        args.reuse
    ):  # carry over systems measured in an earlier run (same dataset) without re-running
        prev = json.loads(args.reuse.read_text())
        prev_min = prev["meta"].get("min_relevant_docs", 2)
        for n, a in prev["aggregate"].items():
            key = f"agentic [MIN_RELEVANT_DOCS={prev_min}]" if n == "agentic" else n
            if key not in {label(s) for s in args.systems}:
                agg[key], items_out[key] = a, prev["items"][n]
    for name in args.systems:
        print(f"Running {name} on {len(items)} questions...")
        all_results[name] = await asyncio.gather(
            *(eval_one(graphs[name], judge, it, sem) for it in items)
        )
        agg[label(name)] = aggregate(all_results[name])
        items_out[label(name)] = [r.__dict__ for r in all_results[name]]

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    meta = {
        "timestamp": ts,
        "n_items": len(items),
        "n_unanswerable": sum(i["category"] in UNANSWERABLE for i in items),
        "llm_model": settings.llm_model,
        "judge_model": settings.judge_model,
        "provider": settings.llm_provider,
        "embedding_model": settings.embedding_model,
        "top_k": settings.top_k,
        "min_relevant_docs": settings.min_relevant_docs,
        "judge_prompt_version": prompts.JUDGE_PROMPT_VERSION,
    }
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    payload = {
        "meta": meta,
        "aggregate": agg,
        "items": items_out,
    }
    (out_dir / f"{ts}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    md = render_markdown(meta, agg)
    (out_dir / "latest.md").write_text(md)
    print(md)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=ROOT / "dataset.jsonl")
    p.add_argument("--reuse", type=Path, help="results JSON whose other systems are carried over")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument(
        "--systems", nargs="+", default=["baseline", "agentic"], choices=["baseline", "agentic"]
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
