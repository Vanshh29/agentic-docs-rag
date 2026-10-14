"""Boundary conditions and adversarial LLM behaviour; all offline."""

from __future__ import annotations

import random
from typing import Any

import pytest
from langchain_core.documents import Document

from docrag import prompts
from docrag.graph import Deps, build_graph, edges, run_question
from docrag.schemas import (
    DocGrade,
    GeneratedAnswer,
    HallucinationGrade,
    RewrittenQuery,
    RouteDecision,
)
from tests.fakes import FakeStore, make_fake_llm, make_settings

DOCS = [
    Document(
        page_content=f"Descale the machine every {i} months.",
        metadata={"source_file": "m.pdf", "page": i, "chunk_id": str(i)},
    )
    for i in range(1, 4)
]


def handlers(**over: Any) -> dict[Any, Any]:
    base: dict[Any, Any] = {
        RouteDecision: lambda t: RouteDecision(route="docs"),
        RewrittenQuery: lambda t: RewrittenQuery(query="descale interval"),
        DocGrade: lambda t: DocGrade(relevant=True, reason=""),
        GeneratedAnswer: lambda t: GeneratedAnswer(
            answer="Every 3 months [1].", sufficient=True, cited_sources=[1]
        ),
        HallucinationGrade: lambda t: HallucinationGrade(grounded=True),
    }
    by_name = {k.__name__: k for k in base}
    base.update({by_name[name]: fn for name, fn in over.items()})
    return base


def build(h: dict[Any, Any], docs: list[Document] | None = None, **settings: Any):  # type: ignore[no-untyped-def]
    llm = make_fake_llm(h)
    store = FakeStore(list(DOCS) if docs is None else docs)
    s = make_settings(top_k=3, **settings)
    return build_graph(Deps(llm, store, s)), llm, store


@pytest.mark.parametrize(
    ("n_relevant", "expected"),
    [(0, "rewrite_query"), (1, "rewrite_query"), (2, "generate"), (3, "generate")],
)
def test_min_relevant_docs_boundary(n_relevant: int, expected: str) -> None:
    s = make_settings(min_relevant_docs=2, max_rewrites=2)
    state: Any = {"relevant_documents": [object()] * n_relevant, "retry_count": 0}
    assert edges.after_grade(state, s) == expected


def test_rewrite_budget_boundary() -> None:
    s = make_settings(max_rewrites=2)
    f = edges.after_grade
    assert f({"relevant_documents": [], "retry_count": 1}, s) == "rewrite_query"  # type: ignore[typeddict-item]
    assert f({"relevant_documents": [], "retry_count": 2}, s) == "respond_insufficient_info"  # type: ignore[typeddict-item]


async def test_max_rewrites_zero_never_rewrites() -> None:
    graph, llm, store = build(
        handlers(DocGrade=lambda t: DocGrade(relevant=False, reason="")), max_rewrites=0
    )
    state = await run_question(graph, "What is the airspeed of a swallow?")
    assert state["status"] == "insufficient" and state["retry_count"] == 0
    assert "RewrittenQuery" not in llm.calls and len(store.queries) == 1


async def test_max_regenerations_zero_declines_immediately() -> None:
    graph, llm, _ = build(
        handlers(HallucinationGrade=lambda t: HallucinationGrade(grounded=False)),
        max_regenerations=0,
        min_relevant_docs=1,
    )
    state = await run_question(graph, "How often must I descale?")
    assert state["status"] == "insufficient"
    assert llm.calls.count("GeneratedAnswer") == 1


@pytest.mark.parametrize("seed", range(25))
async def test_adversarial_random_llm_always_terminates(seed: int) -> None:
    rng = random.Random(seed)
    graph, llm, _ = build(
        handlers(
            RouteDecision=lambda t: RouteDecision(
                route=rng.choice(["docs", "docs", "out_of_scope"])
            ),
            DocGrade=lambda t: DocGrade(relevant=rng.random() < 0.4, reason=""),
            GeneratedAnswer=lambda t: GeneratedAnswer(
                answer=f"x [{rng.randint(0, 9)}]",
                sufficient=rng.random() < 0.7,
                cited_sources=[rng.randint(-1, 9)],
            ),
            HallucinationGrade=lambda t: HallucinationGrade(
                grounded=rng.random() < 0.3, unsupported_claims=[] if rng.random() < 0.5 else ["c"]
            ),
        ),
        min_relevant_docs=2,
    )
    state = await run_question(graph, "How often must I descale the machine?")
    assert state["status"] in {"answered", "insufficient", "out_of_scope"}
    assert state.get("retry_count", 0) <= 2 and state.get("regeneration_count", 0) <= 1
    assert llm.calls.count("RewrittenQuery") <= 2 and llm.calls.count("GeneratedAnswer") <= 2


async def test_never_grounded_and_nothing_relevant_alternating_terminates() -> None:
    flip = iter(bool(i % 2) for i in range(10_000))
    graph, llm, _ = build(
        handlers(
            DocGrade=lambda t: DocGrade(relevant=next(flip), reason=""),
            HallucinationGrade=lambda t: HallucinationGrade(
                grounded=False, unsupported_claims=["z"]
            ),
        ),
        min_relevant_docs=1,
    )
    state = await run_question(graph, "How often must I descale?")
    assert state["status"] == "insufficient"
    assert llm.calls.count("GeneratedAnswer") <= 2


async def test_empty_store_goes_through_rewrites_then_declines() -> None:
    graph, llm, store = build(handlers(), docs=[])
    state = await run_question(graph, "How often must I descale?")
    assert state["status"] == "insufficient" and state["retry_count"] == 2
    assert state["documents"] == [] and "DocGrade" not in llm.calls
    assert len(store.queries) == 3 and "GeneratedAnswer" not in llm.calls


async def test_german_refusal_text() -> None:
    graph, _, _ = build(handlers(), docs=[])
    state = await run_question(graph, "Wie oft muss ich die Maschine entkalken, bitte?")
    assert state["generation"] == prompts.INSUFFICIENT_INFO["de"]
    state_en = await run_question(graph, "How often must I descale the machine please?")
    assert state_en["generation"] == prompts.INSUFFICIENT_INFO["en"]
