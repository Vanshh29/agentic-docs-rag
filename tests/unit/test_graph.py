"""Graph behaviour with a fake LLM: routing, loops, termination, citations."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from docrag.graph import Deps, build_baseline_graph, build_graph, run_question, to_response
from docrag.prompts import DECLINE_HINT, INSUFFICIENT_INFO
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
        page_content="Descale the machine every 3 months. The descale light blinks.",
        metadata={"source_file": "m.pdf", "page": 3, "chunk_id": "a"},
    ),
    Document(
        page_content="The descaling programme takes about 20 minutes.",
        metadata={"source_file": "m.pdf", "page": 3, "chunk_id": "b"},
    ),
    Document(
        page_content="Warranty is 24 months.",
        metadata={"source_file": "m.pdf", "page": 4, "chunk_id": "c"},
    ),
]
SETTINGS = make_settings(top_k=3, min_relevant_docs=2, max_rewrites=2, max_regenerations=1)


def build(handlers, store=None):  # type: ignore[no-untyped-def]
    llm = make_fake_llm(handlers)
    store = store or FakeStore(list(DOCS))
    return build_graph(Deps(llm, store, SETTINGS)), llm, store


def grade_by_keyword(text: str) -> DocGrade:
    return DocGrade(relevant="descal" in text.split("Chunk:")[-1].lower(), reason="kw")


def answer_ok(_: str) -> GeneratedAnswer:
    return GeneratedAnswer(answer="Every 3 months [1].", sufficient=True, cited_sources=[1])


async def test_happy_path_with_citations() -> None:
    graph, llm, _ = build(
        {
            RouteDecision: lambda t: RouteDecision(route="docs"),
            DocGrade: grade_by_keyword,
            GeneratedAnswer: answer_ok,
            HallucinationGrade: lambda t: HallucinationGrade(grounded=True),
        }
    )
    state = await run_question(graph, "How often must I descale the machine?")
    resp = to_response(state)
    assert state["status"] == "answered"
    assert resp.answer == "Every 3 months [1]."
    assert [c.id for c in resp.citations] == [1]
    assert resp.citations[0].source_file == "m.pdf" and resp.citations[0].page == 3
    assert [s.node for s in resp.trace] == [
        "route_question",
        "retrieve",
        "grade_documents",
        "generate",
        "check_hallucination",
        "format_answer",
    ]
    assert llm.calls.count("DocGrade") == 3  # one per retrieved chunk


async def test_out_of_scope_skips_retrieval() -> None:
    graph, llm, store = build({RouteDecision: lambda t: RouteDecision(route="out_of_scope")})
    state = await run_question(graph, "Hallo, wie geht's dir heute?")
    assert state["status"] == "out_of_scope"
    assert store.queries == []
    assert "Dokumentation" in state["generation"]  # answered in German


class VocabMismatchStore(FakeStore):
    """Original wording retrieves the wrong chunk; the rewritten query finds the right ones."""

    def search(self, query: str, k: int, mmr: bool = False) -> list[Document]:
        self.queries.append(query)
        return DOCS[:2] if "descaling interval" in query else DOCS[2:]


async def test_rewrite_then_success() -> None:
    store = VocabMismatchStore(list(DOCS))
    graph, llm, _ = build(
        {
            RouteDecision: lambda t: RouteDecision(route="docs"),
            RewrittenQuery: lambda t: RewrittenQuery(query="descaling interval"),
            DocGrade: grade_by_keyword,
            GeneratedAnswer: answer_ok,
            HallucinationGrade: lambda t: HallucinationGrade(grounded=True),
        },
        store,
    )
    state = await run_question(graph, "limescale thing how often?")
    assert state["retry_count"] == 1
    assert llm.calls.count("RewrittenQuery") == 1  # first retrieve yielded < MIN_RELEVANT_DOCS
    assert [s.node for s in state["trace"]].count("rewrite_query") == 1
    assert store.queries == ["limescale thing how often?", "descaling interval"]
    assert state["status"] == "answered"


async def test_terminates_when_nothing_relevant() -> None:
    graph, llm, store = build(
        {
            RouteDecision: lambda t: RouteDecision(route="docs"),
            RewrittenQuery: lambda t: RewrittenQuery(query="again"),
            DocGrade: lambda t: DocGrade(relevant=False, reason="no"),
        }
    )
    state = await run_question(graph, "What is the airspeed of a swallow?")
    assert state["status"] == "insufficient"
    assert state["retry_count"] == SETTINGS.max_rewrites
    assert len(store.queries) == SETTINGS.max_rewrites + 1  # initial + 2 rewrites
    assert llm.calls.count("RewrittenQuery") == 2


async def test_regenerates_once_with_feedback_then_passes() -> None:
    seen: list[str] = []

    def generate(text: str) -> GeneratedAnswer:
        seen.append(text)
        return GeneratedAnswer(answer="x [1]", sufficient=True, cited_sources=[1])

    verdicts = iter(
        [
            HallucinationGrade(grounded=False, unsupported_claims=["It takes 5 minutes"]),
            HallucinationGrade(grounded=True),
        ]
    )
    graph, _, _ = build(
        {
            RouteDecision: lambda t: RouteDecision(route="docs"),
            DocGrade: grade_by_keyword,
            GeneratedAnswer: generate,
            HallucinationGrade: lambda t: next(verdicts),
        }
    )
    state = await run_question(graph, "How long does descaling take?")
    assert state["status"] == "answered"
    assert state["regeneration_count"] == 1
    assert "It takes 5 minutes" in seen[1] and "It takes 5 minutes" not in seen[0]


async def test_terminates_when_never_grounded() -> None:
    graph, llm, _ = build(
        {
            RouteDecision: lambda t: RouteDecision(route="docs"),
            DocGrade: grade_by_keyword,
            GeneratedAnswer: answer_ok,
            HallucinationGrade: lambda t: HallucinationGrade(
                grounded=False, unsupported_claims=["bad"]
            ),
        }
    )
    state = await run_question(graph, "How often must I descale?")
    assert state["status"] == "insufficient"
    assert llm.calls.count("GeneratedAnswer") == 1 + SETTINGS.max_regenerations
    assert llm.calls.count("HallucinationGrade") == 2


async def test_declined_answer_has_no_citations_and_skips_judge() -> None:
    graph, llm, _ = build(
        {
            RouteDecision: lambda t: RouteDecision(route="docs"),
            DocGrade: grade_by_keyword,
            GeneratedAnswer: lambda t: GeneratedAnswer(
                answer="The sources do not say.", sufficient=False, cited_sources=[1]
            ),
        }
    )
    state = await run_question(graph, "How often must I descale?")
    assert state["status"] == "insufficient"
    assert state["citations"] == []
    assert "HallucinationGrade" not in llm.calls


async def test_baseline_uses_two_llm_calls_only() -> None:
    llm = make_fake_llm({GeneratedAnswer: answer_ok})
    graph = build_baseline_graph(Deps(llm, FakeStore(list(DOCS)), SETTINGS))
    state = await run_question(graph, "How often must I descale?")
    assert llm.calls == ["GeneratedAnswer"]
    assert state["status"] == "answered"
    assert len(state["documents"]) == 3


@pytest.mark.live
async def test_live_smoke() -> None:  # pragma: no cover - requires API key and an index
    pytest.skip("run manually with a configured provider")


async def test_declined_german_question_graph_level() -> None:
    q = "Meine Kaffeemaschine ist kaputt, es kommt kein Wasser raus. Was tun?"
    graph, _, _ = build(
        {
            RouteDecision: lambda t: RouteDecision(route="docs"),
            DocGrade: lambda t: DocGrade(relevant=True, reason=""),
            GeneratedAnswer: lambda t: GeneratedAnswer(
                answer="Pruefen Sie die Pumpe und rufen Sie den Service an.",
                sufficient=False,
                cited_sources=[],
            ),
        }
    )
    resp = to_response(await run_question(graph, q), 1)
    assert resp.declined is True and resp.language == "de"
    assert resp.decline_reason == "insufficient_info"
    assert resp.hint == DECLINE_HINT["de"]
    assert resp.answer == INSUFFICIENT_INFO["de"] and resp.citations == []
