"""Graph nodes: pure functions of (state, deps) returning partial state updates."""

import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from docrag import prompts
from docrag.config import Settings
from docrag.graph.state import GraphState
from docrag.llm import structured
from docrag.schemas import (
    Citation,
    DocGrade,
    GeneratedAnswer,
    HallucinationGrade,
    RewrittenQuery,
    RouteDecision,
    TraceStep,
)
from docrag.store import VectorStore


@dataclass
class Deps:
    """Injected dependencies; tests swap in fakes."""

    llm: BaseChatModel
    store: VectorStore
    settings: Settings


def question_language(text: str) -> str:
    from docrag.ingest.chunking import detect_language

    return "de" if detect_language(text) == "de" else "en"


def _loc(doc: Document) -> str:
    m = doc.metadata
    if "page" in m:
        return f"page {m['page']}"
    return f"section '{m.get('section', '?')}'"


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] ({d.metadata.get('source_file')}, {_loc(d)})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


def _query(state: GraphState) -> str:
    return state.get("rewritten_question") or state["question"]


async def route_question(
    state: GraphState, deps: Deps, config: RunnableConfig | None = None
) -> GraphState:
    chain = structured(deps.llm, RouteDecision)
    decision = await chain.ainvoke(
        [
            ("system", prompts.ROUTE_SYSTEM),
            ("human", prompts.ROUTE_HUMAN.format(question=state["question"])),
        ],
        config,
    )
    return {
        "route": decision.route,
        "trace": [TraceStep(node="route_question", detail=decision.route)],
    }


async def respond_out_of_scope(state: GraphState, deps: Deps) -> GraphState:
    lang = question_language(state["question"])
    return {
        "generation": prompts.OUT_OF_SCOPE[lang],
        "citations": [],
        "status": "out_of_scope",
        "trace": [TraceStep(node="respond_out_of_scope")],
    }


async def retrieve(state: GraphState, deps: Deps) -> GraphState:
    query = _query(state)
    docs = deps.store.search(query, k=deps.settings.top_k, mmr=deps.settings.use_mmr)
    return {
        "documents": docs,
        "trace": [
            TraceStep(
                node="retrieve",
                detail=f"{len(docs)} chunks",
                data={"query": query, "chunk_ids": [d.metadata.get("chunk_id") for d in docs]},
            )
        ],
    }


async def grade_documents(
    state: GraphState, deps: Deps, config: RunnableConfig | None = None
) -> GraphState:
    docs = state.get("documents", [])
    chain = structured(deps.llm, DocGrade)
    inputs: list[LanguageModelInput] = [
        [
            ("system", prompts.GRADE_SYSTEM),
            ("human", prompts.GRADE_HUMAN.format(question=state["question"], chunk=d.page_content)),
        ]
        for d in docs
    ]
    grades = await chain.abatch(inputs, config) if inputs else []
    relevant = [d for d, g in zip(docs, grades, strict=True) if g.relevant]
    return {
        "relevant_documents": relevant,
        "trace": [
            TraceStep(
                node="grade_documents",
                detail=f"{len(relevant)}/{len(docs)} relevant",
                data={"relevant_chunk_ids": [d.metadata.get("chunk_id") for d in relevant]},
            )
        ],
    }


async def rewrite_query(
    state: GraphState, deps: Deps, config: RunnableConfig | None = None
) -> GraphState:
    chain = structured(deps.llm, RewrittenQuery)
    out = await chain.ainvoke(
        [
            ("system", prompts.REWRITE_SYSTEM),
            (
                "human",
                prompts.REWRITE_HUMAN.format(question=state["question"], previous=_query(state)),
            ),
        ],
        config,
    )
    return {
        "rewritten_question": out.query,
        "retry_count": state.get("retry_count", 0) + 1,
        "trace": [TraceStep(node="rewrite_query", detail=out.query)],
    }


async def generate(
    state: GraphState, deps: Deps, config: RunnableConfig | None = None
) -> GraphState:
    docs = state.get("relevant_documents", [])
    feedback = state.get("feedback", [])
    system = prompts.GENERATE_SYSTEM
    if feedback:
        system += prompts.GENERATE_FEEDBACK.format(claims="\n".join(f"- {c}" for c in feedback))
    chain = structured(deps.llm, GeneratedAnswer)
    out = await chain.ainvoke(
        [
            ("system", system),
            (
                "human",
                prompts.GENERATE_HUMAN.format(
                    context=format_context(docs), question=state["question"]
                ),
            ),
        ],
        config,
    )
    update: GraphState = {
        "generation": out.answer,
        "cited_sources": out.cited_sources,
        "sufficient": out.sufficient,
        "trace": [
            TraceStep(
                node="generate",
                detail="regenerated with feedback" if feedback else "generated",
                data={"sufficient": out.sufficient},
            )
        ],
    }
    if feedback:
        update["regeneration_count"] = state.get("regeneration_count", 0) + 1
    return update


async def check_hallucination(
    state: GraphState, deps: Deps, config: RunnableConfig | None = None
) -> GraphState:
    if not state.get("sufficient", True):
        # An honest "not enough information" answer makes no factual claims to verify.
        return {
            "grounded": True,
            "feedback": [],
            "trace": [TraceStep(node="check_hallucination", detail="skipped (declined)")],
        }
    chain = structured(deps.llm, HallucinationGrade)
    grade = await chain.ainvoke(
        [
            ("system", prompts.HALLUCINATION_SYSTEM),
            (
                "human",
                prompts.HALLUCINATION_HUMAN.format(
                    context=format_context(state.get("relevant_documents", [])),
                    answer=state["generation"],
                ),
            ),
        ],
        config,
    )
    return {
        "grounded": grade.grounded,
        "feedback": [] if grade.grounded else grade.unsupported_claims or ["unspecified claim"],
        "trace": [
            TraceStep(
                node="check_hallucination",
                detail="grounded" if grade.grounded else "NOT grounded",
                data={"unsupported_claims": grade.unsupported_claims},
            )
        ],
    }


async def respond_insufficient_info(state: GraphState, deps: Deps) -> GraphState:
    lang = question_language(state["question"])
    return {
        "generation": prompts.INSUFFICIENT_INFO[lang],
        "citations": [],
        "status": "insufficient",
        "trace": [TraceStep(node="respond_insufficient_info")],
    }


_MARKER = re.compile(r"[ \t]*\[(\d+(?:\s*,\s*\d+)*)\]")


def strip_dangling_markers(text: str, valid_ids: set[int]) -> tuple[str, int]:
    """Remove [n] markers without a matching citation. Returns (clean_text, markers_removed).

    Plain string post-processing of already-generated text (no LLM involved); markers like
    "[1, 5]" keep only their valid ids.
    """
    removed = 0

    def fix(m: re.Match[str]) -> str:
        nonlocal removed
        ids = [int(x) for x in m.group(1).split(",")]
        keep = [i for i in ids if i in valid_ids]
        if len(keep) == len(ids):
            return m.group(0)
        removed += len(ids) - len(keep)
        return f" [{', '.join(map(str, keep))}]" if keep else ""

    return _MARKER.sub(fix, text), removed


async def format_answer(state: GraphState, deps: Deps) -> GraphState:
    docs = state.get("relevant_documents", [])
    citations: list[Citation] = []
    answered = state.get("sufficient", True)
    if answered:
        for n in sorted(set(state.get("cited_sources", []))):
            if 1 <= n <= len(docs):
                d = docs[n - 1]
                page = d.metadata.get("page", d.metadata.get("section"))
                citations.append(
                    Citation(
                        id=n,
                        source_file=str(d.metadata.get("source_file")),
                        page=page,
                        snippet=d.page_content[:200],
                    )
                )
    if not answered:  # drop the model's draft: it is typically long and uncited
        lang = question_language(state.get("question", ""))
        return {
            "generation": prompts.INSUFFICIENT_INFO[lang],
            "citations": [],
            "status": "insufficient",
            "trace": [
                TraceStep(
                    node="format_answer",
                    detail="declined: model reported insufficient context",
                )
            ],
        }
    text, removed = strip_dangling_markers(state.get("generation", ""), {c.id for c in citations})
    detail = f"{len(citations)} citations"
    if removed:
        detail += f", removed {removed} dangling marker(s)"
    return {
        "generation": text,
        "citations": citations,
        "status": "answered" if answered else "insufficient",
        "trace": [TraceStep(node="format_answer", detail=detail)],
    }


async def select_all_documents(state: GraphState, deps: Deps) -> GraphState:
    """Baseline-only: skip grading and pass every retrieved chunk to generation."""
    return {"relevant_documents": state.get("documents", [])}
