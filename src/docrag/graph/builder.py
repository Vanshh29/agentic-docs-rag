"""Compiles the agentic graph and offers run helpers."""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from docrag import prompts
from docrag.graph import edges, nodes
from docrag.graph.nodes import Deps, question_language
from docrag.graph.state import GraphState
from docrag.schemas import AskResponse, Confidence, DeclineReason

__all__ = ["Deps", "build_graph", "run_question", "to_response"]


def build_graph(deps: Deps) -> Any:
    g = StateGraph(GraphState)
    for name in (
        "route_question",
        "respond_out_of_scope",
        "retrieve",
        "grade_documents",
        "rewrite_query",
        "generate",
        "check_hallucination",
        "respond_insufficient_info",
        "format_answer",
    ):
        g.add_node(name, partial(getattr(nodes, name), deps=deps))

    g.add_edge(START, "route_question")
    g.add_conditional_edges(
        "route_question",
        edges.after_route,
        {"retrieve": "retrieve", "respond_out_of_scope": "respond_out_of_scope"},
    )
    g.add_edge("respond_out_of_scope", END)
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(
        "grade_documents",
        partial(edges.after_grade, settings=deps.settings),
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "respond_insufficient_info": "respond_insufficient_info",
        },
    )
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("generate", "check_hallucination")
    g.add_conditional_edges(
        "check_hallucination",
        partial(edges.after_hallucination_check, settings=deps.settings),
        {
            "format_answer": "format_answer",
            "generate": "generate",
            "respond_insufficient_info": "respond_insufficient_info",
        },
    )
    g.add_edge("respond_insufficient_info", END)
    g.add_edge("format_answer", END)
    return g.compile()


async def run_question(graph: Any, question: str, callbacks: list[Any] | None = None) -> GraphState:
    initial: GraphState = {
        "question": question,
        "retry_count": 0,
        "regeneration_count": 0,
        "feedback": [],
        "trace": [],
    }
    config = {"callbacks": callbacks or [], "recursion_limit": 40}
    result: GraphState = await graph.ainvoke(initial, config)
    return result


def assess_confidence(state: GraphState, min_relevant_docs: int = 2) -> Confidence | None:
    """Deterministic HEURISTIC confidence label; not a probability and not calibrated.

    Declined answers get None. Otherwise start at "high" and step down one level for each of:
    fewer than `min_relevant_docs` relevant chunks, at least one query rewrite, at least one
    regeneration. A failed or missing hallucination verdict (e.g. the baseline never checks)
    caps the result at "medium"; a failed verdict forces "low".
    """
    if state.get("status", "answered") != "answered":
        return None
    penalties = (
        (len(state.get("relevant_documents", [])) < min_relevant_docs)
        + (state.get("retry_count", 0) > 0)
        + (state.get("regeneration_count", 0) > 0)
    )
    levels: list[Confidence] = ["high", "medium", "low"]
    grounded = state.get("grounded")
    if grounded is False:
        return "low"
    if grounded is None:
        penalties = max(penalties, 1)
    return levels[min(penalties, 2)]


def assess_decline_reason(state: GraphState) -> DeclineReason | None:
    """Deterministic decline reason from the final state; None when the question was answered."""
    if state.get("status", "answered") == "answered":
        return None
    if state.get("status") == "out_of_scope" or state.get("route") == "out_of_scope":
        return "out_of_scope"
    return "insufficient_info"


def to_response(state: GraphState, min_relevant_docs: int = 2) -> AskResponse:
    reason = assess_decline_reason(state)
    lang = "de" if question_language(state.get("question", "")) == "de" else "en"
    return AskResponse(
        answer=state.get("generation", ""),
        language=lang,
        hint=prompts.DECLINE_HINT[lang] if reason == "insufficient_info" else None,
        declined=state.get("status", "answered") != "answered",
        decline_reason=reason,
        confidence=assess_confidence(state, min_relevant_docs),
        citations=state.get("citations", []),
        trace=state.get("trace", []),
    )
