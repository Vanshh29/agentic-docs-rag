"""Naive RAG for comparison: retrieve -> generate -> format (no grading, rewriting or checks)."""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from docrag.graph import nodes
from docrag.graph.nodes import Deps
from docrag.graph.state import GraphState


def build_baseline_graph(deps: Deps) -> Any:
    g = StateGraph(GraphState)
    for name in ("retrieve", "select_all_documents", "generate", "format_answer"):
        g.add_node(name, partial(getattr(nodes, name), deps=deps))
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "select_all_documents")
    g.add_edge("select_all_documents", "generate")
    g.add_edge("generate", "format_answer")
    g.add_edge("format_answer", END)
    return g.compile()
