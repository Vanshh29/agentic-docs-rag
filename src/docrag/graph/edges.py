"""Routing functions for conditional edges."""

from __future__ import annotations

from typing import Literal

from docrag.config import Settings
from docrag.graph.state import GraphState


def after_route(state: GraphState) -> Literal["retrieve", "respond_out_of_scope"]:
    return "retrieve" if state["route"] == "docs" else "respond_out_of_scope"


def after_grade(
    state: GraphState, settings: Settings
) -> Literal["generate", "rewrite_query", "respond_insufficient_info"]:
    if len(state.get("relevant_documents", [])) >= settings.min_relevant_docs:
        return "generate"
    if state.get("retry_count", 0) < settings.max_rewrites:
        return "rewrite_query"
    return "respond_insufficient_info"


def after_hallucination_check(
    state: GraphState, settings: Settings
) -> Literal["format_answer", "generate", "respond_insufficient_info"]:
    if state.get("grounded", False):
        return "format_answer"
    if state.get("regeneration_count", 0) < settings.max_regenerations:
        return "generate"
    return "respond_insufficient_info"
