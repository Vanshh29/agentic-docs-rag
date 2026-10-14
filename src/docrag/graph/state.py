"""Typed graph state."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document

from docrag.schemas import Citation, TraceStep

Status = Literal["answered", "insufficient", "out_of_scope"]


class GraphState(TypedDict, total=False):
    question: str
    rewritten_question: str
    route: Literal["docs", "out_of_scope"]
    documents: list[Document]
    relevant_documents: list[Document]
    generation: str
    cited_sources: list[int]
    sufficient: bool
    citations: list[Citation]
    retry_count: int
    regeneration_count: int
    grounded: bool
    feedback: list[str]  # unsupported claims fed back into generate
    status: Status
    trace: Annotated[list[TraceStep], operator.add]
