"""Structured-output models and API/response schemas."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints


class RouteDecision(BaseModel):
    route: Literal["docs", "out_of_scope"] = Field(
        description="'docs' if the question is about the technical documentation, "
        "'out_of_scope' for greetings or unrelated questions."
    )


class DocGrade(BaseModel):
    relevant: bool = Field(description="True if the chunk helps answer the question.")
    reason: str = Field(description="One short sentence explaining the decision.")


class HallucinationGrade(BaseModel):
    grounded: bool = Field(description="True if every claim is supported by the sources.")
    unsupported_claims: list[str] = Field(
        default_factory=list, description="Claims in the answer not supported by the sources."
    )


class RewrittenQuery(BaseModel):
    query: str = Field(description="The rewritten search query, in the original language.")


class GeneratedAnswer(BaseModel):
    answer: str = Field(description="Answer with inline [n] citation markers.")
    sufficient: bool = Field(
        default=True,
        description="False if the provided context does not contain the information needed.",
    )
    cited_sources: list[int] = Field(
        default_factory=list, description="Numbers of the sources cited in the answer."
    )


class CorrectnessGrade(BaseModel):
    correct: bool
    reason: str


class Citation(BaseModel):
    id: int
    source_file: str
    page: int | str | None = None
    snippet: str


class AskRequest(BaseModel):
    # whitespace is stripped first, then the length constraints apply to the stripped text
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ]


class TraceStep(BaseModel):
    node: str
    detail: str = ""
    data: dict[str, object] = Field(default_factory=dict)


Confidence = Literal["high", "medium", "low"]
DeclineReason = Literal["out_of_scope", "insufficient_info"]


class AskResponse(BaseModel):
    """API response.

    `declined` is True when the system refused (out of scope or insufficient information).
    `confidence` is a deterministic HEURISTIC derived from pipeline state (groundedness verdict,
    number of relevant chunks, rewrite/regeneration counts). It is NOT a calibrated probability.
    It is None for declined answers.
    `decline_reason` is None for answered questions; otherwise "out_of_scope" (the router rejected
    the question) or "insufficient_info" (retrieval/grounding could not support an answer).
    `language` is the detected question language; `hint` is a static localized suggestion, set only
    for "insufficient_info" declines.
    """

    answer: str
    declined: bool = False
    confidence: Confidence | None = None
    decline_reason: DeclineReason | None = None
    language: Literal["de", "en"] = "en"
    hint: str | None = None
    citations: list[Citation]
    trace: list[TraceStep]
