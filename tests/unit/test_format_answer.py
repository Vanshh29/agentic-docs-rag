import pytest
from langchain_core.documents import Document

from docrag.graph.builder import assess_confidence, to_response
from docrag.graph.nodes import Deps, format_answer, strip_dangling_markers
from docrag.prompts import INSUFFICIENT_INFO
from tests.fakes import FakeStore, make_fake_llm, make_settings

DOCS = [
    Document(page_content=f"chunk {i}", metadata={"source_file": "m.pdf", "page": i})
    for i in (1, 2)
]


def deps() -> Deps:
    return Deps(make_fake_llm({}), FakeStore([]), make_settings())


async def test_format_answer_strips_markers_without_citation() -> None:
    out = await format_answer(
        {
            "generation": "Fact A [1]. Fact B [5]. Fact C [1, 7].",
            "relevant_documents": DOCS,
            "cited_sources": [1],
            "sufficient": True,
        },
        deps(),
    )
    assert [c.id for c in out["citations"]] == [1]
    assert out["generation"] == "Fact A [1]. Fact B. Fact C [1]."
    assert "dangling" in out["trace"][0].detail


@pytest.mark.parametrize(
    ("question", "lang"),
    [
        ("Meine Kaffeemaschine ist kaputt, es kommt kein Wasser raus. Was tun?", "de"),
        ("My coffee machine is broken and no water comes out. What should I do?", "en"),
    ],
)
async def test_format_answer_declined_replaces_uncited_draft(question: str, lang: str) -> None:
    draft = "Check the pump, descale it and call a technician for a very long detailed repair."
    out = await format_answer(
        {
            "question": question,
            "generation": draft,
            "relevant_documents": DOCS,
            "cited_sources": [],
            "sufficient": False,
        },
        deps(),
    )
    assert out["generation"] == INSUFFICIENT_INFO[lang]
    assert out["citations"] == [] and out["status"] == "insufficient"
    assert draft not in out["generation"]
    assert out["trace"][0].detail == "declined: model reported insufficient context"


def test_strip_keeps_valid_and_non_citation_brackets() -> None:
    text, n = strip_dangling_markers("See [1] and [a] and [2][9]", {1, 2})
    assert text == "See [1] and [a] and [2]" and n == 1
    assert strip_dangling_markers("no markers", {1}) == ("no markers", 0)


def test_confidence_heuristic() -> None:
    base = {"status": "answered", "grounded": True, "relevant_documents": DOCS}
    a = assess_confidence
    assert a(base) == "high"  # type: ignore[arg-type]
    assert a({**base, "retry_count": 1}) == "medium"  # type: ignore[typeddict-item]
    assert a({**base, "retry_count": 1, "regeneration_count": 1}) == "low"  # type: ignore[typeddict-item]
    assert a({**base, "relevant_documents": DOCS[:1]}) == "medium"  # type: ignore[typeddict-item]
    assert a({**base, "grounded": False}) == "low"  # type: ignore[typeddict-item]
    no_verdict = {k: v for k, v in base.items() if k != "grounded"}
    assert a(no_verdict) == "medium"  # type: ignore[arg-type]  # baseline: never verified
    assert a({"status": "insufficient"}) is None
    r = to_response({"status": "out_of_scope", "generation": "x"})
    assert r.declined is True and r.confidence is None


def test_decline_reason_by_path() -> None:
    answered = to_response({"status": "answered", "generation": "ok [1]"})
    assert answered.declined is False and answered.decline_reason is None
    oos = to_response({"status": "out_of_scope", "route": "out_of_scope", "generation": "x"})
    assert oos.declined is True and oos.decline_reason == "out_of_scope"
    insufficient = to_response({"status": "insufficient", "route": "docs", "generation": "x"})
    assert insufficient.declined is True and insufficient.decline_reason == "insufficient_info"
    # route missing but declined for another reason still maps to insufficient_info
    assert to_response({"status": "insufficient"}).decline_reason == "insufficient_info"
