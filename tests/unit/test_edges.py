from docrag.graph import edges
from tests.fakes import make_settings

S = make_settings(min_relevant_docs=2, max_rewrites=2, max_regenerations=1)


def test_after_route() -> None:
    assert edges.after_route({"route": "docs"}) == "retrieve"
    assert edges.after_route({"route": "out_of_scope"}) == "respond_out_of_scope"


def test_after_grade() -> None:
    docs = [object(), object()]
    assert edges.after_grade({"relevant_documents": docs}, S) == "generate"  # type: ignore[typeddict-item]
    assert edges.after_grade({"relevant_documents": [], "retry_count": 1}, S) == "rewrite_query"
    assert (
        edges.after_grade({"relevant_documents": [], "retry_count": 2}, S)
        == "respond_insufficient_info"
    )


def test_after_hallucination_check() -> None:
    f = edges.after_hallucination_check
    assert f({"grounded": True}, S) == "format_answer"
    assert f({"grounded": False, "regeneration_count": 0}, S) == "generate"
    assert f({"grounded": False, "regeneration_count": 1}, S) == "respond_insufficient_info"
