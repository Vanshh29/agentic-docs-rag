import pytest
from eval.run_eval import ItemResult, _matches, aggregate, judge_item, render_markdown

from docrag.schemas import Citation, CorrectnessGrade, HallucinationGrade
from tests.fakes import make_fake_llm


def test_matches_page_and_section() -> None:
    assert _matches({"source_file": "a.pdf", "page": 2}, {"source_file": "a.pdf", "page": 2})
    assert not _matches({"source_file": "a.pdf", "page": 3}, {"source_file": "a.pdf", "page": 2})
    assert _matches(
        {"source_file": "a.docx", "section": "X"}, {"source_file": "a.docx", "section": "X"}
    )
    assert not _matches({"source_file": "b.pdf", "page": 2}, {"source_file": "a.pdf", "page": 2})


async def test_judge_item_scores_answerable_and_unanswerable() -> None:
    judge = make_fake_llm(
        {
            CorrectnessGrade: lambda t: CorrectnessGrade(correct=True, reason=""),
            HallucinationGrade: lambda t: HallucinationGrade(grounded=True),
        }
    )
    item = {
        "question": "q",
        "reference_answer": "r",
        "category": "lookup",
        "expected_sources": [{"source_file": "a.pdf", "page": 2}],
    }
    state = {
        "documents": [],
        "relevant_documents": [],
        "citations": [
            Citation(id=1, source_file="a.pdf", page=2, snippet="s"),
            Citation(id=2, source_file="a.pdf", page=9, snippet="s"),
        ],
    }
    res = ItemResult("q", "lookup", "en", status="answered", answer="a")
    await judge_item(judge, item, state, res)
    assert res.correct is True and res.citation_precision == 0.5 and res.retrieval_hit is False

    res2 = ItemResult("q", "unanswerable", "en", status="insufficient")
    await judge_item(judge, {**item, "category": "unanswerable", "expected_sources": []}, {}, res2)
    assert res2.refused is True


def test_aggregate_and_markdown() -> None:
    rs = [
        ItemResult(
            "a",
            "lookup",
            "en",
            status="answered",
            correct=True,
            retrieval_hit=True,
            citation_precision=1.0,
            latency_s=2.0,
            llm_calls=2,
        ),
        ItemResult(
            "b",
            "unanswerable",
            "en",
            status="insufficient",
            refused=True,
            latency_s=1.0,
            llm_calls=4,
        ),
    ]
    agg = aggregate(rs)
    assert agg["answer_correctness"] == 1.0 and agg["correct_refusal_rate"] == 1.0
    assert agg["avg_llm_calls"] == 3.0
    meta = {
        "timestamp": "t",
        "n_items": 2,
        "n_unanswerable": 1,
        "llm_model": "m",
        "judge_model": "j",
        "provider": "p",
        "embedding_model": "e",
        "top_k": 6,
        "judge_prompt_version": "v1",
    }
    md = render_markdown(meta, {"baseline": agg, "agentic": agg})
    assert "| Answer correctness" in md and "human review" in md


def test_hit_at_k_respects_rank_order() -> None:
    from eval.run_eval import hit_at_k
    from langchain_core.documents import Document

    def d(page: int) -> Document:
        return Document(page_content="x", metadata={"source_file": "a.pdf", "page": page})

    docs = [d(9), d(8), d(7), d(2), d(6)]
    exp = [{"source_file": "a.pdf", "page": 2}]
    assert not hit_at_k(docs, exp, 1) and not hit_at_k(docs, exp, 3)
    assert hit_at_k(docs, exp, 4) and hit_at_k(docs, exp, 5)
    assert hit_at_k([d(2)], exp, 1)
    assert not hit_at_k([], exp, 3)
    assert hit_at_k(docs, [{"source_file": "a.pdf", "page": 9}, *exp], 1)  # any expected counts


async def test_judge_item_sets_hit_at_1_and_3_and_aggregates() -> None:
    from langchain_core.documents import Document

    judge = make_fake_llm(
        {
            CorrectnessGrade: lambda t: CorrectnessGrade(correct=True, reason=""),
            HallucinationGrade: lambda t: HallucinationGrade(grounded=True),
        }
    )

    def d(page: int) -> Document:
        return Document(page_content="x", metadata={"source_file": "a.pdf", "page": page})

    def item(page: int) -> dict[str, object]:
        return {
            "question": "q",
            "reference_answer": "r",
            "category": "lookup",
            "expected_sources": [{"source_file": "a.pdf", "page": page}],
        }

    results = []
    for page in (1, 3, 5):  # rank 1, rank 3, rank 5 in docs below
        res = ItemResult("q", "lookup", "en", status="insufficient")
        await judge_item(judge, item(page), {"documents": [d(1), d(2), d(3), d(4), d(5)]}, res)
        results.append(res)
    assert [(r.hit_at_1, r.hit_at_3, r.retrieval_hit) for r in results] == [
        (True, True, True),
        (False, True, True),
        (False, False, True),
    ]
    agg = aggregate(results)
    assert agg["hit_rate_at_1"] == pytest.approx(1 / 3)
    assert agg["hit_rate_at_3"] == pytest.approx(2 / 3)
    assert agg["hit_rate_at_k"] == 1.0
    meta = {
        "timestamp": "t", "n_items": 3, "n_unanswerable": 0, "llm_model": "m", "judge_model": "j",
        "provider": "p", "embedding_model": "e", "top_k": 5, "judge_prompt_version": "v1",
    }  # fmt: skip
    old = {k: v for k, v in agg.items() if k not in {"hit_rate_at_1", "hit_rate_at_3"}}
    md = render_markdown(meta, {"new": agg, "old (--reuse)": old})
    assert "hit rate@1" in md and "hit rate@3" in md and "n/a" in md
