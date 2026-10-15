"""Recorded-answers mode: a public demo that never calls a model."""

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

import docrag.api as api_module
from docrag.api import create_app
from docrag.graph import Deps
from docrag.replay import NoEmbeddings, NullChatModel, ReplayBook, normalize
from tests.fakes import FakeStore, make_fake_llm, make_settings

RECORDED = {
    "question": "How long does the machine need to heat up?",
    "response": {
        "answer": "About 25 seconds [1].",
        "declined": False,
        "confidence": "high",
        "language": "en",
        "citations": [
            {"id": 1, "source_file": "espresso.pdf", "page": 2, "snippet": "Heat-up: 25 seconds"}
        ],
        "trace": [
            {"node": "route_question", "detail": "docs", "data": {}},
            {"node": "retrieve", "detail": "6 chunks", "data": {}},
            {"node": "format_answer", "detail": "1 citations", "data": {}},
        ],
    },
}
DOC = Document(
    page_content="x", metadata={"source_file": "espresso.pdf", "page": 2, "chunk_id": "0"}
)


def book_file(tmp_path: Path, entries: list[dict[str, Any]] | None = None) -> Path:
    path = tmp_path / "answers.json"
    path.write_text(json.dumps({"answers": entries if entries is not None else [RECORDED]}))
    return path


def client(tmp_path: Path, **settings: Any) -> TestClient:
    s = make_settings(demo_replay=True, demo_answers_file=book_file(tmp_path), **settings)
    deps = Deps(make_fake_llm({}), FakeStore([DOC]), s)  # no handlers: any model call would fail
    return TestClient(create_app(lambda: deps, preload=False))


@pytest.fixture(autouse=True)
def no_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_module, "REPLAY_STEP_DELAY_S", 0)


# --- matching ---------------------------------------------------------------------------


def test_normalize_ignores_case_spacing_and_trailing_punctuation() -> None:
    assert normalize("  How LONG does   the machine\nneed to heat up?? ") == normalize(
        "how long does the machine need to heat up"
    )
    assert normalize("Wie oft muss der HEPA-Filter H13 gewechselt werden?") == (
        "wie oft muss der hepa-filter h13 gewechselt werden"
    )


def test_book_matches_normalized_questions_and_lists_examples(tmp_path: Path) -> None:
    book = ReplayBook.load(book_file(tmp_path))
    assert book.get("HOW long does the machine need to heat up ?") is not None
    assert book.get("What is the price?") is None
    assert book.examples() == [
        {
            "question": "How long does the machine need to heat up?",
            "language": "en",
            "source": "espresso.pdf",
        }
    ]


def test_a_broken_answers_file_fails_with_a_clear_message(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError, match="answers file"):
        ReplayBook.load(bad)


# --- the API in replay mode ---------------------------------------------------------------


def test_recorded_question_is_answered_without_any_model(tmp_path: Path) -> None:
    r = client(tmp_path).post(
        "/ask", json={"question": "how long does the machine need to heat up"}
    )
    assert r.status_code == 200
    assert r.json()["answer"] == "About 25 seconds [1]."
    assert r.json()["citations"][0]["source_file"] == "espresso.pdf"


def test_unrecorded_question_gets_a_clear_non_retryable_error(tmp_path: Path) -> None:
    r = client(tmp_path).post("/ask", json={"question": "What is the price?"})
    assert r.status_code == 422 and r.json()["code"] == "replay_only"
    s = client(tmp_path).post("/ask/stream", json={"question": "What is the price?"})
    assert s.status_code == 422 and s.json()["code"] == "replay_only"


def test_stream_replays_the_recorded_steps_then_the_result(tmp_path: Path) -> None:
    r = client(tmp_path).post(
        "/ask/stream", json={"question": "How long does the machine need to heat up?"}
    )
    events = [json.loads(ln[6:]) for ln in r.text.splitlines() if ln.startswith("data: ")]
    assert [e["event"] for e in events] == ["node", "node", "node", "result"]
    assert [e["node"] for e in events[:3]] == ["route_question", "retrieve", "format_answer"]
    assert events[-1]["answer"] == "About 25 seconds [1]."


def test_replay_mode_is_read_only_and_reports_itself(tmp_path: Path) -> None:
    c = client(tmp_path)
    assert c.delete("/documents/espresso.pdf").json()["code"] == "demo_mode"
    up = c.post("/ingest", files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")})
    assert up.status_code == 403
    status = c.get("/status").json()
    assert status["replay"] is True and status["examples"][0]["source"] == "espresso.pdf"


def test_normal_mode_reports_no_replay(tmp_path: Path) -> None:
    deps = Deps(make_fake_llm({}), FakeStore([DOC]), make_settings())
    status = TestClient(create_app(lambda: deps, preload=False)).get("/status").json()
    assert status["replay"] is False and status["examples"] == []


def test_replay_requests_still_count_against_the_rate_limit(tmp_path: Path) -> None:
    c = client(tmp_path, ask_rate_per_minute=1)
    q = {"question": "How long does the machine need to heat up?"}
    assert c.post("/ask", json=q).status_code == 200
    assert c.post("/ask", json=q).status_code == 429


def test_missing_answers_file_does_not_take_the_app_down(tmp_path: Path) -> None:
    s = make_settings(demo_replay=True, demo_answers_file=tmp_path / "nope.json")
    deps = Deps(make_fake_llm({}), FakeStore([DOC]), s)
    c = TestClient(create_app(lambda: deps, preload=False))
    assert c.get("/status").json()["examples"] == []
    assert c.post("/ask", json={"question": "anything"}).json()["code"] == "replay_only"


# --- stand-ins ---------------------------------------------------------------------------


def test_stand_ins_refuse_to_do_real_work() -> None:
    with pytest.raises(RuntimeError, match="replay"):
        NoEmbeddings().embed_query("x")
    with pytest.raises(Exception, match="unavailable"):
        NullChatModel().invoke("x")
