"""Public-demo safeguards: read-only mode, request limits and startup preload."""

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from docrag.api import create_app
from docrag.graph import Deps
from docrag.ratelimit import AskLimiter
from docrag.schemas import DocGrade, GeneratedAnswer, HallucinationGrade, RouteDecision
from tests.fakes import FakeStore, make_fake_llm, make_settings

DOC = Document(
    page_content="Descale every 3 months",
    metadata={"source_file": "m.pdf", "page": 3, "chunk_id": "0"},
)
ASK = {"question": "How often to descale?"}


def client(store: FakeStore | None = None, preload: bool = False, **settings: Any) -> TestClient:
    llm = make_fake_llm(
        {
            RouteDecision: lambda t: RouteDecision(route="docs"),
            DocGrade: lambda t: DocGrade(relevant=True, reason=""),
            GeneratedAnswer: lambda t: GeneratedAnswer(
                answer="Every 3 months [1].", sufficient=True, cited_sources=[1]
            ),
            HallucinationGrade: lambda t: HallucinationGrade(grounded=True),
        }
    )
    deps = Deps(llm, store or FakeStore([DOC]), make_settings(top_k=3, **settings))
    return TestClient(create_app(lambda: deps, preload=preload))


# --- demo mode -------------------------------------------------------------------------


def test_demo_mode_blocks_upload_and_delete() -> None:
    c = client(demo_mode=True)
    up = c.post("/ingest", files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")})
    assert up.status_code == 403 and up.json()["code"] == "demo_mode"
    rm = c.delete("/documents/m.pdf")
    assert rm.status_code == 403 and rm.json()["code"] == "demo_mode"
    assert [d["source_file"] for d in c.get("/documents").json()["documents"]] == ["m.pdf"]


def test_demo_mode_still_answers_questions_and_reports_the_flag() -> None:
    c = client(demo_mode=True)
    assert c.post("/ask", json=ASK).status_code == 200
    assert c.get("/status").json()["demo_mode"] is True


def test_normal_mode_keeps_uploads_and_delete_working() -> None:
    c = client()
    assert c.get("/status").json()["demo_mode"] is False
    assert c.delete("/documents/m.pdf").status_code == 200


# --- request limits --------------------------------------------------------------------


def test_per_minute_limit_returns_429_with_retry_after() -> None:
    c = client(ask_rate_per_minute=2)
    assert [c.post("/ask", json=ASK).status_code for _ in range(2)] == [200, 200]
    third = c.post("/ask", json=ASK)
    assert third.status_code == 429
    assert third.json()["code"] == "too_many_requests"
    assert int(third.headers["retry-after"]) >= 1


def test_limit_also_covers_the_streaming_endpoint() -> None:
    c = client(ask_rate_per_minute=1)
    assert c.post("/ask/stream", json=ASK).status_code == 200
    assert c.post("/ask/stream", json=ASK).status_code == 429


def test_daily_limit_is_global_and_has_its_own_code() -> None:
    c = client(ask_daily_limit=1)
    assert c.post("/ask", json=ASK).status_code == 200
    blocked = c.post("/ask", json=ASK)
    assert blocked.status_code == 429 and blocked.json()["code"] == "daily_limit"


def test_limits_are_off_by_default() -> None:
    c = client()
    assert all(c.post("/ask", json=ASK).status_code == 200 for _ in range(5))


def test_invalid_questions_do_not_consume_the_budget() -> None:
    c = client(ask_rate_per_minute=1)
    assert c.post("/ask", json={"question": ""}).status_code == 422
    assert c.post("/ask", json=ASK).status_code == 200


# --- limiter unit ----------------------------------------------------------------------


def test_window_slides_and_keys_are_independent() -> None:
    now = [1_000.0]
    lim = AskLimiter(per_minute=2, daily=0, clock=lambda: now[0])
    assert lim.check("a") is None and lim.check("a") is None
    blocked = lim.check("a")
    assert blocked is not None and blocked.code == "too_many_requests"
    assert 1 <= blocked.retry_after <= 60
    assert lim.check("b") is None  # another client is unaffected
    now[0] += 61
    assert lim.check("a") is None  # the window has moved on


def test_daily_counter_resets_at_the_next_utc_day() -> None:
    now = [86_400 * 10 + 100.0]
    lim = AskLimiter(per_minute=0, daily=1, clock=lambda: now[0])
    assert lim.check("a") is None
    blocked = lim.check("b")
    assert blocked is not None and blocked.code == "daily_limit"
    assert blocked.retry_after == 86_400 - 100
    now[0] = 86_400 * 11 + 1
    assert lim.check("b") is None


def test_rejected_requests_are_not_counted() -> None:
    now = [0.0]
    lim = AskLimiter(per_minute=1, daily=2, clock=lambda: now[0])
    assert lim.check("a") is None
    for _ in range(5):
        assert lim.check("a") is not None  # rejected by the minute window
    now[0] += 61
    assert lim.check("a") is None  # still within the daily budget of 2


# --- startup preload -------------------------------------------------------------------


def _wait_for(cond: Any, timeout: float = 3.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_preload_indexes_the_samples_when_the_index_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []
    monkeypatch.setattr("docrag.api.ingest_path", lambda p, store, s: calls.append(p))
    with client(FakeStore([]), preload=True, samples_dir=tmp_path):
        assert _wait_for(lambda: calls == [tmp_path])


def test_preload_leaves_an_existing_index_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []
    monkeypatch.setattr("docrag.api.ingest_path", lambda p, store, s: calls.append(p))
    with client(FakeStore([DOC]), preload=True, samples_dir=tmp_path) as c:
        assert _wait_for(lambda: c.get("/status").json()["chunks"] == 1)
        time.sleep(0.2)
    assert calls == []
