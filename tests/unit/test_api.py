import json

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from docrag.api import create_app
from docrag.graph import Deps
from docrag.schemas import DocGrade, GeneratedAnswer, HallucinationGrade, RouteDecision
from tests.fakes import FakeStore, make_fake_llm, make_settings

DOCS = [
    Document(
        page_content=f"Descale every 3 months {i}",
        metadata={"source_file": "m.pdf", "page": 3, "chunk_id": str(i)},
    )
    for i in range(3)
]


def client() -> TestClient:
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
    deps = Deps(llm, FakeStore(list(DOCS)), make_settings(top_k=3))
    return TestClient(create_app(lambda: deps))


def test_health_and_ask() -> None:
    c = client()
    assert c.get("/health").json() == {"status": "ok"}
    r = c.post("/ask", json={"question": "How often to descale?"})
    body = r.json()
    assert r.status_code == 200
    assert body["answer"] == "Every 3 months [1]."
    assert body["citations"][0]["source_file"] == "m.pdf"
    assert body["trace"][0]["node"] == "route_question"


def test_frontend_and_status() -> None:
    c = client()
    page = c.get("/")
    assert page.status_code == 200 and "Docs Assistant" in page.text
    assert page.headers["cache-control"] == "no-cache"
    assert c.get("/status").json() == {
        "chunks": 3,
        "model": make_settings().llm_model,
        "demo_mode": False,
        "sources": ["m.pdf"],
    }


def test_ask_validates_input() -> None:
    assert client().post("/ask", json={"question": ""}).status_code == 422


def test_stream_emits_node_events_then_result() -> None:
    r = client().post("/ask/stream", json={"question": "How often to descale?"})
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    assert [e["node"] for e in events[:-1]][:2] == ["route_question", "retrieve"]
    assert events[-1]["event"] == "result" and events[-1]["answer"].startswith("Every")
    assert "decline_reason" in events[-1] and events[-1]["decline_reason"] is None


def test_stream_result_carries_decline_reason_out_of_scope() -> None:
    llm = make_fake_llm({RouteDecision: lambda t: RouteDecision(route="out_of_scope")})
    deps = Deps(llm, FakeStore(list(DOCS)), make_settings(top_k=3))
    c = TestClient(create_app(lambda: deps))
    r = c.post("/ask/stream", json={"question": "what is he doing?"})
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    assert events[-1]["declined"] is True
    assert events[-1]["decline_reason"] == "out_of_scope"
    assert c.post("/ask", json={"question": "hi"}).json()["decline_reason"] == "out_of_scope"


def test_ingest_rejects_unsupported() -> None:
    r = client().post("/ingest", files={"file": ("x.txt", b"hi")})
    assert r.status_code == 400
