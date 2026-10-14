"""API error handling, source serving and response heuristics (offline, fake LLM)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from docrag.api import GENERIC_ERROR, create_app
from docrag.graph import Deps
from docrag.schemas import DocGrade, GeneratedAnswer, HallucinationGrade, RouteDecision
from tests.fakes import FakeStore, make_fake_llm, make_settings
from tests.helpers import make_pdf

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"


def _ok_handlers() -> dict:  # type: ignore[type-arg]
    return {
        RouteDecision: lambda t: RouteDecision(route="docs"),
        DocGrade: lambda t: DocGrade(relevant=True, reason=""),
        GeneratedAnswer: lambda t: GeneratedAnswer(
            answer="Every 3 months [1].", sufficient=True, cited_sources=[1]
        ),
        HallucinationGrade: lambda t: HallucinationGrade(grounded=True),
    }


def _boom(_: str) -> RouteDecision:
    raise RuntimeError("provider down")


def make_client(tmp_path: Path, handlers: dict | None = None, docs: list[Document] | None = None):  # type: ignore[no-untyped-def,type-arg]
    settings = make_settings(
        top_k=3, docs_dir=tmp_path / "raw", samples_dir=tmp_path / "samples", min_relevant_docs=1
    )
    store = FakeStore(docs if docs is not None else [])
    deps = Deps(make_fake_llm(handlers or _ok_handlers()), store, settings)
    return TestClient(create_app(lambda: deps)), store, settings


def test_ask_returns_502_json_on_llm_failure(tmp_path: Path) -> None:
    c, _, _ = make_client(tmp_path, {**_ok_handlers(), RouteDecision: _boom})
    r = c.post("/ask", json={"question": "hello?"})
    assert r.status_code == 502
    assert r.json()["detail"] == GENERIC_ERROR and "provider down" not in r.text


def test_stream_emits_error_event_then_ends(tmp_path: Path) -> None:
    c, _, _ = make_client(tmp_path, {**_ok_handlers(), RouteDecision: _boom})
    r = c.post("/ask/stream", json={"question": "hello?"})
    assert r.status_code == 200
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    assert events[-1]["event"] == "error" and events[-1]["message"] == GENERIC_ERROR
    assert "provider down" not in r.text
    assert all(e["event"] != "result" for e in events)


def test_ask_validation_whitespace_and_length(tmp_path: Path) -> None:
    c, _, _ = make_client(tmp_path)
    assert c.post("/ask", json={"question": "   "}).status_code == 422
    assert c.post("/ask", json={"question": "x" * 2001}).status_code == 422


def test_ingest_corrupt_upload_returns_422_with_results(tmp_path: Path) -> None:
    c, store, settings = make_client(tmp_path)
    r = c.post("/ingest", files={"file": ("bad.pdf", b"not a pdf")})
    assert r.status_code == 422
    body = r.json()
    assert "bad.pdf" in body["detail"]
    assert body["results"][0]["file"] == "bad.pdf" and body["results"][0]["status"] == "failed"
    assert store.count() == 0 and not (settings.docs_dir / "bad.pdf").exists()


@pytest.mark.parametrize("name", ["", ".env.pdf.txt", "x.exe"])
def test_ingest_rejects_bad_names_with_json(tmp_path: Path, name: str) -> None:
    c, _, _ = make_client(tmp_path)
    r = c.post("/ingest", files={"file": (name, b"x")})
    assert r.status_code in (400, 422) and "detail" in r.json()


def test_ingest_success_persists_and_sources_serves(tmp_path: Path) -> None:
    c, store, settings = make_client(tmp_path)
    src = make_pdf(tmp_path / "up.pdf", ["Descale every 3 months"])
    r = c.post("/ingest", files={"file": ("up.pdf", src.read_bytes())})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["files"] == 1 and body["chunks"] >= 1 and body["failed"] == 0
    assert body["results"] == [{"file": "up.pdf", "status": "ok", "chunks": body["chunks"]}]
    assert (settings.docs_dir / "up.pdf").is_file()
    got = c.get("/sources/up.pdf")
    assert got.status_code == 200 and got.content == src.read_bytes()


def test_sources_serves_ingested_sample_and_blocks_traversal(tmp_path: Path) -> None:
    c, store, settings = make_client(
        tmp_path,
        docs=[
            Document(page_content="x", metadata={"source_file": "espresso_machine_manual_en.pdf"})
        ],
    )
    settings.samples_dir.mkdir()
    shutil.copy(SAMPLES / "espresso_machine_manual_en.pdf", settings.samples_dir)
    (tmp_path / "secret.txt").write_text("secret")
    assert c.get("/sources/espresso_machine_manual_en.pdf").status_code == 200
    for bad in [
        "../secret.txt",
        "..%2Fsecret.txt",
        "%2e%2e/secret.txt",
        "%2e%2e%2fsecret.txt",
        "/etc/passwd",
        "%2Fetc%2Fpasswd",
        "..\\secret.txt",
        "not_ingested.pdf",
        "raw/../../secret.txt",
    ]:
        assert c.get(f"/sources/{bad}").status_code == 404, bad


def test_sources_whitelisted_name_without_file_is_404(tmp_path: Path) -> None:
    c, _, _ = make_client(
        tmp_path, docs=[Document(page_content="x", metadata={"source_file": "ghost.pdf"})]
    )
    assert c.get("/sources/ghost.pdf").status_code == 404


def test_response_has_declined_and_confidence(tmp_path: Path) -> None:
    docs = [
        Document(page_content=f"Descale {i}", metadata={"source_file": "m.pdf", "page": 1})
        for i in range(3)
    ]
    c, _, _ = make_client(tmp_path, docs=docs)
    body = c.post("/ask", json={"question": "Descale?"}).json()
    assert body["declined"] is False and body["confidence"] == "high"
    c2, _, _ = make_client(
        tmp_path, {**_ok_handlers(), RouteDecision: lambda t: RouteDecision(route="out_of_scope")}
    )
    body2 = c2.post("/ask", json={"question": "Hi"}).json()
    assert body2["declined"] is True and body2["confidence"] is None


def test_sources_symlink_escape_is_404(tmp_path: Path) -> None:
    c, _, settings = make_client(
        tmp_path, docs=[Document(page_content="x", metadata={"source_file": "link.pdf"})]
    )
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"%PDF secret")
    settings.docs_dir.mkdir()
    (settings.docs_dir / "link.pdf").symlink_to(outside)
    assert c.get("/sources/link.pdf").status_code == 404


def test_confidence_uses_configured_min_relevant_docs(tmp_path: Path) -> None:
    from docrag.graph.builder import assess_confidence

    one = [Document(page_content="x", metadata={"source_file": "m.pdf", "page": 1})]
    state: Any = {"status": "answered", "grounded": True, "relevant_documents": one}
    assert assess_confidence(state, 1) == "high"
    assert assess_confidence(state, 2) == "medium"
    # end to end: MIN_RELEVANT_DOCS=1 with exactly one relevant chunk
    c, _, _ = make_client(tmp_path, docs=one)
    assert c.post("/ask", json={"question": "x?"}).json()["confidence"] == "high"


# --- language / hint (T2) ---


def _gen(sufficient: bool):  # type: ignore[no-untyped-def]
    return lambda t: GeneratedAnswer(
        answer="Long uncited draft.", sufficient=sufficient, cited_sources=[1] if sufficient else []
    )


def _docs() -> list[Document]:
    return [
        Document(page_content=f"Descale {i}", metadata={"source_file": "m.pdf", "page": 1})
        for i in range(3)
    ]


def test_language_and_hint_by_outcome(tmp_path: Path) -> None:
    from docrag.prompts import DECLINE_HINT, INSUFFICIENT_INFO

    c, _, _ = make_client(tmp_path, {**_ok_handlers(), GeneratedAnswer: _gen(False)}, _docs())
    b = c.post("/ask", json={"question": "Meine Maschine ist kaputt, kein Wasser. Was tun?"}).json()
    assert b["decline_reason"] == "insufficient_info" and b["language"] == "de"
    assert b["hint"] == DECLINE_HINT["de"] and b["answer"] == INSUFFICIENT_INFO["de"]
    b = c.post("/ask", json={"question": "The machine is broken, no water. What now?"}).json()
    assert b["language"] == "en" and b["hint"] == DECLINE_HINT["en"]

    c2, _, _ = make_client(
        tmp_path, {**_ok_handlers(), RouteDecision: lambda t: RouteDecision(route="out_of_scope")}
    )
    b = c2.post("/ask", json={"question": "Hallo, wie geht es dir heute?"}).json()
    assert b["decline_reason"] == "out_of_scope" and b["hint"] is None

    c3, _, _ = make_client(tmp_path, docs=_docs())
    b = c3.post("/ask", json={"question": "How often to descale?"}).json()
    assert b["declined"] is False and b["hint"] is None and b["language"] == "en"
    # the streamed result event carries both fields
    r = c.post("/ask/stream", json={"question": "The machine is broken, no water. What now?"})
    events = [json.loads(x[6:]) for x in r.text.splitlines() if x.startswith("data: ")]
    assert events[-1]["event"] == "result" and events[-1]["hint"] == DECLINE_HINT["en"]
    assert events[-1]["language"] == "en"


# --- ProviderError mapping (T3) ---

CLI_TEXT = "You've hit your session limit - resets 5pm"


def _cli_limit(_: str) -> RouteDecision:
    from docrag.claude_cli import ChatClaudeCLI

    # real classification path: CLI result text -> ProviderError
    ChatClaudeCLI._parse(json.dumps({"is_error": True, "result": CLI_TEXT}).encode(), b"", 1)
    raise AssertionError("unreachable")


def test_rate_limit_maps_to_429_without_cli_text(tmp_path: Path) -> None:
    c, _, _ = make_client(tmp_path, {**_ok_handlers(), RouteDecision: _cli_limit})
    r = c.post("/ask", json={"question": "hello?"})
    assert r.status_code == 429
    assert r.json()["code"] == "rate_limited"
    assert "session limit" not in r.text and "5pm" not in r.text
    s = c.post("/ask/stream", json={"question": "hello?"})
    events = [json.loads(x[6:]) for x in s.text.splitlines() if x.startswith("data: ")]
    assert events[-1]["event"] == "error" and events[-1]["code"] == "rate_limited"
    assert "session limit" not in s.text


@pytest.mark.parametrize(
    ("text", "code", "status"),
    [("request timed out", "timeout", 504), ("weird failure", "provider_unavailable", 503)],
)
def test_other_provider_codes(tmp_path: Path, text: str, code: str, status: int) -> None:
    from docrag.errors import ProviderError, classify_provider_text

    def boom(_: str) -> RouteDecision:
        raise ProviderError(classify_provider_text(text))

    c, _, _ = make_client(tmp_path, {**_ok_handlers(), RouteDecision: boom})
    r = c.post("/ask", json={"question": "hello?"})
    assert r.status_code == status and r.json()["code"] == code


def test_wrapped_provider_error_is_classified_and_unknown_is_internal(tmp_path: Path) -> None:
    from docrag.errors import ProviderError

    def wrapped(_: str) -> RouteDecision:
        try:
            raise ProviderError("rate_limited")
        except ProviderError as e:
            raise ValueError("wrapper") from e

    c, _, _ = make_client(tmp_path, {**_ok_handlers(), RouteDecision: wrapped})
    assert c.post("/ask", json={"question": "hello?"}).status_code == 429
    c2, _, _ = make_client(tmp_path, {**_ok_handlers(), RouteDecision: _boom})
    r = c2.post("/ask", json={"question": "hello?"})
    assert r.status_code == 502 and r.json() == {"detail": GENERIC_ERROR, "code": "internal"}


# --- validation (T5) ---


def test_validation_codes_without_pydantic_text(tmp_path: Path) -> None:
    c, _, _ = make_client(tmp_path)
    for path in ("/ask", "/ask/stream"):
        r = c.post(path, json={"question": "x" * 2001})
        assert r.status_code == 422 and r.json()["code"] == "question_too_long"
        assert "String should" not in r.text and "string_too_long" not in r.text
        for body in ({"question": "   "}, {}):
            r = c.post(path, json=body)
            assert r.status_code == 422 and r.json()["code"] == "question_empty"
            assert "String should" not in r.text and "Field required" not in r.text
    # other routes keep FastAPI defaults
    assert isinstance(c.get("/sources").status_code, int)


# --- DELETE safety (T6) ---


def _delete_client(tmp_path: Path, names: list[str]):  # type: ignore[no-untyped-def]
    docs = [Document(page_content="x", metadata={"source_file": n}) for n in names]
    c, store, settings = make_client(tmp_path, docs=docs)
    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    settings.samples_dir.mkdir(parents=True, exist_ok=True)
    return c, settings


def test_delete_symlinks_never_unlink_targets(tmp_path: Path) -> None:
    c, settings = _delete_client(tmp_path, ["out.pdf", "sample.pdf"])
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"outside")
    sample = settings.samples_dir / "sample.pdf"
    sample.write_bytes(b"sample")
    (settings.docs_dir / "out.pdf").symlink_to(outside)
    (settings.docs_dir / "sample.pdf").symlink_to(sample)
    for name in ("out.pdf", "sample.pdf"):
        r = c.delete(f"/documents/{name}")
        assert r.status_code == 200, r.text
        assert r.json()["file_removed"] is False
    assert outside.read_bytes() == b"outside" and sample.read_bytes() == b"sample"
    assert (settings.docs_dir / "out.pdf").is_symlink()


def test_delete_same_name_only_removes_docs_dir_copy(tmp_path: Path) -> None:
    c, settings = _delete_client(tmp_path, ["same.pdf"])
    (settings.docs_dir / "same.pdf").write_bytes(b"upload")
    (settings.samples_dir / "same.pdf").write_bytes(b"sample")
    r = c.delete("/documents/same.pdf")
    assert r.status_code == 200 and r.json()["file_removed"] is True
    assert not (settings.docs_dir / "same.pdf").exists()
    assert (settings.samples_dir / "same.pdf").read_bytes() == b"sample"


# --- disconnect cancels the stream generator (B5) ---


async def test_stream_cancelled_on_client_disconnect(tmp_path: Path) -> None:
    import asyncio

    from langchain_core.runnables import RunnableLambda

    state = {"started": asyncio.Event(), "cancelled": False}

    class HangingLLM(type(make_fake_llm({}))):  # type: ignore[misc]
        def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
            async def hang(_: Any) -> Any:
                state["started"].set()
                try:
                    await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    state["cancelled"] = True
                    raise

            return RunnableLambda(hang)

    deps = Deps(HangingLLM(handlers={}, calls=[]), FakeStore([]), make_settings(top_k=3))
    app = create_app(lambda: deps)
    body = json.dumps({"question": "hello?"}).encode()
    sent = [False]
    disconnect = asyncio.Event()

    async def receive() -> dict[str, Any]:
        if not sent[0]:
            sent[0] = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(_: Any) -> None:
        return None

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "path": "/ask/stream",
        "raw_path": b"/ask/stream",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json"), (b"content-length", b"%d" % len(body))],
        "server": ("test", 80),
        "client": ("test", 1),
        "scheme": "http",
    }
    task = asyncio.create_task(app(scope, receive, send))
    await asyncio.wait_for(state["started"].wait(), 5)
    disconnect.set()
    await asyncio.wait_for(task, 5)
    assert state["cancelled"] is True


@pytest.mark.parametrize(
    "text",
    ["Failed to generate answer", "accurate schema", "unlimited context", "moderate"],
)
def test_classify_no_false_rate_limit(text: str) -> None:
    from docrag.errors import classify_provider_text

    assert classify_provider_text(text) == "provider_unavailable"


@pytest.mark.parametrize(
    "text",
    [
        "Rate limit exceeded",
        "429",
        "Claude usage limit reached",
        "overloaded_error",
        "You've hit your session limit \u00b7 resets 6:20pm",
    ],
)
def test_classify_rate_limit_true_positives(text: str) -> None:
    from docrag.errors import classify_provider_text

    assert classify_provider_text(text) == "rate_limited"
