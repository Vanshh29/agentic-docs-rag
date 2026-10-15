"""FastAPI app: /ask, /ask/stream (SSE node-level progress), /ingest, /health."""

from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from docrag.config import get_settings
from docrag.errors import MESSAGES, STATUS, find_provider_error
from docrag.graph import Deps, build_graph, to_response
from docrag.graph.builder import run_question
from docrag.graph.state import GraphState
from docrag.ingest import ingest_path
from docrag.ingest.loaders import SUPPORTED
from docrag.ingest.pipeline import IngestStats
from docrag.ratelimit import AskLimiter
from docrag.schemas import AskRequest, AskResponse

STATIC = Path(__file__).parent / "static"
log = logging.getLogger(__name__)
GENERIC_ERROR = MESSAGES["internal"]


QUESTION_ROUTES = ("/ask", "/ask/stream")


def _classify(exc: BaseException) -> str:
    pe = find_provider_error(exc)
    return pe.code if pe else "internal"


def _error_response(exc: BaseException) -> HTTPException:
    code = _classify(exc)
    return HTTPException(STATUS[code], {"detail": MESSAGES[code], "code": code})


def default_deps() -> Deps:
    from docrag.llm import make_embeddings, make_llm
    from docrag.store import ChromaStore

    s = get_settings()
    return Deps(make_llm(s), ChromaStore(make_embeddings(s), s), s)


def _is_sample(d: Deps, name: str) -> bool:
    """True when the file exists only as a committed sample (no uploaded copy in docs_dir)."""
    try:
        root = d.settings.docs_dir.resolve()
        samples = d.settings.samples_dir.resolve()
        up = (root / name).resolve()
        sm = (samples / name).resolve()
        if not up.is_relative_to(root) or not sm.is_relative_to(samples):
            return False
        return not up.is_file() and sm.is_file()
    except (OSError, ValueError):
        return False


def _remove_upload(d: Deps, name: str) -> bool:
    """Delete the persisted uploaded copy. Only inside docs_dir; never touches samples_dir."""
    root = d.settings.docs_dir.resolve()
    samples = d.settings.samples_dir.resolve()
    try:
        candidate = (root / name).resolve()
    except (OSError, ValueError):  # e.g. a NUL byte in the name: skip the disk step
        return False
    if candidate == root or not candidate.is_relative_to(root):
        return False
    if root == samples or candidate.is_relative_to(samples):
        return False  # committed sample manuals are never deleted
    if not candidate.is_file():
        return False
    try:
        candidate.unlink()
    except OSError:
        log.exception("could not remove uploaded copy %s", candidate)
        return False
    return True


def _demo_blocked() -> HTTPException:
    return HTTPException(
        403,
        {
            "detail": "Uploading and removing documents is disabled in this demo.",
            "code": "demo_mode",
        },
    )


def create_app(
    deps_factory: Callable[[], Deps] = default_deps, preload: bool | None = None
) -> FastAPI:
    """`preload` (default: the PRELOAD setting) warms the models and indexes the sample manuals
    in the background at startup, so a fresh container with an empty disk is ready quickly."""
    cache: dict[str, Any] = {}
    init_lock = threading.Lock()
    do_preload = get_settings().preload if preload is None else preload

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if do_preload:
            cache["warming"] = True  # set before the thread starts so /status never misses it
            threading.Thread(target=warm_up, name="preload", daemon=True).start()
        yield

    app = FastAPI(title="Agentic Documentation RAG", version="0.1.0", lifespan=lifespan)

    def warm_up() -> None:
        try:
            d = deps()
            if d.store.count() == 0 and d.settings.samples_dir.is_dir():
                ingest_path(d.settings.samples_dir, d.store, d.settings)
        except Exception:
            log.exception("preload failed; the app will initialise on the first request")
        finally:
            cache["warming"] = False

    def deps() -> Deps:
        if "graph" not in cache:  # lazy: the embedding model loads on first use, not at import
            with init_lock:
                if "graph" not in cache:
                    d = deps_factory()
                    cache["deps"] = d
                    cache["limiter"] = AskLimiter(
                        d.settings.ask_rate_per_minute, d.settings.ask_daily_limit
                    )
                    cache["graph"] = build_graph(d)  # set last: readers check "graph"
        return cache["deps"]  # type: ignore[no-any-return]

    def graph() -> Any:
        deps()
        return cache["graph"]

    def limit_asks(request: Request) -> None:
        rejection = cache["limiter"].check(request.client.host if request.client else "unknown")
        if rejection is None:
            return
        message = (
            "The demo has reached its daily question limit. Please come back tomorrow."
            if rejection.code == "daily_limit"
            else "Too many questions in a short time. Please wait a moment and try again."
        )
        raise HTTPException(
            429,
            {"detail": message, "code": rejection.code},
            headers={"Retry-After": str(rejection.retry_after)},
        )

    @app.exception_handler(HTTPException)
    async def http_exc(request: Request, exc: HTTPException) -> JSONResponse:
        # structured error bodies ({detail, code}) pass through; plain ones keep {"detail": ...}
        if isinstance(exc.detail, dict) and "code" in exc.detail:
            return JSONResponse(exc.detail, status_code=exc.status_code, headers=exc.headers)
        return JSONResponse(
            {"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exc(request: Request, exc: RequestValidationError) -> JSONResponse:
        if request.url.path not in QUESTION_ROUTES:
            from fastapi.exception_handlers import request_validation_exception_handler

            return await request_validation_exception_handler(request, exc)
        types = {str(e.get("type")) for e in exc.errors()}
        if "string_too_long" in types:
            code, detail = "question_too_long", "The question is too long (max 2000 characters)."
        else:  # string_too_short / missing / malformed body
            code, detail = "question_empty", "The question must not be empty."
        return JSONResponse({"detail": detail, "code": code}, status_code=422)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        # no-cache: browsers must revalidate, otherwise a redeploy can serve a stale page for days
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/status")
    async def status() -> dict[str, Any]:
        d = await run_in_threadpool(deps)
        return {
            "chunks": await run_in_threadpool(d.store.count),
            "model": d.settings.llm_model,
            "demo_mode": d.settings.demo_mode,
            "indexing": bool(cache.get("warming")),
            "sources": sorted(await run_in_threadpool(d.store.list_sources)),
        }

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    async def ask(req: AskRequest, request: Request) -> AskResponse:
        try:
            d = await run_in_threadpool(deps)
        except Exception as e:
            log.exception("/ask failed")
            raise _error_response(e) from e
        limit_asks(request)  # outside the try: a 429 must not be reported as a server error
        try:
            g = cache["graph"]
            return to_response(await run_question(g, req.question), d.settings.min_relevant_docs)
        except Exception as e:
            log.exception("/ask failed")
            raise _error_response(e) from e

    @app.post("/ask/stream")
    async def ask_stream(req: AskRequest, request: Request) -> StreamingResponse:
        await run_in_threadpool(deps)
        limit_asks(request)

        async def events() -> AsyncIterator[str]:
            try:
                async for line in run_stream():
                    yield line
            except Exception:
                log.exception("/ask/stream failed")
                code = _classify(sys.exc_info()[1] or Exception())
                err = {"event": "error", "message": MESSAGES[code], "code": code}
                yield f"data: {json.dumps(err)}\n\n"

        async def run_stream() -> AsyncIterator[str]:
            d = await run_in_threadpool(deps)
            g = cache["graph"]
            state: GraphState = {
                "question": req.question,
                "retry_count": 0,
                "regeneration_count": 0,
                "feedback": [],
                "trace": [],
            }
            async for update in g.astream(state, {"recursion_limit": 40}, stream_mode="updates"):
                for node, delta in update.items():
                    for k, v in delta.items():
                        if k == "trace":
                            state["trace"] = [*state["trace"], *v]
                        else:
                            state[k] = v  # type: ignore[literal-required]
                    step = delta.get("trace", [None])[-1]
                    payload = {"event": "node", "node": node, "detail": step.detail if step else ""}
                    yield f"data: {json.dumps(payload)}\n\n"
            final = to_response(state, d.settings.min_relevant_docs).model_dump(mode="json")
            yield f"data: {json.dumps({'event': 'result', **final})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/sources/{file:path}")
    async def source(file: str) -> FileResponse:
        d = await run_in_threadpool(deps)
        # Whitelist: only names the store actually holds; never join a raw request path.
        known = set(await run_in_threadpool(d.store.list_sources))
        if file not in known:
            raise HTTPException(404, "Unknown source")
        for base in (d.settings.docs_dir, d.settings.samples_dir):
            root = base.resolve()
            candidate = (root / file).resolve()
            if candidate.is_relative_to(root) and candidate.is_file():
                return FileResponse(candidate)
        raise HTTPException(404, "Source file is not available on the server")

    @app.get("/documents")
    async def documents() -> dict[str, Any]:
        d = await run_in_threadpool(deps)
        try:
            counts = await run_in_threadpool(d.store.count_by_source)
        except Exception as e:
            log.exception("/documents failed")
            raise HTTPException(502, GENERIC_ERROR) from e
        return {
            "documents": [
                {"source_file": n, "chunks": c, "is_sample": _is_sample(d, n)}
                for n, c in sorted(counts.items())
            ]
        }

    @app.delete("/documents/{file:path}")
    async def delete_document(file: str) -> dict[str, Any]:
        d = await run_in_threadpool(deps)
        if d.settings.demo_mode:
            raise _demo_blocked()

        def work() -> dict[str, Any] | None:
            # Whitelist: only names the index actually holds; never act on a raw request path.
            counts = d.store.count_by_source()
            if file not in counts:
                return None
            d.store.delete_by_source(file)
            return {
                "deleted": file,
                "chunks": counts[file],
                "file_removed": _remove_upload(d, file),
            }

        try:
            result = await run_in_threadpool(work)
        except Exception as e:
            log.exception("DELETE /documents failed")
            raise HTTPException(502, GENERIC_ERROR) from e
        if result is None:
            raise HTTPException(404, "Unknown document")
        return result

    @app.post("/ingest", response_model=None)
    async def ingest(file: UploadFile) -> JSONResponse | dict[str, Any]:
        d = await run_in_threadpool(deps)
        if d.settings.demo_mode:
            raise _demo_blocked()
        name = Path(file.filename or "").name
        if not name or name.startswith(".") or Path(name).suffix.lower() not in SUPPORTED:
            raise HTTPException(400, f"Unsupported file type; use one of {sorted(SUPPORTED)}")

        def work() -> tuple[IngestStats, Path | None]:
            with tempfile.TemporaryDirectory() as tmp:
                dest = Path(tmp) / name
                with dest.open("wb") as out:
                    shutil.copyfileobj(file.file, out)
                stats = ingest_path(dest, d.store, d.settings)
                saved: Path | None = None
                if stats.files:  # persist so /sources can serve it; failures are not kept
                    d.settings.docs_dir.mkdir(parents=True, exist_ok=True)
                    saved = d.settings.docs_dir / name
                    shutil.copyfile(dest, saved)
            return stats, saved

        try:
            stats, _ = await run_in_threadpool(work)
        except Exception as e:
            log.exception("/ingest failed")
            raise HTTPException(
                502, "Ingestion failed due to a server error; see server logs."
            ) from e
        results: list[dict[str, Any]] = [
            {"file": f, "status": "failed", "error": err.split(":")[0]}
            for f, err in stats.failed_files
        ]
        results += [{"file": f, "status": "ok", "chunks": stats.chunks} for f in stats.sources]
        if not stats.files:
            reason = "could not be read" if stats.failed else "contained no extractable text"
            return JSONResponse(
                status_code=422,
                content={"detail": f"{name} {reason}", "results": results},
            )
        return {
            "files": stats.files,
            "pages_or_sections": stats.pages_or_sections,
            "chunks": stats.chunks,
            "failed": stats.failed,
            "results": results,
        }

    return app


app = create_app()
