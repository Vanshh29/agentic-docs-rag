"""Offline fakes: a scripted chat model with structured output, an in-memory store, embeddings."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from docrag.config import Settings

Handler = Callable[[str], BaseModel]


def make_settings(**overrides: Any) -> Settings:
    """Settings that ignore the developer's local .env so unit tests are hermetic."""
    return Settings(_env_file=None, **overrides)


class FakeLLM(BaseChatModel):
    """Chat model whose `with_structured_output(Schema)` dispatches to a per-schema handler.

    The handler receives the concatenated message text and returns a Schema instance.
    """

    handlers: dict[Any, Any] = {}
    calls: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="fake"))])

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        handler = self.handlers[schema]

        def run(messages: Any) -> BaseModel:
            text = "\n".join(m if isinstance(m, str) else str(m[1]) for m in messages)
            self.calls.append(schema.__name__)
            result: BaseModel = handler(text)
            return result

        return RunnableLambda(run)


def make_fake_llm(handlers: dict[Any, Handler]) -> FakeLLM:
    return FakeLLM(handlers=handlers, calls=[])


class FakeStore:
    """Keyword-overlap store; records queries."""

    def __init__(self, docs: list[Document]) -> None:
        self.docs = docs
        self.queries: list[str] = []

    def add_documents(self, docs: list[Document], ids: list[str]) -> None:
        self.docs.extend(docs)

    def search(self, query: str, k: int, mmr: bool = False) -> list[Document]:
        self.queries.append(query)
        words = set(re.findall(r"\w+", query.lower()))
        scored = sorted(
            self.docs,
            key=lambda d: -len(words & set(re.findall(r"\w+", d.page_content.lower()))),
        )
        return scored[:k]

    def count(self) -> int:
        return len(self.docs)

    def delete_by_source(self, source_file: str, keep_ids: Sequence[str] = ()) -> None:
        self.docs = [
            d
            for d in self.docs
            if d.metadata.get("source_file") != source_file
            or d.metadata.get("chunk_id") in keep_ids
        ]

    def list_sources(self) -> list[str]:
        return sorted({str(d.metadata["source_file"]) for d in self.docs})

    def count_by_source(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self.docs:
            name = str(d.metadata["source_file"])
            counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items()))

    def reset(self) -> None:
        self.docs = []


class HashEmbeddings(Embeddings):
    """Deterministic bag-of-words hashing embeddings (offline; fine for integration tests)."""

    dim = 256

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for w in re.findall(r"\w+", text.lower()):
            vec[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
