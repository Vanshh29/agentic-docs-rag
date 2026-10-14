"""Factories for chat models and embeddings (provider-agnostic, swappable via config)."""

from __future__ import annotations

from typing import TypeVar

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from docrag.config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)


def make_llm(settings: Settings | None = None, model: str | None = None) -> BaseChatModel:
    """Build a chat model for the configured provider. OpenRouter uses its OpenAI-compatible API."""
    s = settings or get_settings()
    name = model or s.llm_model
    if s.llm_provider == "claude_cli":
        from docrag.claude_cli import ChatClaudeCLI

        return ChatClaudeCLI(model=name)
    if s.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(  # type: ignore[call-arg]
            model_name=name,
            temperature=s.llm_temperature,
            max_tokens=s.llm_max_tokens,
            api_key=s.anthropic_api_key,  # type: ignore[arg-type]
        )
    from langchain_openai import ChatOpenAI

    if s.llm_provider == "openrouter":
        return ChatOpenAI(
            model=name,
            temperature=s.llm_temperature,
            max_completion_tokens=s.llm_max_tokens,
            api_key=s.openrouter_api_key,  # type: ignore[arg-type]
            base_url=s.openrouter_base_url,
        )
    return ChatOpenAI(
        model=name,
        temperature=s.llm_temperature,
        max_completion_tokens=s.llm_max_tokens,
        api_key=s.openai_api_key,  # type: ignore[arg-type]
    )


def make_judge_llm(settings: Settings | None = None) -> BaseChatModel:
    """Judge model: temperature 0, separately configurable model."""
    s = (settings or get_settings()).model_copy(update={"llm_temperature": 0.0})
    return make_llm(s, model=s.judge_model)


class E5Embeddings(Embeddings):
    """Wraps a HF embedding model, adding the 'query:'/'passage:' prefixes E5 models expect."""

    def __init__(self, inner: Embeddings, e5: bool) -> None:
        self._inner = inner
        self._e5 = e5

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._e5:
            texts = [f"passage: {t}" for t in texts]
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(f"query: {text}" if self._e5 else text)


def make_embeddings(settings: Settings | None = None) -> Embeddings:
    s = settings or get_settings()
    from langchain_huggingface import HuggingFaceEmbeddings

    inner = HuggingFaceEmbeddings(
        model_name=s.embedding_model, encode_kwargs={"normalize_embeddings": True}
    )
    return E5Embeddings(inner, e5="e5" in s.embedding_model.lower())


def structured(llm: BaseChatModel, schema: type[T]) -> Runnable[LanguageModelInput, T]:
    """`.with_structured_output` with the most portable mode for OpenAI-compatible endpoints."""
    from langchain_openai import ChatOpenAI

    if isinstance(llm, ChatOpenAI):
        return llm.with_structured_output(schema, method="function_calling")  # type: ignore[return-value]
    return llm.with_structured_output(schema)  # type: ignore[return-value]
