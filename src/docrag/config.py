"""Application settings, loaded from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # LLM
    llm_provider: Literal["claude_cli", "openrouter", "anthropic", "openai"] = "claude_cli"
    llm_model: str = "haiku"  # claude_cli alias; use a full model id for other providers
    judge_model: str = "haiku"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 600  # also keeps OpenRouter from reserving credit for huge outputs
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Embeddings
    embedding_model: str = "intfloat/multilingual-e5-small"

    # Retrieval / graph
    top_k: int = Field(default=6, ge=1)
    use_mmr: bool = False
    min_relevant_docs: int = Field(default=1, ge=1)  # spec said 2; see README (eval)
    max_rewrites: int = Field(default=2, ge=0)
    max_regenerations: int = Field(default=1, ge=0)

    # Ingestion
    chunk_size: int = Field(default=800, ge=100)
    chunk_overlap: int = Field(default=120, ge=0)
    chroma_dir: Path = Path("./data/chroma")
    collection_name: str = "docs"
    docs_dir: Path = Path("./data/raw")  # where uploaded files are kept and served from
    samples_dir: Path = Path("./data/samples")  # also servable (committed sample manuals)

    @model_validator(mode="after")
    def _check_chunking(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
