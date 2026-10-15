"""Recorded-answers mode: a public demo that replays saved answers and never calls a model.

`scripts/record_demo.py` runs the real pipeline once and saves each answer (text, citations and
reasoning steps). In replay mode the API looks questions up in that file instead of calling a
model, so a public deployment needs no API key and cannot run up a bill.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

from docrag.errors import ProviderError
from docrag.schemas import AskResponse


def normalize(question: str) -> str:
    """Match key: ignores case, spacing and trailing punctuation, nothing more."""
    return re.sub(r"\s+", " ", question).strip().casefold().rstrip("?!. ").strip()


class ReplayBook:
    def __init__(self, items: list[tuple[str, AskResponse]]) -> None:
        self._items = items
        self._by_key = {normalize(q): r for q, r in items}

    @classmethod
    def empty(cls) -> ReplayBook:
        return cls([])

    @classmethod
    def load(cls, path: Path) -> ReplayBook:
        try:
            data = json.loads(path.read_text())
            items = [
                (a["question"], AskResponse.model_validate(a["response"])) for a in data["answers"]
            ]
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise ValueError(f"cannot read the recorded answers file {path}: {e}") from e
        return cls(items)

    def get(self, question: str) -> AskResponse | None:
        return self._by_key.get(normalize(question))

    def examples(self) -> list[dict[str, Any]]:
        """Recorded questions, in file order, for the UI's example cards."""
        return [
            {
                "question": q,
                "language": r.language,
                "source": r.citations[0].source_file if r.citations else None,
            }
            for q, r in self._items
        ]


class NoEmbeddings(Embeddings):
    """Stand-in for the embedding model: replay never embeds anything, so it is never loaded."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("replay mode makes no model calls")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("replay mode makes no model calls")


class NullChatModel(BaseChatModel):
    """Stand-in for the chat model in replay mode; any real call is a bug and fails loudly."""

    @property
    def _llm_type(self) -> str:
        return "replay-null"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise ProviderError("provider_unavailable")
