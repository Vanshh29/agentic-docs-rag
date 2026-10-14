"""Typed provider errors with fixed, client-safe messages (never embed CLI/provider text)."""

from __future__ import annotations

import re
from typing import Literal

ErrorCode = Literal["provider_unavailable", "rate_limited", "timeout", "internal"]

MESSAGES: dict[str, str] = {
    "provider_unavailable": "The language model provider is currently unavailable.",
    "rate_limited": "The language model provider rate limit was reached; try again later.",
    "timeout": "The language model provider timed out.",
    "internal": "The request could not be completed due to a server error; see server logs.",
}
STATUS: dict[str, int] = {
    "provider_unavailable": 503,
    "rate_limited": 429,
    "timeout": 504,
    "internal": 502,
}

# Whole words/phrases only: bare "rate"/"limit" would match "generate", "unlimited", etc.
_RATE_RE = re.compile(
    r"\b(?:rate[ _-]?limit(?:ed)?|too many requests|429|quota|overloaded"
    r"|usage limit|session limit|limit reached)(?![a-z0-9])",
    re.IGNORECASE,
)


class ProviderError(Exception):
    """LLM provider failure. `str(exc)` is the fixed message for `code`."""

    def __init__(self, code: ErrorCode) -> None:
        super().__init__(MESSAGES[code])
        self.code: ErrorCode = code


def classify_provider_text(text: str) -> ErrorCode:
    """Map provider/CLI result text to an error code (case-insensitive phrase match)."""
    low = text.lower()
    if "timed out" in low:
        return "timeout"
    if _RATE_RE.search(low):
        return "rate_limited"
    return "provider_unavailable"


def find_provider_error(exc: BaseException | None) -> ProviderError | None:
    """Walk __cause__/__context__ to find a wrapped ProviderError."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, ProviderError):
            return exc
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return None
