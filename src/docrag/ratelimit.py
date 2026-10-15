"""In-memory request limits for the question endpoints of a public deployment.

Two independent limits, both disabled when set to 0:
- per client: at most `per_minute` accepted requests in any sliding 60 s window
- global: at most `daily` accepted requests per UTC day (a spend cap for the LLM key)

State lives in the process, so it resets on restart and is not shared between instances.
That is fine for a single-instance demo; use a shared store for anything larger.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

WINDOW_S = 60
DAY_S = 86_400
MAX_TRACKED_CLIENTS = 10_000

LimitCode = Literal["too_many_requests", "daily_limit"]


@dataclass(frozen=True)
class Rejection:
    code: LimitCode
    retry_after: int  # seconds


class AskLimiter:
    def __init__(self, per_minute: int, daily: int, clock: Callable[[], float] = time.time) -> None:
        self._per_minute = per_minute
        self._daily = daily
        self._clock = clock
        self._windows: dict[str, deque[float]] = {}
        self._day = -1
        self._today = 0

    @property
    def enabled(self) -> bool:
        return self._per_minute > 0 or self._daily > 0

    def check(self, key: str) -> Rejection | None:
        """Admit the request (and count it) or return why it was rejected (and count nothing)."""
        now = self._clock()
        if self._daily:
            day = int(now // DAY_S)
            if day != self._day:
                self._day, self._today = day, 0
            if self._today >= self._daily:
                return Rejection("daily_limit", max(1, int((day + 1) * DAY_S - now)))
        window: deque[float] | None = None
        if self._per_minute:
            window = self._windows.setdefault(key, deque())
            while window and window[0] <= now - WINDOW_S:
                window.popleft()
            if len(window) >= self._per_minute:
                return Rejection("too_many_requests", max(1, math.ceil(window[0] + WINDOW_S - now)))
        if window is not None:
            window.append(now)
            self._prune()
        if self._daily:
            self._today += 1
        return None

    def _prune(self) -> None:
        if len(self._windows) > MAX_TRACKED_CLIENTS:
            cutoff = self._clock() - WINDOW_S
            self._windows = {k: w for k, w in self._windows.items() if w and w[-1] > cutoff}
