"""ChatClaudeCLI subprocess handling and error classification (offline, patched subprocess)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from docrag.claude_cli import ChatClaudeCLI
from docrag.errors import ProviderError, classify_provider_text


class HangingProc:
    returncode: int | None = None

    def __init__(self) -> None:
        self.killed = False
        self.waited = False

    async def communicate(self, data: bytes | None = None) -> tuple[bytes, bytes]:
        await asyncio.sleep(3600)
        return b"", b""

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        self.waited = True
        return -9


def _patch(monkeypatch: pytest.MonkeyPatch, proc: HangingProc) -> None:
    async def fake_exec(*args: Any, **kwargs: Any) -> HangingProc:
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)


async def test_timeout_kills_child_and_raises_provider_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = HangingProc()
    _patch(monkeypatch, proc)
    llm = ChatClaudeCLI(timeout_s=0.05)
    with pytest.raises(ProviderError) as ei:
        await llm._agenerate([HumanMessage("q")], json_schema={})
    assert ei.value.code == "timeout"
    assert proc.killed and proc.waited


async def test_cancellation_kills_child(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = HangingProc()
    _patch(monkeypatch, proc)
    llm = ChatClaudeCLI(timeout_s=60)
    task = asyncio.create_task(llm._agenerate([HumanMessage("q")], json_schema={}))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert proc.killed and proc.waited


def test_classification_and_fixed_messages() -> None:
    assert classify_provider_text("You've hit your session limit") == "rate_limited"
    assert classify_provider_text("Server OVERLOADED") == "rate_limited"
    assert classify_provider_text("Request timed out") == "timeout"
    assert classify_provider_text("boom") == "provider_unavailable"
    text = "You've hit your session limit - secret detail"
    with pytest.raises(ProviderError) as ei:
        ChatClaudeCLI._parse(json.dumps({"is_error": True, "result": text}).encode(), b"", 1)
    assert ei.value.code == "rate_limited" and "secret detail" not in str(ei.value)
    with pytest.raises(ProviderError) as ei2:
        ChatClaudeCLI._parse(b"not json", b"stderr secret", 1)
    assert ei2.value.code == "provider_unavailable" and "secret" not in str(ei2.value)


async def test_spawn_failure_maps_to_provider_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("claude")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", boom)
    with pytest.raises(ProviderError) as ei:
        await ChatClaudeCLI()._agenerate([HumanMessage("q")], json_schema={})
    assert ei.value.code == "provider_unavailable"
    assert "claude" not in str(ei.value)


def test_sync_spawn_failure_maps_to_provider_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError("not executable")

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(ProviderError) as ei:
        ChatClaudeCLI()._generate([HumanMessage("q")], json_schema={})
    assert ei.value.code == "provider_unavailable"


async def test_kill_race_process_already_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    class RacyProc(HangingProc):
        def kill(self) -> None:
            self.killed = True
            raise ProcessLookupError

    proc = RacyProc()
    _patch(monkeypatch, proc)
    with pytest.raises(ProviderError) as ei:
        await ChatClaudeCLI(timeout_s=0.05)._agenerate([HumanMessage("q")], json_schema={})
    assert ei.value.code == "timeout" and proc.killed and proc.waited
