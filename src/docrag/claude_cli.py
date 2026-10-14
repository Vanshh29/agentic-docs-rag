"""Chat model backed by the local `claude` CLI (headless `claude -p`).

Uses the user's existing Claude login (e.g. a Pro/Max subscription), so no API key or API
billing is involved. Only structured output is supported, which is all the graph needs.
Requires the Claude Code CLI to be installed and logged in.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any, TypeVar

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from docrag.errors import ProviderError, classify_provider_text

T = TypeVar("T", bound=BaseModel)


class ChatClaudeCLI(BaseChatModel):
    model: str = "haiku"
    max_concurrency: int = 3
    timeout_s: float = 120.0
    _sem: asyncio.Semaphore | None = None

    @property
    def _llm_type(self) -> str:
        return "claude-cli"

    def with_structured_output(  # type: ignore[override]
        self, schema: type[T], **kwargs: Any
    ) -> Runnable[LanguageModelInput, T]:
        parse: Runnable[Any, T] = RunnableLambda(lambda m: schema.model_validate_json(m.content))
        return self.bind(json_schema=schema.model_json_schema()) | parse  # type: ignore[no-any-return]

    @staticmethod
    def _split(messages: list[BaseMessage]) -> tuple[str, str]:
        system = "\n\n".join(str(m.content) for m in messages if m.type == "system")
        user = "\n\n".join(str(m.content) for m in messages if m.type != "system")
        return system, user

    def _args(self, system: str, schema: dict[str, Any]) -> list[str]:
        return [
            "claude", "-p", "--model", self.model, "--tools", "", "--no-session-persistence",
            "--output-format", "json", "--json-schema", json.dumps(schema),
            "--system-prompt", system or "You are a helpful assistant.",
        ]  # fmt: skip

    @staticmethod
    def _parse(stdout: bytes, stderr: bytes, code: int | None) -> ChatResult:
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise ProviderError("provider_unavailable") from e
        if not isinstance(data, dict):
            raise ProviderError("provider_unavailable")
        if data.get("is_error") or data.get("structured_output") is None:
            # classify on the CLI text, but never carry that text into the exception
            raise ProviderError(classify_provider_text(str(data.get("result") or "")))
        msg = AIMessage(content=json.dumps(data["structured_output"]))
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.max_concurrency)
        system, user = self._split(messages)
        async with self._sem:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *self._args(system, kwargs["json_schema"]),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except OSError as e:  # CLI missing or not executable (FileNotFoundError is an OSError)
                raise ProviderError("provider_unavailable") from e
            try:
                out, err = await asyncio.wait_for(
                    proc.communicate(user.encode()), timeout=self.timeout_s
                )
            except BaseException as e:  # incl. cancellation: never orphan the child process
                if proc.returncode is None:
                    with contextlib.suppress(ProcessLookupError):  # child exited just now
                        proc.kill()
                await asyncio.shield(proc.wait())
                if isinstance(e, TimeoutError):
                    raise ProviderError("timeout") from e
                raise
        return self._parse(out, err, proc.returncode)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        import subprocess

        system, user = self._split(messages)
        try:
            p = subprocess.run(
                self._args(system, kwargs["json_schema"]),
                input=user.encode(),
                capture_output=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            raise ProviderError("timeout") from e
        except OSError as e:  # CLI missing or not executable
            raise ProviderError("provider_unavailable") from e
        return self._parse(p.stdout, p.stderr, p.returncode)
