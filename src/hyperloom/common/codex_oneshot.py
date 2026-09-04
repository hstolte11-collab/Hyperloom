# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Single-shot, tool-free Codex completions through the Codex Agent SDK.

The OpenAI-side mirror of :mod:`hyperloom.common.claude_oneshot`. The
``chat.completions`` HTTP path in :mod:`hyperloom.common.llm_config`
authenticates with a bearer API key, a channel that rejects a ChatGPT/Codex
subscription login outright. Driving single-shot OpenAI-side inference through
the Codex SDK under :class:`~hyperloom.common.codex_session.CodexSession`'s
``native_oauth`` mode hands credential resolution to the Codex CLI, which reads
the login the operator already holds in their own ``CODEX_HOME``.

Callers keep the ``AsyncOpenAI`` shape they already consume --
``client.chat.completions.create(model=..., messages=..., ...)`` returning an
object with ``choices[0].message.content``, ``choices[0].finish_reason`` and an
OpenAI-spelled ``usage`` -- so critic reasoning, robustness RCA, the proposal
scorer and the framework audit are unchanged from the HTTP path, and
:func:`hyperloom.common.llm_config.achat_completion` flattens the result
exactly as before.

Each call is one independent, tool-free turn with no writable roots, no private
``CODEX_HOME`` created and nothing written under the operator's. The sandbox
preset itself follows the deployment's ``HYPERLOOM_CODEX_SANDBOX_MODE``, exactly
as the Orchestrator's :class:`~hyperloom.orchestrator.roles.codex.CodexBackend`
does, so a container without bubblewrap that already runs the Orchestrator can
run these calls too. Streaming and tool calling are refused rather than
emulated.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Iterator, Mapping, Sequence

from .codex_session import CodexSessionError, native_oauth_codex_home, run_codex_turn

__all__ = ["CodexOneShotClient", "CodexOneShotError", "DEFAULT_TIMEOUT_SEC", "SyncCodexOneShotClient"]

DEFAULT_TIMEOUT_SEC = 300.0

# chat.completions parameters this transport cannot honour. Rejected rather than
# silently dropped so a caller that needs them fails at the call, not on a
# quietly degraded answer. ``stream=True`` IS honoured on both clients: the
# single completed turn is replayed as the chunk sequence the SDK would emit
# (see ``_stream_chunks``), which is what ``stream_chat_completion_text`` /
# ``astream_chat_completion_text`` (proposal scorer, framework audit) consume.
_UNSUPPORTED_PARAMS: tuple[str, ...] = ("tools", "tool_choice", "functions", "function_call", "n")


class CodexOneShotError(RuntimeError):
    """A single-shot Codex completion could not be served."""


def _prompt_from_messages(messages: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """Split OpenAI chat messages into (developer_instructions, user_prompt).

    ``system`` turns become the thread's developer instructions; every other
    turn is flattened into the prompt in order, role-labelled when there is
    more than one, exactly as :mod:`claude_oneshot` flattens Anthropic turns.
    """
    system_parts: list[str] = []
    others: list[tuple[str, str]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if not isinstance(content, str):
            # Multi-part content (text blocks) -> join the text pieces.
            content = "\n".join(str(part.get("text", "")) for part in content if isinstance(part, Mapping))
        if role in ("system", "developer"):
            system_parts.append(content)
        else:
            others.append((role, content))
    if len(others) == 1:
        prompt = others[0][1]
    else:
        prompt = "\n\n".join(f"[{role}]\n{content}" for role, content in others)
    return "\n\n".join(system_parts).strip(), prompt


def _usage_namespace(usage: Mapping[str, int]) -> SimpleNamespace:
    """OpenAI-spelled usage so existing accumulators (``prompt_tokens`` /
    ``completion_tokens``) keep working without knowing the transport."""
    prompt = int(usage.get("input_tokens", 0) or 0)
    completion = int(usage.get("output_tokens", 0) or 0)
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        # Preserved for callers that already understand the SDK spellings.
        input_tokens=prompt,
        output_tokens=completion,
        cache_read_input_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
        reasoning_output_tokens=int(usage.get("reasoning_output_tokens", 0) or 0),
    )


@dataclass
class _Completions:
    client: "CodexOneShotClient"

    async def create(self, **params: Any) -> Any:
        stream = bool(params.pop("stream", False))
        params.pop("stream_options", None)
        completion = await self.client._create(**params)
        return _astream_chunks(completion) if stream else completion


@dataclass
class _Chat:
    completions: _Completions


@dataclass
class CodexOneShotClient:
    """Tool-free Codex completion client shaped like ``openai.AsyncOpenAI``.

    :class:`SyncCodexOneShotClient` is the ``openai.OpenAI``-shaped twin.

    Attributes:
        codex_home: The operator's ``CODEX_HOME`` holding the CLI login. Must
            satisfy :func:`native_oauth_codex_home` (absolute, existing,
            non-symlink, regular 0600 ``auth.json``).
        cwd: Working directory for the read-only thread.
        timeout_s: Wall-clock budget for one completion, CLI startup included.
        env: Values overlaid on ``os.environ`` for the CLI child.
        sandbox_mode: Codex sandbox preset; blank (default) defers to the
            deployment's ``HYPERLOOM_CODEX_SANDBOX_MODE`` like every other
            Codex role. ``writable_roots`` is always empty regardless.
        component: Producer label used to tag the child's calls.
        operation: What the child is being spawned to do.
        codex_bin: Optional Codex runtime path; the SDK resolves its own when
            empty.
    """

    codex_home: str
    cwd: Path = field(default_factory=Path.cwd)
    timeout_s: float = DEFAULT_TIMEOUT_SEC
    env: dict[str, str] | None = None
    sandbox_mode: str = ""
    component: str = ""
    operation: str = ""
    codex_bin: str = ""
    chat: _Chat = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.codex_home = str(Path(self.codex_home).expanduser().resolve())
        self.cwd = Path(self.cwd).expanduser().resolve()
        self.chat = _Chat(completions=_Completions(client=self))

    # ``openai.AsyncOpenAI`` / ``openai.OpenAI`` lifecycle surface. Each turn
    # starts and closes its own CodexSession, so there is no pooled transport
    # to release; these exist so callers written against the SDK
    # (``await client.close()``, ``async with client:``, ``with client:``)
    # need no transport-specific branch.
    async def close(self) -> None:
        return None

    async def __aenter__(self) -> "CodexOneShotClient":
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False

    def __enter__(self) -> "CodexOneShotClient":
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    async def _create(self, **params: Any) -> Any:
        unsupported = [name for name in _UNSUPPORTED_PARAMS if params.get(name)]
        if unsupported:
            raise CodexOneShotError(
                f"CodexOneShotClient cannot serve chat.completions parameter(s) {', '.join(unsupported)}; "
                "the native_oauth transport is a single tool-free turn"
            )
        model = str(params.get("model") or "").strip()
        if not model:
            raise CodexOneShotError("chat.completions.create requires a model")
        messages = params.get("messages") or []
        developer_instructions, prompt = _prompt_from_messages(messages)
        if not prompt.strip():
            raise CodexOneShotError("chat.completions.create requires a non-empty user message")
        max_tokens = params.get("max_completion_tokens") or params.get("max_tokens")

        try:
            result = await run_codex_turn(
                prompt=prompt,
                developer_instructions=developer_instructions,
                cwd=self.cwd,
                model=model,
                timeout_sec=float(self.timeout_s),
                writable_roots=(),
                sandbox_mode=self.sandbox_mode,
                codex_bin=self.codex_bin,
                env=self.env,
                component=self.component,
                operation=self.operation,
                auth_mode="native_oauth",
                codex_home=self.codex_home,
            )
        except asyncio.TimeoutError as exc:
            raise CodexOneShotError(f"Codex one-shot exceeded {self.timeout_s:g}s") from exc
        except CodexSessionError as exc:
            raise CodexOneShotError(str(exc)) from exc
        if result.error:
            raise CodexOneShotError(result.error)

        usage = _usage_namespace(result.usage or {})
        finish_reason = "stop"
        if max_tokens is not None and usage.completion_tokens >= int(max_tokens):
            finish_reason = "length"
        message = SimpleNamespace(role="assistant", content=result.text)
        choice = SimpleNamespace(index=0, message=message, finish_reason=finish_reason)
        return SimpleNamespace(
            id=result.thread_id,
            model=model,
            choices=[choice],
            usage=usage,
        )


def _stream_chunks(completion: Any) -> Iterator[Any]:
    """Replay one completed response as the two-chunk stream the SDK would emit.

    Matches what :func:`hyperloom.common.llm_config.stream_chat_completion_text`
    consumes: a content chunk carrying ``choices[0].delta.content`` followed by a
    usage-only chunk with empty ``choices`` (``stream_options.include_usage``).
    """
    choice = completion.choices[0]
    yield SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=choice.message.content), finish_reason=None)],
        usage=None,
    )
    yield SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason=choice.finish_reason)],
        usage=None,
    )
    yield SimpleNamespace(choices=[], usage=completion.usage)


async def _astream_chunks(completion: Any) -> AsyncIterator[Any]:
    """Async twin of :func:`_stream_chunks` for ``astream_chat_completion_text``."""
    for chunk in _stream_chunks(completion):
        yield chunk


@dataclass
class _SyncCompletions:
    client: "SyncCodexOneShotClient"

    def create(self, **params: Any) -> Any:
        stream = bool(params.pop("stream", False))
        params.pop("stream_options", None)
        completion = self.client._run(**params)
        return _stream_chunks(completion) if stream else completion


@dataclass
class _SyncChat:
    completions: _SyncCompletions


@dataclass
class SyncCodexOneShotClient(CodexOneShotClient):
    """``openai.OpenAI``-shaped twin of :class:`CodexOneShotClient`.

    Returned by :func:`hyperloom.common.llm_config.get_openai_client` under
    ``native_oauth`` for the synchronous callers (framework audit, breakdown
    reporter). Runs each turn on a private event loop; calling it from inside a
    running loop is a caller error, exactly as with the sync ``openai`` client
    used from async code.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.chat = _SyncChat(completions=_SyncCompletions(client=self))  # type: ignore[assignment]

    def _run(self, **params: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._create(**params))
        raise CodexOneShotError(
            "SyncCodexOneShotClient used inside a running event loop; use CodexOneShotClient and await instead"
        )

    def close(self) -> None:  # type: ignore[override]
        return None


def ensure_available(codex_home: str) -> Path:
    """Fail now if the transport is unusable: validated home plus importable SDK.

    Raises:
        CodexOneShotError: If the home/login is invalid or the SDK is missing.
    """
    try:
        home = native_oauth_codex_home({"INFERENCE_OPTIMIZER_CODEX_HOME": codex_home})
        from .codex_session import load_codex_sdk

        load_codex_sdk()
    except CodexSessionError as exc:
        raise CodexOneShotError(str(exc)) from exc
    return home
