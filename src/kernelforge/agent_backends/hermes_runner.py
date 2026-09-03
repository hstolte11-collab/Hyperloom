# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Isolated Hermes one-shot adapter for endpoint_agnostic_runner_v1."""

from __future__ import annotations

import json
import math
import os
import re
import stat
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml


class HermesRunnerError(RuntimeError):
    """Hermes runner input, execution, or evidence failed closed."""


_REQUEST_KEYS = {
    "schema",
    "request_id",
    "provider",
    "protocol",
    "base_url",
    "api_key_env",
    "model",
    "capabilities",
    "sandbox",
    "timeout_seconds",
    "retry",
    "egress",
    "environment",
    "messages",
    "output_schema",
    "fallback",
}
_RESULT_KEYS = {
    "schema",
    "request_id",
    "provider",
    "protocol",
    "model",
    "status",
    "structured_output",
    "attempts",
    "timing",
    "capability_receipt",
    "diagnostics",
    "fallback_used",
    "promotion_authority",
}
_ID = re.compile(r"[A-Za-z0-9_.-]+")
_ALLOWED_ENV = (
    "HOME",
    "PATH",
    "LANG",
    "LC_ALL",
    "TERM",
    "TMPDIR",
    "XDG_RUNTIME_DIR",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
)


def _strict_pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise HermesRunnerError("duplicate JSON key")
        result[key] = value
    return result


def _strict_loads(text: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                HermesRunnerError("non-finite JSON number")
            ),
        )
    except HermesRunnerError:
        raise
    except (TypeError, ValueError) as exc:
        raise HermesRunnerError("invalid JSON") from exc


def _canonical(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HermesRunnerError("value is not strict JSON") from exc


def _write_new(path: Path, payload: bytes) -> None:
    path = path.absolute()
    if path.exists() or path.is_symlink():
        raise HermesRunnerError(f"refusing to overwrite {path}")
    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise HermesRunnerError("short evidence write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _regular_bytes(path: Path) -> bytes:
    path = path.absolute()
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise HermesRunnerError(f"missing evidence file: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise HermesRunnerError(f"evidence file must be regular and non-symlink: {path}")
    if path.resolve(strict=True) != path:
        raise HermesRunnerError(f"evidence file traverses a symlink: {path}")
    return path.read_bytes()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HermesRunnerError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HermesRunnerError(f"{name} must be a nonnegative integer")
    return value


def _default_command_runner(
    argv: Sequence[str],
    *,
    cwd: str,
    env: Mapping[str, str],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        env=dict(env),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


class HermesOneShotRunner:
    """Run one exact, tool-free Hermes candidate-authoring request."""

    def __init__(
        self,
        *,
        executable: str | os.PathLike[str],
        profile: str,
        profile_root: str | os.PathLike[str],
        inference_provider: str,
        model: str,
        evidence_root: str | os.PathLike[str],
        command_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.executable = Path(executable).absolute()
        self.profile = str(profile).strip()
        self.profile_root = Path(profile_root).absolute()
        self.inference_provider = str(inference_provider).strip()
        self.model = str(model).strip()
        self.evidence_root = Path(evidence_root).absolute()
        self.command_runner = command_runner or _default_command_runner
        self._validate_static_configuration()

    def _validate_static_configuration(self) -> None:
        if not self.profile or _ID.fullmatch(self.profile) is None:
            raise HermesRunnerError("invalid Hermes profile name")
        if self.profile_root.name != self.profile:
            raise HermesRunnerError("Hermes profile root/name mismatch")
        if self.profile_root.is_symlink() or not self.profile_root.is_dir():
            raise HermesRunnerError("Hermes profile must be a regular directory")
        if self.profile_root.resolve(strict=True) != self.profile_root:
            raise HermesRunnerError("Hermes profile traverses a symlink")
        try:
            executable_info = os.lstat(self.executable)
        except OSError as exc:
            raise HermesRunnerError("Hermes executable is missing") from exc
        if stat.S_ISLNK(executable_info.st_mode):
            raise HermesRunnerError("Hermes executable cannot be a symlink")
        if not stat.S_ISREG(executable_info.st_mode) or not os.access(
            self.executable,
            os.X_OK,
        ):
            raise HermesRunnerError("Hermes executable must be regular and executable")
        if not self.inference_provider or not self.model:
            raise HermesRunnerError("explicit Hermes inference provider/model required")
        config_path = self.profile_root / "config.yaml"
        raw = _regular_bytes(config_path)
        try:
            config = yaml.safe_load(raw.decode("utf-8", errors="strict")) or {}
        except (UnicodeDecodeError, yaml.YAMLError) as exc:
            raise HermesRunnerError("Hermes profile config is invalid") from exc
        if not isinstance(config, dict):
            raise HermesRunnerError("Hermes profile config must be an object")
        model = config.get("model")
        if not isinstance(model, dict) or (
            model.get("provider") != self.inference_provider
            or model.get("default") != self.model
        ):
            raise HermesRunnerError("Hermes profile provider/model drift")
        if config.get("fallback_providers") not in (None, []):
            raise HermesRunnerError("Hermes profile fallback chain must be empty")
        if config.get("fallback_model") not in (None, "", []):
            raise HermesRunnerError("Hermes profile legacy fallback must be empty")
        if config.get("mcp_servers") not in (None, {}):
            raise HermesRunnerError("Hermes profile MCP servers must be empty")
        if config.get("plugins") not in (None, {}):
            raise HermesRunnerError("Hermes profile plugins must be empty")
        platform_toolsets = config.get("platform_toolsets")
        if not isinstance(platform_toolsets, dict) or platform_toolsets.get("cli") != []:
            raise HermesRunnerError("Hermes profile CLI toolset declaration must be empty")
        if self.evidence_root.is_symlink():
            raise HermesRunnerError("Hermes evidence root cannot be a symlink")
        self.evidence_root.mkdir(parents=True, exist_ok=True)
        if self.evidence_root.resolve(strict=True) != self.evidence_root:
            raise HermesRunnerError("Hermes evidence root traverses a symlink")

    def _validate_request(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict) or set(value) != _REQUEST_KEYS:
            raise HermesRunnerError("Hermes request key closure failed")
        if value["schema"] != "endpoint_agnostic_runner_v1.request":
            raise HermesRunnerError("Hermes request schema drift")
        if not isinstance(value["request_id"], str) or _ID.fullmatch(
            value["request_id"]
        ) is None:
            raise HermesRunnerError("invalid Hermes request_id")
        if value["provider"] != "hermes" or value["protocol"] != "hermes_oneshot":
            raise HermesRunnerError("Hermes provider/protocol drift")
        if value["model"] != self.model:
            raise HermesRunnerError("Hermes requested model drift")
        if value["base_url"] is not None or value["api_key_env"] is not None:
            raise HermesRunnerError("Hermes request cannot carry endpoint credentials")
        if value["capabilities"] != ["coder", "structured_output"]:
            raise HermesRunnerError("Hermes capability roster drift")
        if value["sandbox"] != {"mode": "read_only", "writable_roots": []}:
            raise HermesRunnerError("Hermes request must be read-only")
        _positive_int(value["timeout_seconds"], "timeout_seconds")
        if value["retry"] != {"max_attempts": 1}:
            raise HermesRunnerError("Hermes request must have one attempt")
        if value["egress"] is not True:
            raise HermesRunnerError("Hermes OpenAI-Codex request requires egress")
        if value["environment"] != {}:
            raise HermesRunnerError("Hermes request environment must be empty")
        if value["fallback"] != "none":
            raise HermesRunnerError("Hermes request fallback must be none")
        messages = value["messages"]
        if not isinstance(messages, list) or not messages:
            raise HermesRunnerError("Hermes messages must be nonempty")
        for message in messages:
            if (
                not isinstance(message, dict)
                or set(message) != {"role", "content"}
                or message["role"] not in {"system", "user"}
                or not isinstance(message["content"], str)
                or not message["content"]
            ):
                raise HermesRunnerError("Hermes message contract drift")
        if value["output_schema"] != {
            "type": "object",
            "required": ["text"],
            "properties": {"text": {"type": "string"}},
        }:
            raise HermesRunnerError("Hermes output schema drift")
        _canonical(value)
        return value

    @staticmethod
    def _prompt(messages: Sequence[Mapping[str, str]]) -> str:
        sections = []
        for message in messages:
            sections.append(
                f"## {message['role'].capitalize()}\n{message['content']}"
            )
        return "\n\n".join(sections)

    @staticmethod
    def _minimal_environment() -> dict[str, str]:
        return {
            name: os.environ[name]
            for name in _ALLOWED_ENV
            if os.environ.get(name)
        }

    def _validate_usage(self, path: Path) -> dict[str, Any]:
        raw = _regular_bytes(path)
        try:
            value = _strict_loads(raw.decode("utf-8", errors="strict"))
        except UnicodeDecodeError as exc:
            raise HermesRunnerError("Hermes usage is not UTF-8") from exc
        if not isinstance(value, dict):
            raise HermesRunnerError("Hermes usage must be an object")
        if value.get("provider") != self.inference_provider:
            raise HermesRunnerError("Hermes usage provider drift")
        if value.get("model") != self.model:
            raise HermesRunnerError("Hermes usage model drift")
        if value.get("completed") is not True or value.get("failed") is not False:
            raise HermesRunnerError("Hermes usage completion drift")
        if _positive_int(value.get("api_calls"), "usage api_calls") != 1:
            raise HermesRunnerError("Hermes usage must contain exactly one API call")
        if not isinstance(value.get("session_id"), str) or not value["session_id"]:
            raise HermesRunnerError("Hermes usage session identity is missing")
        for name in (
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "total_tokens",
        ):
            if name in value and value[name] is not None:
                _nonnegative_int(value[name], f"usage {name}")
        return value

    async def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        value = self._validate_request(dict(request))
        attempt = self.evidence_root / value["request_id"]
        try:
            attempt.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise HermesRunnerError("Hermes request attempt already exists") from exc
        _write_new(attempt / "request.json", _canonical(value))
        usage_path = attempt / "usage.json"
        prompt = self._prompt(value["messages"])
        argv = [
            str(self.executable),
            "-p",
            self.profile,
            "-z",
            prompt,
            "--provider",
            self.inference_provider,
            "--model",
            self.model,
            "--reasoning",
            "low",
            "--toolsets",
            "none",
            "--usage-file",
            str(usage_path),
            "--safe-mode",
        ]
        started = time.monotonic()
        try:
            completed = self.command_runner(
                argv,
                cwd=str(attempt),
                env=self._minimal_environment(),
                timeout=value["timeout_seconds"],
            )
        except Exception as exc:
            raise HermesRunnerError("Hermes one-shot process failed") from exc
        elapsed = time.monotonic() - started
        stdout = str(getattr(completed, "stdout", "") or "")
        stderr = str(getattr(completed, "stderr", "") or "")
        _write_new(attempt / "stdout.log", stdout.encode("utf-8", errors="replace"))
        _write_new(attempt / "stderr.log", stderr.encode("utf-8", errors="replace"))
        if getattr(completed, "returncode", None) != 0:
            raise HermesRunnerError(
                f"Hermes one-shot exited with exit {completed.returncode}"
            )
        if stderr:
            raise HermesRunnerError("Hermes one-shot emitted unexpected stderr")
        usage = self._validate_usage(usage_path)
        text = stdout.strip()
        if not text:
            raise HermesRunnerError("Hermes one-shot returned empty candidate source")
        result = {
            "schema": "endpoint_agnostic_runner_v1.result",
            "request_id": value["request_id"],
            "provider": value["provider"],
            "protocol": value["protocol"],
            "model": value["model"],
            "status": "success",
            "structured_output": {"text": text},
            "attempts": 1,
            "timing": {"elapsed_seconds": elapsed},
            "capability_receipt": list(value["capabilities"]),
            "diagnostics": {
                "stderr_tail": "",
                "hermes_provider": usage["provider"],
                "hermes_model": usage["model"],
                "hermes_session_id": usage["session_id"],
                "api_calls": usage["api_calls"],
            },
            "fallback_used": False,
            "promotion_authority": False,
        }
        if set(result) != _RESULT_KEYS or not math.isfinite(elapsed):
            raise HermesRunnerError("internal Hermes result closure failed")
        _write_new(attempt / "result.json", _canonical(result))
        return result


__all__ = ["HermesOneShotRunner", "HermesRunnerError"]
