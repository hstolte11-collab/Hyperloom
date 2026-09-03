# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Strict KernelForge adapter for endpoint_agnostic_runner_v1."""

from __future__ import annotations

import inspect
import math
from typing import Any, Mapping

from kernelforge.agent_backends.base import (
    AgentCapabilities,
    AgentRunResult,
)

_SECRET = ("key", "token", "secret", "password", "credential", "auth")
_FORBIDDEN_OUTPUT = {"keep", "correct", "evaluator", "promotion", "authority"}
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
_CAPABILITIES = {
    "architect",
    "coder",
    "reviewer",
    "workflow",
    "session_resume",
    "tools",
    "structured_output",
}


def _contains_secret_name(value: str) -> bool:
    lowered = value.lower()
    return any(fragment in lowered for fragment in _SECRET)


def _finite_tree(value: Any, *, reject_secret_names: bool = True) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite runner value")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or (reject_secret_names and _contains_secret_name(key)):
                raise ValueError("secret or non-string runner field")
            _finite_tree(item, reject_secret_names=reject_secret_names)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item, reject_secret_names=reject_secret_names)


def _reject_secret_names(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or _contains_secret_name(key):
                raise ValueError("secret or non-string runner field")
            _reject_secret_names(item)
    elif isinstance(value, list):
        for item in value:
            _reject_secret_names(item)


class EndpointAgnosticBackend:
    """Map KernelForge's provider-neutral spec to the frozen runner contract."""

    name = "endpoint_agnostic"
    capabilities = AgentCapabilities(writable=True, resumable=True, session_env=True)

    def __init__(self, runtime, runner):
        self.runtime = runtime
        self.runner = runner
        options = runtime.options
        if runtime.provider != "endpoint_agnostic" or not runtime.model.strip():
            raise ValueError("explicit endpoint runtime required")
        required = {"provider", "protocol", "capabilities", "egress", "fallback"}
        if not isinstance(options, dict) or not required.issubset(options):
            raise ValueError("explicit endpoint runner options required")
        if not isinstance(options["provider"], str) or not options["provider"]:
            raise ValueError("explicit provider required")
        if not isinstance(options["protocol"], str) or not options["protocol"]:
            raise ValueError("explicit protocol required")
        if (
            not isinstance(options["capabilities"], list)
            or not options["capabilities"]
            or any(item not in _CAPABILITIES for item in options["capabilities"])
            or len(set(options["capabilities"])) != len(options["capabilities"])
        ):
            raise ValueError("invalid capability roster")
        if not isinstance(options["egress"], bool):
            raise ValueError("egress must be explicit boolean")
        if options["fallback"] != "none" or runtime.fallback_provider not in {"", "none"}:
            raise ValueError("fallback must be none")
        if options["provider"] == "openai_compatible" and not options.get("base_url"):
            raise ValueError("openai-compatible provider requires base_url")
        api_key_env = options.get("api_key_env")
        if api_key_env is not None and (
            not isinstance(api_key_env, str) or not api_key_env.isidentifier()
        ):
            raise ValueError("api_key_env must be an environment-variable name or null")
        if not callable(runner):
            raise TypeError("runner must be callable")

    def _request(self, spec) -> dict[str, Any]:
        options = self.runtime.options
        for key, value in spec.env.items():
            if not isinstance(key, str) or _contains_secret_name(key) or not isinstance(value, str):
                raise ValueError("secret-named or non-string session environment is forbidden")
        request = {
            "schema": "endpoint_agnostic_runner_v1.request",
            "request_id": f"kernelforge-{id(spec):x}",
            "provider": options["provider"],
            "protocol": options["protocol"],
            "base_url": options.get("base_url"),
            "api_key_env": options.get("api_key_env"),
            "model": spec.model,
            "capabilities": list(options["capabilities"]),
            "sandbox": {
                "mode": "workspace_write" if spec.writable else "read_only",
                "writable_roots": [spec.cwd] if spec.writable else [],
            },
            "timeout_seconds": spec.timeout_sec,
            "retry": {"max_attempts": 1},
            "egress": options["egress"],
            "environment": dict(spec.env),
            "messages": [
                {"role": "system", "content": spec.system_prompt},
                {"role": "user", "content": spec.user_prompt},
            ],
            "output_schema": {
                "type": "object",
                "required": ["text"],
                "properties": {
                    "text": {"type": "string"},
                    "session_id": {"type": "string"},
                },
            },
            "fallback": "none",
        }
        if set(request) != _REQUEST_KEYS:
            raise ValueError("internal request key closure failed")
        _finite_tree(request, reject_secret_names=False)
        return request

    async def run(self, spec, usage=None):
        resolved = spec.resolved(self.runtime)
        request = self._request(resolved)
        response = self.runner(request)
        if inspect.isawaitable(response):
            response = await response
        return self._normalize(response, request)

    def _normalize(self, result: Any, request: Mapping[str, Any]) -> AgentRunResult:
        if not isinstance(result, dict) or set(result) != _RESULT_KEYS:
            raise ValueError("malformed runner result key closure")
        _finite_tree(result, reject_secret_names=False)
        if result["schema"] != "endpoint_agnostic_runner_v1.result":
            raise ValueError("malformed runner result schema")
        for key in ("request_id", "provider", "protocol", "model"):
            if result[key] != request[key]:
                raise ValueError("runner result identity mismatch")
        if result["status"] != "success":
            raise ValueError("runner result is not successful")
        if (
            result["attempts"] != 1
            or result["fallback_used"] is not False
            or result["promotion_authority"] is not False
        ):
            raise ValueError("invalid attempt or authority receipt")
        if result["capability_receipt"] != request["capabilities"]:
            raise ValueError("capability receipt mismatch")
        diagnostics = result["diagnostics"]
        output = result["structured_output"]
        _reject_secret_names(diagnostics)
        _reject_secret_names(output)
        if (
            not isinstance(diagnostics, dict)
            or diagnostics.get("stderr_tail")
            or not isinstance(output, dict)
        ):
            raise ValueError("invalid diagnostics or structured output")
        if any(key.lower() in _FORBIDDEN_OUTPUT for key in output):
            raise ValueError("authority fields are forbidden")
        if not isinstance(output.get("text"), str):
            raise ValueError("structured text required")
        session_id = output.get("session_id", "")
        if not isinstance(session_id, str):
            raise ValueError("session_id must be a string")
        return AgentRunResult(
            text=output["text"],
            session_id=session_id,
            num_turns=1,
            stderr_tail="",
        )
