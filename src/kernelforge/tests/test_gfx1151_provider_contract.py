# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RED contract tests for KernelForge's endpoint-agnostic provider adapter."""

import asyncio

import pytest

from kernelforge.agent_backends.base import (
    AgentRunResult,
    AgentRunSpec,
    AgentRuntimeConfig,
    AgentToolPolicy,
)
from kernelforge.agent_backends.endpoint_agnostic import EndpointAgnosticBackend


RUNTIME = AgentRuntimeConfig(
    provider="endpoint_agnostic",
    model="explicit-model",
    fallback_provider="",
    options={
        "provider": "openai_compatible",
        "protocol": "openai_compatible",
        "api_key_env": "TEST_PROVIDER_KEY",
        "base_url": "http://runner.invalid/v1",
        "capabilities": ["coder", "tools", "structured_output", "session_resume"],
        "egress": True,
        "fallback": "none",
    },
)


def _result(request):
    return {
        "schema": "endpoint_agnostic_runner_v1.result",
        "request_id": request["request_id"],
        "provider": request["provider"],
        "protocol": request["protocol"],
        "model": request["model"],
        "status": "success",
        "structured_output": {"text": "candidate source", "session_id": "resume-7"},
        "attempts": 1,
        "timing": {"elapsed_seconds": 0.01},
        "capability_receipt": request["capabilities"],
        "diagnostics": {"stderr_tail": ""},
        "fallback_used": False,
        "promotion_authority": False,
    }


def test_run_maps_spec_to_exact_runner_v1_and_result_to_agent_result():
    seen = []

    async def runner(request):
        seen.append(request)
        return _result(request)

    spec = AgentRunSpec(
        system_prompt="system",
        user_prompt="user",
        cwd="/work/session-7",
        model="run-model",
        timeout_sec=17,
        writable=True,
        target_files=["kernel.cpp"],
        env={"SESSION_ID": "resume-7"},
        tool_policy=AgentToolPolicy(read=True, write=True, shell=True),
    )
    result = asyncio.run(EndpointAgnosticBackend(RUNTIME, runner).run(spec))
    assert len(seen) == 1
    request = seen[0]
    assert set(request) == {
        "schema", "request_id", "provider", "protocol", "base_url", "api_key_env",
        "model", "capabilities", "sandbox", "timeout_seconds", "retry", "egress",
        "environment", "messages", "output_schema", "fallback",
    }
    assert isinstance(request["request_id"], str) and request["request_id"]
    assert request["schema"] == "endpoint_agnostic_runner_v1.request"
    assert request["provider"] == "openai_compatible"
    assert request["protocol"] == "openai_compatible"
    assert request["model"] == "run-model"
    assert request["messages"] == [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}]
    assert request["base_url"] == "http://runner.invalid/v1"
    assert request["api_key_env"] == "TEST_PROVIDER_KEY"
    assert request["sandbox"] == {"mode": "workspace_write", "writable_roots": ["/work/session-7"]}
    assert request["timeout_seconds"] == 17
    assert request["retry"] == {"max_attempts": 1}
    assert request["egress"] is True
    assert request["output_schema"] == {
        "type": "object",
        "required": ["text"],
        "properties": {"text": {"type": "string"}, "session_id": {"type": "string"}},
    }
    assert request["fallback"] == "none"
    assert request["environment"] == {"SESSION_ID": "resume-7"}
    assert result.text == "candidate source"
    assert result.session_id == "resume-7"


def test_runner_options_require_explicit_provider_protocol_model_and_fallback_none():
    with pytest.raises((ValueError, TypeError)):
        EndpointAgnosticBackend(
            AgentRuntimeConfig(provider="endpoint_agnostic", model="m", options={}),
            lambda request: _result(request),
        )


@pytest.mark.parametrize("bad", [None, {}, {"schema": "wrong"}, {"status": "success"}])
def test_malformed_or_empty_runner_result_is_rejected(bad):
    async def runner(_request):
        return bad

    with pytest.raises((ValueError, TypeError)):
        asyncio.run(EndpointAgnosticBackend(RUNTIME, runner).run(AgentRunSpec("s", "u", "/tmp")))


def test_runner_error_is_not_retried_or_cross_provider_fallback_used():
    calls = []

    async def runner(request):
        calls.append(request)
        raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError):
        asyncio.run(EndpointAgnosticBackend(RUNTIME, runner).run(AgentRunSpec("s", "u", "/tmp")))
    assert len(calls) == 1


def test_result_cannot_claim_deterministic_correctness_or_promotion_authority():
    async def runner(request):
        result = _result(request)
        result["structured_output"] = {"text": "x", "KEEP": True, "correct": True}
        result["promotion_authority"] = True
        return result

    with pytest.raises((ValueError, TypeError)):
        asyncio.run(EndpointAgnosticBackend(RUNTIME, runner).run(AgentRunSpec("s", "u", "/tmp")))


def test_structured_output_cannot_override_canonical_evaluator():
    async def runner(request):
        result = _result(request)
        result["structured_output"] = {"text": "x", "KEEP": True, "correct": True}
        return result

    with pytest.raises((ValueError, TypeError)):
        asyncio.run(EndpointAgnosticBackend(RUNTIME, runner).run(AgentRunSpec("s", "u", "/tmp")))


def test_credential_value_never_appears_in_request_error_or_normalized_result(monkeypatch):
    secret = "super-secret-token"
    monkeypatch.setenv("TEST_PROVIDER_KEY", secret)
    seen = []

    async def runner(request):
        seen.append(repr(request))
        response = _result(request)
        response["diagnostics"]["stderr_tail"] = secret
        return response

    spec = AgentRunSpec("system", "ordinary prompt", "/tmp", env={"SESSION_ID": "session-1"})
    with pytest.raises((ValueError, TypeError)) as caught:
        asyncio.run(EndpointAgnosticBackend(RUNTIME, runner).run(spec))
    assert secret not in " ".join(seen)
    assert secret not in str(caught.value)


def test_secret_named_session_environment_is_rejected_before_runner_call():
    calls = []

    async def runner(request):
        calls.append(request)
        return _result(request)

    spec = AgentRunSpec("s", "u", "/tmp", env={"API_TOKEN": "secret"})
    with pytest.raises((ValueError, TypeError)):
        asyncio.run(EndpointAgnosticBackend(RUNTIME, runner).run(spec))
    assert calls == []
