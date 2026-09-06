# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU-only regressions for native provider and runtime deployment wiring."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest

from hyperloom.common import codex_session as cs, llm_config


@pytest.mark.parametrize("anthropic", [False, True])
def test_native_routes_actual_kernel_callers(monkeypatch, anthropic):
    from hyperloom.agents.kernel.tools import tracelens_skill_runner, _candidate_review_agent
    from hyperloom.agents.kernel.tools.backends import forge_submit

    monkeypatch.setattr(
        os,
        "environ",
        {
            "INFERENCE_OPTIMIZER_CODEX_AUTH_MODE": "native_oauth",
            **({"ANTHROPIC_BASE_URL": "https://example.invalid"} if anthropic else {}),
        },
    )
    assert tracelens_skill_runner._should_use_codex_runner()
    assert forge_submit._openai_only_provider()
    assert _candidate_review_agent._resolve_backend() == "codex"
    assert not llm_config.has_openai_side()
    assert not llm_config.is_openai_only()


@pytest.mark.parametrize(
    "env, mixed, expected",
    [
        ({}, False, "claude"),
        ({"OPENAI_BASE_URL": "https://example.invalid"}, False, "codex"),
        ({"ANTHROPIC_BASE_URL": "https://example.invalid"}, False, "claude"),
        ({"OPENAI_BASE_URL": "x", "ANTHROPIC_BASE_URL": "y"}, False, "claude"),
        ({"OPENAI_BASE_URL": "x", "ANTHROPIC_BASE_URL": "y"}, True, "codex"),
    ],
)
def test_transport_preserves_gateway_defaults(env, mixed, expected):
    assert llm_config.resolve_agent_provider(env, prefer_codex_when_mixed=mixed) == expected


def test_explicit_forge_claude_is_preserved(monkeypatch):
    from hyperloom.agents.kernel.tools.backends import forge_submit

    monkeypatch.setenv("INFERENCE_OPTIMIZER_CODEX_AUTH_MODE", "native_oauth")
    monkeypatch.setenv("FORGE_AGENT_BACKEND", "claude")
    assert not forge_submit._openai_only_provider()


def test_executable_precedence_and_validation(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for path in (first, second):
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o700)
    env = {"INFERENCE_OPTIMIZER_CODEX_BIN": str(second), "PATH": str(tmp_path)}
    assert cs.resolve_codex_binary(str(first), env) == str(first)
    assert cs.resolve_codex_binary("", env) == str(second)
    assert cs.resolve_codex_binary("first", env) == str(first)
    assert cs.resolve_codex_binary("", {}) is None
    for path in (tmp_path, tmp_path / "missing"):
        with pytest.raises(cs.CodexSessionUnavailableError):
            cs.resolve_codex_binary(str(path), env)
    first.chmod(0o600)
    with pytest.raises(cs.CodexSessionUnavailableError):
        cs.resolve_codex_binary(str(first), env)


def test_session_override_reaches_sdk(tmp_path, monkeypatch):
    from hyperloom.agents.kernel.tests.test_codex_session import _install_fake_sdk, _gateway_env

    record = _install_fake_sdk(monkeypatch)
    binary = tmp_path / "codex"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o700)
    asyncio.run(
        cs.run_codex_turn(
            prompt="p",
            developer_instructions="i",
            cwd=tmp_path,
            model="gpt-6-astra",
            timeout_sec=5,
            env={**_gateway_env(), "INFERENCE_OPTIMIZER_CODEX_BIN": str(binary)},
        )
    )
    assert record["config"].kwargs["codex_bin"] == str(binary)


def test_quantization_cli_handoff(tmp_path, monkeypatch):
    from hyperloom.inference_optimizer.cli import quantization
    from hyperloom.inference_optimizer.session import paths
    from hyperloom.orchestrator.phases import quantization_request_handlers as handler

    captured = {}

    async def fake(**kwargs):
        captured.update(kwargs)
        return str(tmp_path / "quantized")

    monkeypatch.setenv("HYPERLOOM_QUANTIZE_ENABLED", "1")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CODEX_AUTH_MODE", "native_oauth")
    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)
    monkeypatch.setattr(handler, "run_quantization_prelude_async", fake)
    asyncio.run(
        quantization._run_quantization_prelude(
            SimpleNamespace(model="/models/source", quantize="fp8", codex_model="gpt-6-astra", claude_model="other")
        )
    )
    assert captured["provider"] == "codex"
    assert captured["model"] == "gpt-6-astra"


def test_quantization_adapter_handoff(tmp_path, monkeypatch):
    from hyperloom.agents import quantization
    from hyperloom.orchestrator.phases import quantization_request_handlers as handler

    captured = {}

    async def fake(prompt, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            status="success",
            quantized_model_dir=tmp_path / "quantized",
            assessment=SimpleNamespace(final="success", eval_gap=0),
        )

    monkeypatch.setattr(quantization, "quantize_via_prompt", fake)
    asyncio.run(
        handler.run_quantization_prelude_async(
            prompt="fp8", source_model="/models/source", workspace=tmp_path, provider="codex", model="gpt-6-astra"
        )
    )
    assert captured["provider"] == "codex"
    assert captured["model"] == "gpt-6-astra"


def test_specialist_deployment_binary(tmp_path, monkeypatch):
    from hyperloom.orchestrator.specialists.subprocess_ import resolve_codex_executable

    binary = tmp_path / "codex"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o700)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CODEX_BIN", str(binary))
    assert resolve_codex_executable() == str(binary)
    with pytest.raises(cs.CodexSessionUnavailableError):
        resolve_codex_executable(str(tmp_path / "missing"))


def test_forge_deployment_binary(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "agents/kernel/tools/backends"))
    from hyperloom.agents.kernel.tools.backends import forge_submit

    binary = tmp_path / "codex"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o700)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CODEX_AUTH_MODE", "native_oauth")
    monkeypatch.delenv("FORGE_AGENT_BACKEND", raising=False)
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "local")
    forge_submit._reset_knowledge_config_cache()
    env = {"INFERENCE_OPTIMIZER_CODEX_BIN": str(binary)}
    forge_submit._apply_kernel_backend_env(env)
    assert env["FORGE_AGENT_CLI"] == str(binary)
    env["FORGE_AGENT_CLI"] = str(tmp_path / "missing")
    with pytest.raises(cs.CodexSessionUnavailableError):
        forge_submit._apply_kernel_backend_env(env)
    forge_submit._reset_knowledge_config_cache()


def test_ray_forwards_native_runtime_selection(monkeypatch):
    from hyperloom.agents.kernel.tools.backends import ray_runtime

    pins = {
        "INFERENCE_OPTIMIZER_CODEX_AUTH_MODE": "native_oauth",
        "INFERENCE_OPTIMIZER_CODEX_HOME": "/campaign/auth",
        "INFERENCE_OPTIMIZER_CODEX_BIN": "/campaign/bundle/bin/codex",
        "CODEX_MODEL": "gpt-6-astra",
        "FORGE_AGENT_BACKEND": "codex",
        "FORGE_AGENT_CLI": "/campaign/bundle/bin/codex",
        "FORGE_AGENT_MODEL": "gpt-6-astra",
        "FORGE_AGENT_FALLBACK_PROVIDER": "none",
        "FORGE_AGENT_FALLBACK_MODEL": "none",
        "FORGE_AGENT_OPTIONS_JSON": '{"auth_mode":"native_oauth","home":"/campaign/auth"}',
        "KERNEL_AGENTS_MODEL": "gpt-6-astra",
    }
    monkeypatch.setattr(os, "environ", pins)
    forwarded = ray_runtime.safe_runtime_env()["env_vars"]
    assert {key: forwarded.get(key) for key in pins} == pins


def test_forge_request_native_provider_and_explicit_model():
    from hyperloom.orchestrator.kernel.request_handlers import _resolve_forge_agent

    env = {"INFERENCE_OPTIMIZER_CODEX_AUTH_MODE": "native_oauth", "CODEX_MODEL": "gpt-6-astra"}
    assert _resolve_forge_agent({}, env=env) == ("codex", "gpt-6-astra")
    assert _resolve_forge_agent({"agent_backend": "claude", "llm_model": "chosen"}, env=env) == ("claude", "chosen")


def test_fusion_runtime_override(tmp_path, monkeypatch):
    from hyperloom.agents.kernel.tools import forge_fusion

    binary = tmp_path / "codex"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o700)
    monkeypatch.setattr(os, "environ", {"INFERENCE_OPTIMIZER_CODEX_BIN": str(binary)})
    forge_fusion._inject_author_gateway_env("codex")
    assert os.environ["FORGE_AGENT_CLI"] == str(binary)


def test_collective_native_runtime_does_not_alias_credentials(tmp_path, monkeypatch):
    from hyperloom.agents.kernel.tools import forge_collective

    binary = tmp_path / "codex"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o700)
    monkeypatch.setattr(os, "environ", {"INFERENCE_OPTIMIZER_CODEX_BIN": str(binary)})
    forge_collective._inject_author_gateway_env("codex")
    assert os.environ == {"INFERENCE_OPTIMIZER_CODEX_BIN": str(binary), "FORGE_AGENT_CLI": str(binary)}
