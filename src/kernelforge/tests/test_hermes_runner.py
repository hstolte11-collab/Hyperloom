# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""RED contracts for the isolated Hermes endpoint runner."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from kernelforge.agent_backends.hermes_runner import (
    HermesRunnerError,
    HermesOneShotRunner,
)


REQUEST_KEYS = {
    "schema", "request_id", "provider", "protocol", "base_url", "api_key_env",
    "model", "capabilities", "sandbox", "timeout_seconds", "retry", "egress",
    "environment", "messages", "output_schema", "fallback",
}
RESULT_KEYS = {
    "schema", "request_id", "provider", "protocol", "model", "status",
    "structured_output", "attempts", "timing", "capability_receipt",
    "diagnostics", "fallback_used", "promotion_authority",
}


def request(**changes):
    value = {
        "schema": "endpoint_agnostic_runner_v1.request",
        "request_id": "attempt-hermes-codex",
        "provider": "hermes",
        "protocol": "hermes_oneshot",
        "base_url": None,
        "api_key_env": None,
        "model": "gpt-5.6-sol",
        "capabilities": ["coder", "structured_output"],
        "sandbox": {"mode": "read_only", "writable_roots": []},
        "timeout_seconds": 90,
        "retry": {"max_attempts": 1},
        "egress": True,
        "environment": {},
        "messages": [
            {"role": "system", "content": "Return full candidate source only."},
            {"role": "user", "content": "Implement candidate_kernel."},
        ],
        "output_schema": {
            "type": "object",
            "required": ["text"],
            "properties": {"text": {"type": "string"}},
        },
        "fallback": "none",
    }
    value.update(changes)
    return value


def make_profile(tmp_path: Path) -> Path:
    root = tmp_path / "profiles" / "hyperloomcandidate"
    root.mkdir(parents=True)
    (root / "config.yaml").write_text(
        "model:\n"
        "  default: gpt-5.6-sol\n"
        "  provider: openai-codex\n"
        "fallback_providers: []\n"
        "platform_toolsets:\n"
        "  cli: []\n"
        "mcp_servers: {}\n"
        "plugins: {}\n"
    )
    return root


def make_executable(tmp_path: Path) -> Path:
    path = tmp_path / "hermes"
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def success_runner(captured):
    def run(argv, *, cwd, env, timeout):
        captured.update(argv=list(argv), cwd=cwd, env=dict(env), timeout=timeout)
        usage_path = Path(argv[argv.index("--usage-file") + 1])
        usage_path.write_text(json.dumps({
            "estimated_cost_usd": 0.0,
            "cost_status": "unavailable",
            "cost_source": "unavailable",
            "input_tokens": 120,
            "output_tokens": 40,
            "cache_read_tokens": 10,
            "cache_write_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 160,
            "api_calls": 1,
            "model": "gpt-5.6-sol",
            "provider": "openai-codex",
            "session_id": "session-hermes-1",
            "completed": True,
            "failed": False,
            "service_tier": None,
        }) + "\n")
        return subprocess.CompletedProcess(argv, 0, "__global__ void candidate_kernel() {}\n", "")
    return run


def build_runner(tmp_path: Path, command_runner):
    return HermesOneShotRunner(
        executable=make_executable(tmp_path),
        profile="hyperloomcandidate",
        profile_root=make_profile(tmp_path),
        inference_provider="openai-codex",
        model="gpt-5.6-sol",
        evidence_root=tmp_path / "evidence",
        command_runner=command_runner,
    )


def test_hermes_runner_emits_exact_tool_free_safe_mode_command_and_result(tmp_path):
    captured = {}
    runner = build_runner(tmp_path, success_runner(captured))

    result = asyncio.run(runner(request()))

    assert set(result) == RESULT_KEYS
    assert result["status"] == "success"
    assert result["structured_output"] == {"text": "__global__ void candidate_kernel() {}"}
    assert result["attempts"] == 1
    assert result["fallback_used"] is False
    assert result["promotion_authority"] is False
    assert result["capability_receipt"] == ["coder", "structured_output"]
    argv = captured["argv"]
    assert argv[0] == str(runner.executable)
    assert argv[1:3] == ["-p", "hyperloomcandidate"]
    assert argv.count("-z") == 1
    assert argv[argv.index("--provider") + 1] == "openai-codex"
    assert argv[argv.index("--model") + 1] == "gpt-5.6-sol"
    assert argv[argv.index("--toolsets") + 1] == "none"
    assert "--safe-mode" in argv
    assert "--usage-file" in argv
    assert "--yolo" not in argv
    assert "claude" not in " ".join(argv).lower()
    assert all("TOKEN" not in key and "KEY" not in key and "SECRET" not in key for key in captured["env"])
    assert captured["timeout"] == 90


def test_hermes_runner_writes_closed_evidence_and_refuses_request_id_reuse(tmp_path):
    runner = build_runner(tmp_path, success_runner({}))
    value = request()

    first = asyncio.run(runner(value))
    attempt = runner.evidence_root / value["request_id"]
    assert first["status"] == "success"
    assert sorted(path.name for path in attempt.iterdir()) == [
        "request.json", "result.json", "stderr.log", "stdout.log", "usage.json"
    ]
    assert json.loads((attempt / "request.json").read_text()) == value
    assert json.loads((attempt / "result.json").read_text()) == first
    with pytest.raises(HermesRunnerError, match="already exists"):
        asyncio.run(runner(value))


def test_hermes_runner_rejects_request_contract_drift_before_process(tmp_path):
    calls = []
    runner = build_runner(tmp_path, lambda *a, **k: calls.append((a, k)))
    mutations = []
    mutations.append({**request(), "unexpected": 1})
    mutations.append(request(provider="openai-codex"))
    mutations.append(request(protocol="custom_command"))
    mutations.append(request(model="gpt-5.5"))
    mutations.append(request(fallback="codex"))
    mutations.append(request(retry={"max_attempts": 2}))
    mutations.append(request(api_key_env="OPENAI_API_KEY"))
    mutations.append(request(environment={"OPENAI_API_KEY": "secret"}))
    mutations.append(request(sandbox={"mode": "workspace_write", "writable_roots": ["/tmp"]}))
    mutations.append(request(capabilities=["coder", "tools", "structured_output"]))
    mutations.append(request(egress=False))
    for index, bad in enumerate(mutations):
        bad["request_id"] = f"bad-{index}"
        with pytest.raises(HermesRunnerError):
            asyncio.run(runner(bad))
    assert calls == []


def test_hermes_runner_rejects_bad_profile_or_executable(tmp_path):
    profile = make_profile(tmp_path)
    executable = make_executable(tmp_path)
    evidence = tmp_path / "evidence"
    link = tmp_path / "hermes-link"; link.symlink_to(executable)
    with pytest.raises(HermesRunnerError, match="executable.*symlink"):
        HermesOneShotRunner(executable=link, profile="hyperloomcandidate", profile_root=profile,
                            inference_provider="openai-codex", model="gpt-5.6-sol",
                            evidence_root=evidence, command_runner=success_runner({}))
    bad_profile = tmp_path / "profiles" / "bad"
    bad_profile.mkdir()
    (bad_profile / "config.yaml").write_text("fallback_providers:\n  - provider: claude\n    model: x\n")
    with pytest.raises(HermesRunnerError, match="profile"):
        HermesOneShotRunner(executable=executable, profile="bad", profile_root=bad_profile,
                            inference_provider="openai-codex", model="gpt-5.6-sol",
                            evidence_root=evidence, command_runner=success_runner({}))


def test_hermes_runner_preserves_failure_without_fallback_claim(tmp_path):
    def fail(argv, *, cwd, env, timeout):
        usage = Path(argv[argv.index("--usage-file") + 1])
        usage.write_text(json.dumps({"provider":"openai-codex", "model":"gpt-5.6-sol",
                                     "api_calls":1, "completed":False, "failed":True}) + "\n")
        return subprocess.CompletedProcess(argv, 9, "", "provider unavailable")
    runner = build_runner(tmp_path, fail)
    with pytest.raises(HermesRunnerError, match="exit 9"):
        asyncio.run(runner(request()))
    attempt = runner.evidence_root / "attempt-hermes-codex"
    assert (attempt / "request.json").is_file()
    assert (attempt / "usage.json").is_file()
    assert (attempt / "stderr.log").read_text() == "provider unavailable"
    assert not (attempt / "result.json").exists()


@pytest.mark.parametrize("changes", [
    {"provider":"openrouter"}, {"model":"gpt-5.5"}, {"api_calls":2},
    {"completed":False}, {"failed":True}, {"session_id":""},
])
def test_hermes_runner_rejects_usage_identity_or_completion_drift(tmp_path, changes):
    def run(argv, *, cwd, env, timeout):
        usage = {"provider":"openai-codex", "model":"gpt-5.6-sol", "api_calls":1,
                 "completed":True, "failed":False, "session_id":"session-1",
                 "input_tokens":1, "output_tokens":1, "total_tokens":2}
        usage.update(changes)
        Path(argv[argv.index("--usage-file") + 1]).write_text(json.dumps(usage) + "\n")
        return subprocess.CompletedProcess(argv, 0, "source\n", "")
    runner = build_runner(tmp_path, run)
    with pytest.raises(HermesRunnerError, match="usage"):
        asyncio.run(runner(request()))
