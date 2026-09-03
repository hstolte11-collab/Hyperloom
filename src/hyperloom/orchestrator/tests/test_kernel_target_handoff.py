"""Kernel-target binding contracts for candidate-handoff v4."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from hyperloom.orchestrator.candidate_control import (
    CandidateControlError,
    CandidateControlPlane,
    bind_magpie,
    bind_tracelens,
    build_agent_request,
    strict_json_file,
)
from hyperloom.orchestrator.kernel_target_registry import (
    KernelTargetRegistryError,
    bind_kernel_target,
    load_registry,
)

ROOT = Path(__file__).resolve().parents[4]
REGISTRY_PATH = ROOT / "src/hyperloom/orchestrator/kernel-targets-gfx1151-v1.json"
VLLM = Path(os.environ.get("HYPERLOOM_TEST_VLLM_SOURCE_ROOT", "/__missing_vllm_source__"))
SGLANG = Path(os.environ.get("HYPERLOOM_TEST_SGLANG_SOURCE_ROOT", "/__missing_sglang_source__"))
pytestmark = pytest.mark.skipif(
    not (VLLM.is_dir() and SGLANG.is_dir()),
    reason="set HYPERLOOM_TEST_{VLLM,SGLANG}_SOURCE_ROOT for external registry integration tests",
)
RUNNER_HASH = "42884f444743101616eb8cc1f94db7466c8c79a37d60736ae29be7d8e424d4cf"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _handoff(tmp_path: Path, target_id: str = "vllm-rdna-hybrid-w4a16-decode"):
    registry = load_registry(
        REGISTRY_PATH,
        source_roots={"vllm_rocm10": VLLM, "sglang_async_v7": SGLANG},
    )
    framework = registry.targets[target_id].framework
    binding = bind_kernel_target(registry, target_id=target_id, framework=framework)
    source = tmp_path / "candidate" / "candidate.py"
    source.parent.mkdir(parents=True)
    source.write_text("def candidate(x):\n    return x\n")
    trace = tmp_path / "trace.md"
    trace.write_text("analysis\n")
    candidates = tmp_path / "candidates.json"
    candidates.write_text("{}\n")
    return {
        "schema": "hyperloom.candidate-handoff.v4",
        "attempt_id": "attempt-0001",
        "parent_attempt_id": None,
        "target": {
            "isa": "gfx1151",
            "board": "strix-halo-radeon-8060s",
            "rocm_root": "/opt/rocm/core-10.0",
            "fallback": "none",
        },
        "kernel_target": binding,
        "operation": {
            "name": "w4a16_decode",
            "shape": [1, 4096, 4096],
            "dtype": "bfloat16",
            "layout": "contiguous",
        },
        "source": {
            "allowed_root": str(source.parent),
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": _sha(source),
            "repository_commit": "a" * 40,
        },
        "provider": {
            "runner_contract_sha256": RUNNER_HASH,
            "provider": "custom",
            "protocol": "custom_command",
            "model": "explicit-test-model",
            "fallback": "none",
        },
        "compiler": {
            "rocm_root": "/opt/rocm/core-10.0",
            "offload_arch": "gfx1151",
            "command": ["/opt/rocm/core-10.0/bin/hipcc", "candidate.cpp", "-o", "candidate.so"],
            "hsa_override_gfx_version": "forbidden",
        },
        "evaluation_plan": {
            "correctness": "deterministic_reference",
            "performance": "optional",
            "repetitions": 1,
            "promotion_margin": 0.0,
        },
        "tracelens": bind_tracelens(trace, candidates),
        "magpie": bind_magpie(
            {
                "status": "already_patched",
                "script_sha256": "b" * 64,
                "patch_sha256": "c" * 64,
                "promotion_authority": False,
            }
        ),
        "explore_policy": {
            "label": "arbor-pattern",
            "mode": "explore",
            "budget": 1,
            "is_source_module": False,
        },
        "promotion_authority": False,
    }


def test_kernel_target_binding_is_hash_closed_and_propagated_to_agent(tmp_path):
    value = _handoff(tmp_path)
    binding = value["kernel_target"]
    assert binding["registry_id"] == "gfx1151-rocm10-kernel-targets-v3"
    assert binding["target_id"] == "vllm-rdna-hybrid-w4a16-decode"
    assert binding["framework"] == "vllm"
    assert binding["candidate_contract"]["target_id"] == binding["target_id"]
    plane = CandidateControlPlane(tmp_path / "state")
    attempt = plane.create_attempt(value)
    assert strict_json_file(attempt / "candidate-handoff.json") == value

    request = build_agent_request(
        value,
        messages=[{"role": "user", "content": "author a bounded candidate"}],
        sandbox_root=value["source"]["allowed_root"],
    )
    assert request["environment"]["KERNEL_TARGET_ID"] == binding["target_id"]
    assert request["environment"]["KERNEL_FRAMEWORK"] == "vllm"
    assert request["environment"]["KERNEL_TARGET_CONTRACT_SHA256"] == binding["candidate_contract_sha256"]
    assert request["fallback"] == "none"


def test_kernel_target_binding_rejects_tampering_and_noneditable_targets(tmp_path):
    for mutation in ("target", "contract_hash", "fallback", "contract_target"):
        value = _handoff(tmp_path / mutation)
        binding = value["kernel_target"]
        if mutation == "target":
            binding["target_id"] = "missing"
        elif mutation == "contract_hash":
            binding["candidate_contract_sha256"] = "0" * 64
        elif mutation == "fallback":
            binding["fallback"] = "auto"
        else:
            binding["candidate_contract"]["target_id"] = "other"
        with pytest.raises(CandidateControlError):
            CandidateControlPlane(tmp_path / f"state-{mutation}").create_attempt(value)

    registry = load_registry(
        REGISTRY_PATH,
        source_roots={"vllm_rocm10": VLLM, "sglang_async_v7": SGLANG},
    )
    with pytest.raises(KernelTargetRegistryError, match="control-only"):
        bind_kernel_target(
            registry,
            target_id="gfx1151-w4a4-python-fallback-controls",
            framework="vllm",
        )
    with pytest.raises(KernelTargetRegistryError, match="not usable"):
        bind_kernel_target(
            registry,
            target_id="gfx1151-w8a8-triton-scaled-mm",
            framework="sglang",
        )


def test_sglang_native_target_binds_into_handoff_v4(tmp_path):
    value = _handoff(tmp_path, "sglang-gfx1151-w4a16")
    assert value["kernel_target"]["framework"] == "sglang"
    assert value["kernel_target"]["source_lineage"] == "sglang_async_v7"
    plane = CandidateControlPlane(tmp_path / "state-sglang")
    attempt = plane.create_attempt(value)
    assert strict_json_file(attempt / "candidate-handoff.json") == value


@pytest.mark.parametrize(
    "provider,protocol",
    [("codex", "codex_app_server"), ("hermes", "hermes_oneshot")],
)
def test_qualified_source_only_providers_receive_tool_free_target_bound_requests(tmp_path, provider, protocol):
    value = _handoff(tmp_path / provider)
    value["provider"] = {
        "runner_contract_sha256": RUNNER_HASH,
        "provider": provider,
        "protocol": protocol,
        "model": "gpt-5.6-sol",
        "fallback": "none",
    }
    request = build_agent_request(
        value,
        messages=[{"role": "user", "content": "author source only"}],
        sandbox_root=value["source"]["allowed_root"],
    )
    assert request["provider"] == provider
    assert request["protocol"] == protocol
    assert request["model"] == "gpt-5.6-sol"
    assert request["capabilities"] == ["coder", "structured_output"]
    assert request["sandbox"] == {"mode": "read_only", "writable_roots": []}
    assert request["egress"] is True
    assert request["environment"] == {}
    assert request["retry"] == {"max_attempts": 1}
    assert request["fallback"] == "none"
    rendered = "\n".join(message["content"] for message in request["messages"])
    assert value["kernel_target"]["target_id"] in rendered
    assert value["kernel_target"]["candidate_contract_sha256"] in rendered
