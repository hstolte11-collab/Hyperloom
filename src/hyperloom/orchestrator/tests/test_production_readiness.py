"""Production-readiness control-plane contracts for gfx1151 optimization."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from hyperloom.orchestrator.candidate_control import bind_magpie, bind_tracelens
from hyperloom.orchestrator.kernel_target_registry import bind_kernel_target, load_registry
from hyperloom.orchestrator.production_readiness import (
    ProductionReadinessControl,
    ProductionReadinessError,
)

ROOT = Path(__file__).resolve().parents[4]
REGISTRY = ROOT / "src/hyperloom/orchestrator/kernel-targets-gfx1151-v1.json"
PROVIDERS = ROOT / "src/hyperloom/orchestrator/production-providers-gfx1151-v1.json"
VLLM = Path(os.environ.get("HYPERLOOM_TEST_VLLM_SOURCE_ROOT", "/__missing_vllm_source__"))
SGLANG = Path(os.environ.get("HYPERLOOM_TEST_SGLANG_SOURCE_ROOT", "/__missing_sglang_source__"))
pytestmark = pytest.mark.skipif(
    not (VLLM.is_dir() and SGLANG.is_dir()),
    reason="set HYPERLOOM_TEST_{VLLM,SGLANG}_SOURCE_ROOT for external registry integration tests",
)
ROOTS = {"vllm_rocm10": VLLM, "sglang_async_v7": SGLANG}
RUNNER_HASH = "42884f444743101616eb8cc1f94db7466c8c79a37d60736ae29be7d8e424d4cf"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _handoff(tmp_path: Path, attempt_id="attempt-0001", provider="codex"):
    registry = load_registry(REGISTRY, source_roots=ROOTS)
    binding = bind_kernel_target(
        registry,
        target_id="gfx1151-w4a4-dot-baseline",
        framework="vllm",
    )
    source = tmp_path / "candidate.hip"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text('extern "C" __global__ void candidate() {}\n')
    trace = tmp_path / "trace.md"
    trace.write_text("analysis\n")
    candidates = tmp_path / "candidates.json"
    candidates.write_text("{}\n")
    protocols = {"codex": "codex_app_server", "hermes": "hermes_oneshot"}
    return {
        "schema": "hyperloom.candidate-handoff.v4",
        "attempt_id": attempt_id,
        "parent_attempt_id": None,
        "target": {
            "isa": "gfx1151",
            "board": "strix-halo-radeon-8060s",
            "rocm_root": "/opt/rocm/core-10.0",
            "fallback": "none",
        },
        "kernel_target": binding,
        "operation": {
            "name": "w4a4_dot_baseline",
            "shape": [4, 64, 256],
            "dtype": "bfloat16",
            "layout": "rowmajor",
        },
        "source": {
            "allowed_root": str(tmp_path),
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": _sha(source),
            "repository_commit": "a" * 40,
        },
        "provider": {
            "runner_contract_sha256": RUNNER_HASH,
            "provider": provider,
            "protocol": protocols[provider],
            "model": "gpt-5.6-sol",
            "fallback": "none",
        },
        "compiler": {
            "rocm_root": "/opt/rocm/core-10.0",
            "offload_arch": "gfx1151",
            "command": ["hipcc", "candidate.hip", "-o", "candidate.so"],
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


def _control(tmp_path: Path) -> ProductionReadinessControl:
    registry = load_registry(REGISTRY, source_roots=ROOTS)
    proof = tmp_path / "FUNCTIONAL-TARGET-MATRIX.json"
    proof.write_text(
        json.dumps(
            {
                "schema": "hyperloom.gfx1151-functional-target-matrix.v3",
                "status": "PASS",
                "registry_id": registry.registry_id,
                "registry_sha256": registry.manifest_sha256,
                "target_count": len(registry.targets),
                "targets": [
                    {
                        "target_id": target_id,
                        "status": "CONTROL_ONLY" if target.status == "control_only" else "PASS",
                        "fallback_used": False,
                        "promotion_authority": False,
                    }
                    for target_id, target in sorted(registry.targets.items())
                ],
                "promotion_authority": False,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return ProductionReadinessControl(
        root=tmp_path / "control",
        registry_path=REGISTRY,
        source_roots=ROOTS,
        provider_roster_path=PROVIDERS,
        functional_proof_path=proof,
    )


def test_initialize_and_status_bind_exact_registry_providers_and_manual_activation(tmp_path):
    control = _control(tmp_path)
    epoch = control.initialize()
    assert epoch["status"] == "READY_NOT_ACTIVATED"
    assert epoch["registry"]["registry_id"] == "gfx1151-rocm10-kernel-targets-v3"
    assert epoch["registry"]["target_count"] == 23
    assert [row["id"] for row in epoch["providers"]] == [
        "codex-native-oauth",
        "hermes-openai-codex",
    ]
    assert all(row["model"] == "gpt-5.6-sol" for row in epoch["providers"])
    assert epoch["provider_fallback"] == "none"
    assert epoch["model_fallback"] == "none"
    assert epoch["kernel_fallback"] == "none"
    assert epoch["functional_proof"]["target_count"] == 23
    assert epoch["activation"] == "manual_external_approval_required"
    assert epoch["promotion_authority"] is False
    assert control.initialize() == epoch
    status = control.status()
    assert status["healthy"] is True
    assert status["counts"] == {"pending": 0, "retryable": 0, "complete": 0}
    assert not hasattr(control, "promote")


def test_create_attempt_requires_exact_provider_and_kernel_binding(tmp_path):
    control = _control(tmp_path)
    control.initialize()
    handoff = _handoff(tmp_path / "handoff")
    attempt = control.create_attempt(handoff)
    assert attempt.name == "attempt-0001"
    assert control.status()["counts"]["pending"] == 1
    bad = _handoff(tmp_path / "bad", attempt_id="attempt-bad")
    bad["provider"]["model"] = "fallback-model"
    with pytest.raises(ProductionReadinessError, match="provider roster"):
        control.create_attempt(bad)


def test_failure_is_immutable_and_resume_allocates_new_parent_bound_attempt(tmp_path):
    control = _control(tmp_path)
    control.initialize()
    first = control.create_attempt(_handoff(tmp_path / "first"))
    failure = control.record_failure("attempt-0001", category="provider_failure", message="bounded failure")
    assert failure["status"] == "RETRYABLE"
    assert control.status()["counts"]["retryable"] == 1
    second = control.resume("attempt-0001", "attempt-0002")
    assert second.name == "attempt-0002"
    assert (first / "failure.json").is_file()
    assert control.status()["counts"] == {"pending": 1, "retryable": 1, "complete": 0}
    with pytest.raises(ProductionReadinessError):
        control.resume("attempt-0001", "attempt-0002")


def test_completed_attempt_requires_cleanup_receipt_and_stays_nonpromoting(tmp_path):
    control = _control(tmp_path)
    control.initialize()
    control.create_attempt(_handoff(tmp_path / "done"))

    def generator(request):
        return {
            "schema": "endpoint_agnostic_runner_v1.result",
            "request_id": request["request_id"],
            "provider": request["provider"],
            "protocol": request["protocol"],
            "model": request["model"],
            "status": "success",
            "structured_output": {"text": "candidate source"},
            "attempts": 1,
            "timing": {},
            "capability_receipt": request["capabilities"],
            "diagnostics": {"stderr_tail": ""},
            "fallback_used": False,
            "promotion_authority": False,
        }

    result = control.run(
        "attempt-0001",
        generator=generator,
        compiler=lambda text, spec: {"status": "PASS", "binary_sha256": "d" * 64},
        evaluator=lambda binary, plan: {
            "correctness": {"status": "PASS", "mismatches": 0},
            "performance": {"status": "NOT_REQUIRED"},
            "abba": {"status": "NOT_REQUIRED", "minimum_ratio": 0.0},
        },
    )
    assert result["status"] == "REJECT"
    assert control.status()["counts"]["pending"] == 1
    cleanup = control.record_cleanup(
        "attempt-0001",
        {
            "kfd_clear": True,
            "containers_absent": True,
            "listeners_clear": True,
            "lock_released": True,
            "production_mutated": False,
            "promotion_authority": False,
        },
    )
    assert cleanup["status"] == "PASS"
    status = control.status()
    assert status["counts"] == {"pending": 0, "retryable": 0, "complete": 1}
    assert status["production_activated"] is False
