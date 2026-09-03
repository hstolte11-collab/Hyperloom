"""Fail-closed contracts for the gfx1151 kernel-target registry."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hyperloom.orchestrator.kernel_target_registry import (
    KernelTargetRegistryError,
    candidate_contract,
    load_registry,
    select_target,
)

ROOT = Path(__file__).resolve().parents[4]
REGISTRY = ROOT / "src/hyperloom/orchestrator/kernel-targets-gfx1151-v1.json"
VLLM = Path(os.environ.get("HYPERLOOM_TEST_VLLM_SOURCE_ROOT", "/__missing_vllm_source__"))
SGLANG = Path(os.environ.get("HYPERLOOM_TEST_SGLANG_SOURCE_ROOT", "/__missing_sglang_source__"))
pytestmark = pytest.mark.skipif(
    not (VLLM.is_dir() and SGLANG.is_dir()),
    reason="set HYPERLOOM_TEST_{VLLM,SGLANG}_SOURCE_ROOT for external registry integration tests",
)
SOURCE_ROOTS = {"vllm_rocm10": VLLM, "sglang_async_v7": SGLANG}

TARGET_IDS = {
    "vllm-rdna-hybrid-w4a16-decode",
    "vllm-rdna-hybrid-w4a16-prefill",
    "vllm-triton-w4a16-fallback",
    "gfx1151-w4a4-perf-quantizer",
    "gfx1151-w4a4-m128-v4q-quantizer",
    "gfx1151-w4a4-m1-pair2-dot",
    "gfx1151-w4a4-m128-v3-linear-wmma",
    "gfx1151-w4a4-python-fallback-controls",
    "gfx1151-w4a4-dot-baseline",
    "gfx1151-w4a4-wmma-baseline-padded",
    "gfx1151-w4a4-dot-mtile4",
    "gfx1151-w4a4-dot-mtile8",
    "gfx1151-w4a4-wmma-16x64",
    "gfx1151-w4a4-wmma-32x64",
    "gfx1151-w4a4-wmma-32x32",
    "gfx1151-w8a8-dynamic-token-quantizer",
    "gfx1151-w8a8-triton-scaled-mm",
    "sglang-gfx1151-fp8-e4m3fnuz",
    "sglang-gfx1151-mxfp4-e2m1-e8m0",
    "sglang-gfx1151-w8a8",
    "sglang-gfx1151-w4a8",
    "sglang-gfx1151-w4a4",
    "sglang-gfx1151-w4a16",
}


def test_source_current_registry_closes_exact_roster_and_lineage():
    registry = load_registry(REGISTRY, source_roots=SOURCE_ROOTS)
    assert registry.schema == "hyperloom.kernel-target-registry.v1"
    assert registry.registry_id == "gfx1151-rocm10-kernel-targets-v3"
    assert set(registry.targets) == TARGET_IDS
    assert registry.platform == {
        "isa": "gfx1151",
        "board": "strix-halo-radeon-8060s",
        "rocm_root": "/opt/rocm/core-10.0",
    }
    assert registry.source_lineages["vllm_rocm10"] == {
        "repository_commit": "f27c6456c8819b619b8439e6b96b664482cdaa20",
        "repository_tree": "bd7403d13337ec0832a5e6341bb47979a283a456",
    }
    assert registry.source_lineages["sglang_async_v7"] == {
        "repository_commit": "f63458b5beaceabbd9d749b9fc956370e1b649e6",
        "repository_tree": "cdafa106e874c774da83f03fb7dc7a07ba79eef6",
    }
    assert registry.fallback == "none"
    assert registry.promotion_authority is False


def test_every_target_selects_only_for_its_exact_framework():
    registry = load_registry(REGISTRY, source_roots=SOURCE_ROOTS)
    for target_id in sorted(TARGET_IDS):
        target = registry.targets[target_id]
        other = "sglang" if target.framework == "vllm" else "vllm"
        assert select_target(registry, target_id=target_id, framework=target.framework).id == target_id
        assert target.framework_support[target.framework] in {"native", "control_only"}
        with pytest.raises(KernelTargetRegistryError, match="not usable"):
            select_target(registry, target_id=target_id, framework=other)
    for bad in ("", "auto", "hybrid-w4a16", "missing"):
        with pytest.raises(KernelTargetRegistryError):
            select_target(registry, target_id=bad, framework="vllm")


def test_candidate_contract_exposes_only_bound_edit_paths_and_oracles():
    registry = load_registry(REGISTRY, source_roots=SOURCE_ROOTS)
    for target_id, target in registry.targets.items():
        if target.status == "control_only":
            with pytest.raises(KernelTargetRegistryError, match="control-only"):
                candidate_contract(target)
            continue
        contract = candidate_contract(target)
        assert contract["target_id"] == target_id
        assert contract["framework"] == target.framework
        assert contract["allowed_edit_paths"]
        assert set(contract["allowed_edit_paths"]).issubset(target.source_files)
        assert contract["symbols"]
        assert contract["materialization"]["kind"]
        assert contract["correctness"]["oracle"]
        assert contract["fallback"] == "none"
        assert contract["promotion_authority"] is False


def test_registry_rejects_source_hash_drift_and_unknown_fields(tmp_path):
    payload = json.loads(REGISTRY.read_text())
    first = sorted(payload["files"])[0]
    payload["files"][first]["sha256"] = "0" * 64
    bad_hash = tmp_path / "bad-hash.json"
    bad_hash.write_text(json.dumps(payload, sort_keys=True) + "\n")
    with pytest.raises(KernelTargetRegistryError, match="source file drift"):
        load_registry(bad_hash, source_roots=SOURCE_ROOTS)
    payload = json.loads(REGISTRY.read_text())
    payload["unexpected"] = True
    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps(payload, sort_keys=True) + "\n")
    with pytest.raises(KernelTargetRegistryError, match="key closure"):
        load_registry(extra, source_roots=SOURCE_ROOTS)


def test_registry_closes_non_targets_instead_of_silently_dropping_them():
    registry = load_registry(REGISTRY, source_roots=SOURCE_ROOTS)
    closed = {row["id"]: row["reason"] for row in registry.closed_donors}
    assert set(closed) == {
        "rdna3-w4a16-gfx1100-only",
        "aiter-w8a8-gfx1151-unqualified",
        "task10.8-campaign-variants-not-donors",
        "historical-archive-binaries-not-source-current",
    }
    assert all(reason for reason in closed.values())


def test_w4a4_dispatcher_is_control_only_but_all_native_routes_are_editable():
    registry = load_registry(REGISTRY, source_roots=SOURCE_ROOTS)
    dispatcher = registry.targets["gfx1151-w4a4-python-fallback-controls"]
    assert dispatcher.status == "control_only"
    assert dispatcher.allowed_edit_paths == ()
    assert dispatcher.source_files == frozenset(
        {
            "vllm/model_executor/layers/quantization/gfx1151_w4a4_archive.py",
            "tests/quantization/test_gfx1151_w4a4_archive.py",
            "tests/quantization/test_gfx1151_archive_trace_integrity.py",
        }
    )
    routes = {
        "gfx1151-w4a4-dot-baseline": "s4s4_g32_dot_baseline",
        "gfx1151-w4a4-wmma-baseline-padded": "s4s4_g32_wmma_baseline",
        "gfx1151-w4a4-dot-mtile4": "s4s4_g32_dot_mtile<4>",
        "gfx1151-w4a4-dot-mtile8": "s4s4_g32_dot_mtile<8>",
        "gfx1151-w4a4-wmma-16x64": "s4s4_g32_wmma_tiled<16, 64>",
        "gfx1151-w4a4-wmma-32x64": "s4s4_g32_wmma_tiled<32, 64>",
        "gfx1151-w4a4-wmma-32x32": "s4s4_g32_wmma_tiled<32, 32>",
    }
    for target_id, symbol in routes.items():
        target = select_target(registry, target_id=target_id, framework="vllm")
        assert target.status == "available"
        assert target.allowed_edit_paths == ("csrc/quantization/gfx1151_archive_w4a4/gfx1151_w4a4_perf_op.hip",)
        assert symbol in target.symbols
        contract = candidate_contract(target)
        assert contract["allowed_edit_paths"] == list(target.allowed_edit_paths)
