# SPDX-License-Identifier: MIT
"""CPU-only RED contract for gfx1151 autonomy integration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.candidate_control import (
    CandidateControlError,
    CandidateControlPlane,
    build_agent_request,
    bind_magpie,
    bind_tracelens,
    candidate_artifacts_from_done,
    strict_json_file,
)
from hyperloom.orchestrator.loop.coordinator import _resolvable_artifacts_from_done

RUNNER_HASH = "42884f444743101616eb8cc1f94db7466c8c79a37d60736ae29be7d8e424d4cf"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_source(tmp_path: Path) -> Path:
    path = tmp_path / "candidates" / "candidate.py"
    path.parent.mkdir(parents=True)
    path.write_text("def kernel(x):\n    return x + 1\n")
    return path


def handoff(tmp_path: Path) -> dict:
    source = candidate_source(tmp_path)
    trace = tmp_path / "tracelens.md"; trace.write_text("# Analysis\nno embedded payload\n")
    candidates = tmp_path / "tracelens-candidates.json"; candidates.write_text('{"candidates":[]}\n')
    return {
        "schema": "hyperloom.candidate-handoff.v3",
        "attempt_id": "attempt-0001",
        "parent_attempt_id": None,
        "target": {
            "isa": "gfx1151",
            "board": "strix-halo-radeon-8060s",
            "rocm_root": "/opt/rocm/core-10.0",
            "fallback": "none",
        },
        "operation": {
            "name": "vector_increment",
            "shape": [1024],
            "dtype": "float32",
            "layout": "contiguous",
        },
        "source": {
            "allowed_root": str(source.parent),
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": sha(source),
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
            "performance": "paired_abba",
            "repetitions": 4,
            "promotion_margin": 0.03,
        },
        "tracelens": bind_tracelens(trace, candidates),
        "magpie": bind_magpie({
            "status": "already_patched",
            "script_sha256": "b" * 64,
            "patch_sha256": "c" * 64,
            "promotion_authority": False,
        }),
        "explore_policy": {
            "label": "arbor-pattern",
            "mode": "explore",
            "budget": 1,
            "is_source_module": False,
        },
        "promotion_authority": False,
    }


def test_tracelens_and_magpie_bindings_are_exact_and_non_authoritative(tmp_path):
    trace = tmp_path / "trace.md"; trace.write_text("analysis\n")
    candidates = tmp_path / "candidates.json"; candidates.write_text("{}\n")
    bound = bind_tracelens(trace, candidates)
    assert bound["analysis"] == {"path": str(trace), "bytes": 9, "sha256": sha(trace)}
    assert bound["candidates"]["sha256"] == sha(candidates)
    assert bound["prompt_excerpt"] == "analysis\n"
    with pytest.raises(CandidateControlError):
        bind_tracelens(trace, trace.parent / "missing")
    magpie = bind_magpie({"status":"applied", "script_sha256":"a"*64, "patch_sha256":"b"*64, "promotion_authority":False})
    assert magpie["promotion_authority"] is False
    for bad in ({"status":"unknown", "script_sha256":"a"*64, "patch_sha256":"b"*64, "promotion_authority":False},
                {"status":"applied", "script_sha256":"a"*64, "patch_sha256":"b"*64, "promotion_authority":True}):
        with pytest.raises(CandidateControlError): bind_magpie(bad)


def test_handoff_is_write_once_rooted_and_strict(tmp_path):
    plane = CandidateControlPlane(tmp_path / "state")
    value = handoff(tmp_path)
    attempt = plane.create_attempt(value)
    saved = strict_json_file(attempt / "candidate-handoff.json")
    assert saved == value
    with pytest.raises(CandidateControlError): plane.create_attempt(value)
    for mutation in ("extra", "path", "hash", "fallback", "authority", "arbor"):
        bad = handoff(tmp_path / mutation)
        if mutation == "extra": bad["x"] = 1
        if mutation == "path": bad["source"]["path"] = "/tmp/outside.py"
        if mutation == "hash": bad["source"]["sha256"] = "0" * 64
        if mutation == "fallback": bad["provider"]["fallback"] = "claude"
        if mutation == "authority": bad["promotion_authority"] = True
        if mutation == "arbor": bad["explore_policy"]["is_source_module"] = True
        with pytest.raises(CandidateControlError):
            CandidateControlPlane(tmp_path / f"state-{mutation}").create_attempt(bad)


def test_agent_request_matches_endpoint_runner_v1_and_fallback_none(tmp_path):
    value = handoff(tmp_path)
    request = build_agent_request(
        value,
        messages=[{"role":"system", "content":"author one candidate"}, {"role":"user", "content":"increment"}],
        sandbox_root=value["source"]["allowed_root"],
    )
    assert set(request) == {
        "schema", "request_id", "provider", "protocol", "base_url", "api_key_env",
        "model", "capabilities", "sandbox", "timeout_seconds", "retry", "egress",
        "environment", "messages", "output_schema", "fallback",
    }
    assert request["provider"] == "custom" and request["protocol"] == "custom_command"
    assert request["base_url"] is None and request["api_key_env"] is None
    assert request["fallback"] == "none" and request["retry"] == {"max_attempts": 1}
    assert request["sandbox"] == {"mode":"workspace_write", "writable_roots":[value["source"]["allowed_root"]]}
    assert request["egress"] is False


def test_coordinator_routes_only_hash_bound_candidate_handoffs(tmp_path):
    value = handoff(tmp_path)
    plane = CandidateControlPlane(tmp_path / "state")
    attempt = plane.create_attempt(value)
    path = attempt / "candidate-handoff.json"
    entry = {
        "kind": "gfx1151_candidate_handoff",
        "source": str(path),
        "target": "candidate-handoff.json",
        "bytes": path.stat().st_size,
        "sha256": sha(path),
    }
    done = {"artifacts_written": [entry]}
    assert candidate_artifacts_from_done(done, [attempt]) == [entry]
    assert _resolvable_artifacts_from_done(done, [attempt]) == [entry]
    for mutation in ("hash", "bytes", "escape", "symlink"):
        bad = dict(entry)
        if mutation == "hash": bad["sha256"] = "0" * 64
        if mutation == "bytes": bad["bytes"] += 1
        if mutation == "escape": bad["source"] = "/tmp/outside.json"
        if mutation == "symlink":
            link = attempt / "handoff-link.json"; link.symlink_to(path); bad["source"] = str(link)
        with pytest.raises(CandidateControlError):
            _resolvable_artifacts_from_done({"artifacts_written": [bad]}, [attempt])


def test_control_plane_uses_deterministic_evaluator_and_resumes_without_rerun(tmp_path):
    calls = {"generate":0, "compile":0, "evaluate":0}
    value = handoff(tmp_path)
    plane = CandidateControlPlane(tmp_path / "state")
    plane.create_attempt(value)

    def generate(request):
        calls["generate"] += 1
        return {
            "schema":"endpoint_agnostic_runner_v1.result",
            "request_id":request["request_id"], "provider":request["provider"],
            "protocol":request["protocol"], "model":request["model"], "status":"success",
            "structured_output":{"text":"def kernel(x): return x + 1", "KEEP":True},
            "attempts":1, "timing":{"elapsed_seconds":0.1},
            "capability_receipt":request["capabilities"], "diagnostics":{"stderr_tail":""},
            "fallback_used":False, "promotion_authority":False,
        }

    def compile_candidate(source_text, compiler):
        calls["compile"] += 1
        return {"status":"PASS", "binary_sha256":"d"*64, "command":compiler["command"]}

    def evaluate_candidate(binary_sha256, plan):
        calls["evaluate"] += 1
        return {
            "correctness":{"status":"PASS", "mismatches":0, "max_error":0.0},
            "performance":{"status":"PASS", "candidate_ns":90.0, "incumbent_ns":100.0},
            "abba":{"status":"PASS", "order":["A","B","B","A"], "minimum_ratio":1.05},
        }

    result = plane.run("attempt-0001", generator=generate, compiler=compile_candidate, evaluator=evaluate_candidate)
    assert result["status"] == "PASS"
    assert result["decision"] == "candidate_accept"
    assert result["promotion_authority"] is False
    assert result["correctness"]["mismatches"] == 0
    assert "KEEP" not in result
    assert calls == {"generate":1, "compile":1, "evaluate":1}
    again = plane.run("attempt-0001", generator=generate, compiler=compile_candidate, evaluator=evaluate_candidate)
    assert again == result
    assert calls == {"generate":1, "compile":1, "evaluate":1}


def test_evaluator_failure_rejects_candidate_and_agent_cannot_override(tmp_path):
    value = handoff(tmp_path)
    plane = CandidateControlPlane(tmp_path / "state"); plane.create_attempt(value)
    def generate(request):
        return {"schema":"endpoint_agnostic_runner_v1.result", "request_id":request["request_id"],
                "provider":request["provider"], "protocol":request["protocol"], "model":request["model"],
                "status":"success", "structured_output":{"text":"bad", "correct":True, "promotion":True},
                "attempts":1, "timing":{}, "capability_receipt":request["capabilities"],
                "diagnostics":{"stderr_tail":""}, "fallback_used":False, "promotion_authority":False}
    result = plane.run(
        "attempt-0001", generator=generate,
        compiler=lambda *_: {"status":"PASS", "binary_sha256":"d"*64, "command":[]},
        evaluator=lambda *_: {"correctness":{"status":"FAIL", "mismatches":1, "max_error":1.0},
                              "performance":{"status":"SKIP"}, "abba":{"status":"SKIP"}},
    )
    assert result["status"] == "REJECT"
    assert result["decision"] == "candidate_reject"
    assert result["promotion_authority"] is False
