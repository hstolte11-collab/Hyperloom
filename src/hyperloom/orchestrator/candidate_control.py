# SPDX-License-Identifier: MIT
"""Fail-closed gfx1151 candidate handoff and evidence control plane."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence


class CandidateControlError(RuntimeError):
    """A candidate artifact or transition violates the frozen contract."""


_HANDOFF_V3_KEYS = {
    "schema",
    "attempt_id",
    "parent_attempt_id",
    "target",
    "operation",
    "source",
    "provider",
    "compiler",
    "evaluation_plan",
    "tracelens",
    "magpie",
    "explore_policy",
    "promotion_authority",
}
_HANDOFF_V4_KEYS = _HANDOFF_V3_KEYS | {"kernel_target"}
_RESULT_KEYS = {
    "schema",
    "status",
    "attempt_id",
    "handoff",
    "agent",
    "generated_source",
    "compile",
    "correctness",
    "performance",
    "abba",
    "decision",
    "promotion_authority",
}
_RUNNER_RESULT_KEYS = {
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
_SHA = re.compile(r"[0-9a-f]{64}")
_ID = re.compile(r"[A-Za-z0-9_.-]+")


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise CandidateControlError("duplicate JSON key")
        result[key] = value
    return result


def _strict_loads(text: str):
    try:
        return json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                CandidateControlError("non-finite JSON number")
            ),
        )
    except CandidateControlError:
        raise
    except (TypeError, ValueError) as exc:
        raise CandidateControlError("invalid JSON") from exc


def _regular_bytes(path: Path) -> bytes:
    path = path.absolute()
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise CandidateControlError(f"missing artifact: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise CandidateControlError(f"artifact must be a regular non-symlink: {path}")
    if path.resolve(strict=True) != path:
        raise CandidateControlError(f"artifact traverses a symlink: {path}")
    return path.read_bytes()


def strict_json_file(path: str | os.PathLike[str]):
    raw = _regular_bytes(Path(path))
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise CandidateControlError("JSON artifact is not UTF-8") from exc
    return _strict_loads(text)


def _meta(path: Path) -> dict[str, Any]:
    data = _regular_bytes(path)
    return {
        "path": str(path.absolute()),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path = path.absolute()
    if path.exists() or path.is_symlink():
        raise CandidateControlError(f"refusing to overwrite {path}")
    payload = _canonical(value)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    fd = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise CandidateControlError("short atomic write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise CandidateControlError(f"invalid {name} SHA-256")
    return value


def _identity(value: Any, name: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise CandidateControlError(f"invalid {name}")
    return value


def _finite_tree(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise CandidateControlError("non-finite evidence")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CandidateControlError("non-string evidence key")
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)


def bind_tracelens(analysis_path: Path, candidates_path: Path) -> dict[str, Any]:
    analysis = _meta(Path(analysis_path))
    candidates = _meta(Path(candidates_path))
    text = Path(analysis_path).read_text(encoding="utf-8")
    lines = []
    for line in text.splitlines(keepends=True):
        lowered = line.lower()
        if "base64" in lowered or "data:" in lowered and ";base64," in lowered:
            continue
        lines.append(line)
    excerpt = "".join(lines)[:8192]
    return {
        "analysis": analysis,
        "candidates": candidates,
        "prompt_excerpt": excerpt,
        "promotion_authority": False,
    }


def bind_magpie(receipt: Mapping[str, Any]) -> dict[str, Any]:
    keys = {"status", "script_sha256", "patch_sha256", "promotion_authority"}
    if not isinstance(receipt, Mapping) or set(receipt) != keys:
        raise CandidateControlError("Magpie receipt key closure failed")
    if receipt["status"] not in {"upstream_atomic", "already_patched", "applied"}:
        raise CandidateControlError("unsupported Magpie status")
    _digest(receipt["script_sha256"], "Magpie script")
    _digest(receipt["patch_sha256"], "Magpie patch")
    if receipt["promotion_authority"] is not False:
        raise CandidateControlError("Magpie cannot claim promotion authority")
    return dict(receipt)


def _validate_tracelens(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {
        "analysis", "candidates", "prompt_excerpt", "promotion_authority"
    }:
        raise CandidateControlError("TraceLens binding key closure failed")
    if value["promotion_authority"] is not False or not isinstance(value["prompt_excerpt"], str):
        raise CandidateControlError("TraceLens authority/excerpt drift")
    for key in ("analysis", "candidates"):
        if value[key] != _meta(Path(value[key]["path"])):
            raise CandidateControlError("TraceLens artifact binding drift")


def _validate_kernel_target_binding(value: Any) -> dict[str, Any]:
    keys = {
        "registry_id",
        "registry_path",
        "registry_bytes",
        "registry_sha256",
        "target_id",
        "framework",
        "source_lineage",
        "candidate_contract",
        "candidate_contract_sha256",
        "fallback",
        "promotion_authority",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise CandidateControlError("kernel target binding key closure failed")
    for key in ("registry_id", "target_id", "framework", "source_lineage"):
        _identity(value[key], f"kernel target {key}")
    registry_path = Path(value["registry_path"]).absolute()
    if _meta(registry_path) != {
        "path": str(registry_path),
        "bytes": value["registry_bytes"],
        "sha256": value["registry_sha256"],
    }:
        raise CandidateControlError("kernel target registry binding drift")
    contract = value["candidate_contract"]
    contract_keys = {
        "schema",
        "target_id",
        "framework",
        "family",
        "role",
        "source_lineage",
        "source_files",
        "allowed_edit_paths",
        "symbols",
        "domain",
        "materialization",
        "correctness",
        "fallback",
        "promotion_authority",
    }
    if not isinstance(contract, dict) or set(contract) != contract_keys:
        raise CandidateControlError("kernel candidate contract key closure failed")
    if contract["schema"] != "hyperloom.kernel-candidate-contract.v1":
        raise CandidateControlError("kernel candidate contract schema drift")
    if (
        contract["target_id"] != value["target_id"]
        or contract["framework"] != value["framework"]
        or contract["source_lineage"] != value["source_lineage"]
        or contract["fallback"] != "none"
        or contract["promotion_authority"] is not False
    ):
        raise CandidateControlError("kernel candidate contract identity drift")
    if (
        not isinstance(contract["allowed_edit_paths"], list)
        or not contract["allowed_edit_paths"]
        or not isinstance(contract["source_files"], list)
        or not set(contract["allowed_edit_paths"]).issubset(contract["source_files"])
        or not isinstance(contract["symbols"], list)
        or not contract["symbols"]
    ):
        raise CandidateControlError("kernel candidate edit/symbol contract drift")
    if hashlib.sha256(_canonical(contract)).hexdigest() != value["candidate_contract_sha256"]:
        raise CandidateControlError("kernel candidate contract hash drift")
    if value["fallback"] != "none" or value["promotion_authority"] is not False:
        raise CandidateControlError("kernel target fallback or authority drift")
    _finite_tree(value)
    return value


def _validate_handoff(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CandidateControlError("candidate handoff key closure failed")
    schema = value.get("schema")
    expected_keys = (
        _HANDOFF_V4_KEYS
        if schema == "hyperloom.candidate-handoff.v4"
        else _HANDOFF_V3_KEYS
    )
    if set(value) != expected_keys:
        raise CandidateControlError("candidate handoff key closure failed")
    if schema not in {
        "hyperloom.candidate-handoff.v3",
        "hyperloom.candidate-handoff.v4",
    }:
        raise CandidateControlError("candidate handoff schema drift")
    attempt_id = _identity(value["attempt_id"], "attempt_id")
    parent = value["parent_attempt_id"]
    if parent is not None:
        _identity(parent, "parent_attempt_id")
    target = value["target"]
    if target != {
        "isa": "gfx1151",
        "board": "strix-halo-radeon-8060s",
        "rocm_root": "/opt/rocm/core-10.0",
        "fallback": "none",
    }:
        raise CandidateControlError("target profile drift")
    operation = value["operation"]
    if (
        not isinstance(operation, dict)
        or set(operation) != {"name", "shape", "dtype", "layout"}
        or not isinstance(operation["shape"], list)
        or not operation["shape"]
    ):
        raise CandidateControlError("operation contract drift")
    source = value["source"]
    if not isinstance(source, dict) or set(source) != {
        "allowed_root", "path", "bytes", "sha256", "repository_commit"
    }:
        raise CandidateControlError("source binding key closure failed")
    allowed_root = Path(source["allowed_root"]).absolute()
    source_path = Path(source["path"]).absolute()
    try:
        source_path.relative_to(allowed_root)
    except ValueError as exc:
        raise CandidateControlError("candidate source escapes allowed root") from exc
    if _meta(source_path) != {
        "path": str(source_path),
        "bytes": source["bytes"],
        "sha256": source["sha256"],
    }:
        raise CandidateControlError("candidate source binding drift")
    if not isinstance(source["repository_commit"], str) or len(source["repository_commit"]) != 40:
        raise CandidateControlError("repository commit binding drift")
    provider = value["provider"]
    if not isinstance(provider, dict) or set(provider) != {
        "runner_contract_sha256", "provider", "protocol", "model", "fallback"
    }:
        raise CandidateControlError("provider binding key closure failed")
    _digest(provider["runner_contract_sha256"], "runner contract")
    for key in ("provider", "protocol", "model"):
        _identity(provider[key], f"provider {key}")
    if provider["fallback"] != "none":
        raise CandidateControlError("provider fallback must be none")
    compiler = value["compiler"]
    if (
        not isinstance(compiler, dict)
        or set(compiler) != {"rocm_root", "offload_arch", "command", "hsa_override_gfx_version"}
        or compiler["rocm_root"] != target["rocm_root"]
        or compiler["offload_arch"] != "gfx1151"
        or compiler["hsa_override_gfx_version"] != "forbidden"
        or not isinstance(compiler["command"], list)
        or not compiler["command"]
        or any(not isinstance(item, str) or not item for item in compiler["command"])
    ):
        raise CandidateControlError("compiler contract drift")
    plan = value["evaluation_plan"]
    if not isinstance(plan, dict) or set(plan) != {
        "correctness", "performance", "repetitions", "promotion_margin"
    }:
        raise CandidateControlError("evaluation plan drift")
    if schema == "hyperloom.candidate-handoff.v3":
        plan_valid = (
            plan["correctness"] == "deterministic_reference"
            and plan["performance"] == "paired_abba"
            and plan["repetitions"] >= 4
            and plan["promotion_margin"] > 0
        )
    else:
        plan_valid = (
            plan["correctness"] == "deterministic_reference"
            and plan["performance"] in {"optional", "paired_abba"}
            and isinstance(plan["repetitions"], int)
            and not isinstance(plan["repetitions"], bool)
            and plan["repetitions"] >= 1
            and isinstance(plan["promotion_margin"], (int, float))
            and not isinstance(plan["promotion_margin"], bool)
            and math.isfinite(float(plan["promotion_margin"]))
            and plan["promotion_margin"] >= 0
        )
    if not plan_valid:
        raise CandidateControlError("evaluation plan drift")
    _validate_tracelens(value["tracelens"])
    bind_magpie(value["magpie"])
    explore = value["explore_policy"]
    if explore != {
        "label": "arbor-pattern",
        "mode": "explore",
        "budget": 1,
        "is_source_module": False,
    }:
        raise CandidateControlError("EXPLORE/Arbor policy drift")
    if value["promotion_authority"] is not False:
        raise CandidateControlError("handoff claims promotion authority")
    if schema == "hyperloom.candidate-handoff.v4":
        _validate_kernel_target_binding(value["kernel_target"])
    _finite_tree(value)
    return value


def build_agent_request(
    handoff: Mapping[str, Any],
    *,
    messages: Sequence[Mapping[str, Any]],
    sandbox_root: str,
) -> dict[str, Any]:
    value = _validate_handoff(dict(handoff))
    provider = value["provider"]
    root = Path(sandbox_root).absolute()
    if root != Path(value["source"]["allowed_root"]).absolute():
        raise CandidateControlError("agent sandbox root differs from source root")
    source_only_provider = (provider["provider"], provider["protocol"]) in {
        ("codex", "codex_app_server"),
        ("hermes", "hermes_oneshot"),
    }
    request_messages = [dict(item) for item in messages]
    if source_only_provider:
        kernel_target = value.get("kernel_target")
        if not isinstance(kernel_target, dict):
            raise CandidateControlError("source-only provider requires kernel target")
        target_context = {
            "target_id": kernel_target["target_id"],
            "framework": kernel_target["framework"],
            "candidate_contract_sha256": kernel_target["candidate_contract_sha256"],
            "candidate_contract": kernel_target["candidate_contract"],
            "fallback": "none",
            "promotion_authority": False,
        }
        request_messages.insert(
            0,
            {
                "role": "system",
                "content": (
                    "Author source only for this exact kernel target contract. "
                    "Use no tools and make no correctness or promotion claims.\n"
                    + _canonical(target_context).decode("utf-8")
                ),
            },
        )
    request = {
        "schema": "endpoint_agnostic_runner_v1.request",
        "request_id": value["attempt_id"],
        "provider": provider["provider"],
        "protocol": provider["protocol"],
        "base_url": None,
        "api_key_env": None,
        "model": provider["model"],
        "capabilities": (
            ["coder", "structured_output"]
            if source_only_provider
            else ["coder", "tools", "structured_output", "session_resume"]
        ),
        "sandbox": (
            {"mode": "read_only", "writable_roots": []}
            if source_only_provider
            else {"mode": "workspace_write", "writable_roots": [str(root)]}
        ),
        "timeout_seconds": 1800,
        "retry": {"max_attempts": 1},
        "egress": source_only_provider,
        "environment": (
            {}
            if source_only_provider
            else {
                "ROCM_ROOT": value["target"]["rocm_root"],
                "GPU_TARGET": value["target"]["isa"],
            }
        ),
        "messages": request_messages,
        "output_schema": {
            "type": "object",
            "required": ["text"],
            "properties": {"text": {"type": "string"}},
        },
        "fallback": "none",
    }
    if value["schema"] == "hyperloom.candidate-handoff.v4" and not source_only_provider:
        kernel_target = value["kernel_target"]
        request["environment"].update(
            {
                "KERNEL_TARGET_ID": kernel_target["target_id"],
                "KERNEL_FRAMEWORK": kernel_target["framework"],
                "KERNEL_TARGET_CONTRACT_SHA256": kernel_target[
                    "candidate_contract_sha256"
                ],
            }
        )
    return request


def candidate_artifacts_from_done(
    done_payload: Mapping[str, Any] | None,
    resolve_bases: Sequence[Path],
) -> list[dict[str, Any]]:
    """Resolve candidate handoffs strictly; malformed entries are blockers."""

    if not isinstance(done_payload, Mapping):
        return []
    artifacts = done_payload.get("artifacts_written")
    if not isinstance(artifacts, list):
        return []
    candidate_entries = [
        entry
        for entry in artifacts
        if isinstance(entry, dict) and entry.get("kind") == "gfx1151_candidate_handoff"
    ]
    if not candidate_entries:
        return []
    bases = [Path(base).absolute().resolve(strict=True) for base in resolve_bases]
    accepted: list[dict[str, Any]] = []
    for entry in candidate_entries:
        if set(entry) != {"kind", "source", "target", "bytes", "sha256"}:
            raise CandidateControlError("candidate artifact key closure failed")
        target = entry["target"]
        if (
            not isinstance(target, str)
            or not target
            or Path(target).is_absolute()
            or ".." in Path(target).parts
        ):
            raise CandidateControlError("candidate artifact target is invalid")
        raw = Path(entry["source"])
        candidates = [raw] if raw.is_absolute() else [base / raw for base in bases]
        resolved_source = None
        for candidate in candidates:
            absolute = candidate.absolute()
            try:
                data = _regular_bytes(absolute)
            except CandidateControlError:
                continue
            resolved = absolute.resolve(strict=True)
            if any(resolved.is_relative_to(base) for base in bases):
                resolved_source = absolute
                break
        if resolved_source is None:
            raise CandidateControlError("candidate artifact source is missing, symlinked, or outside roots")
        meta = _meta(resolved_source)
        if entry["bytes"] != meta["bytes"] or entry["sha256"] != meta["sha256"]:
            raise CandidateControlError("candidate artifact size/hash drift")
        accepted.append(entry)
    return accepted


class CandidateControlPlane:
    """Write-once candidate lifecycle with deterministic evaluator authority."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root).absolute()
        if self.root.is_symlink():
            raise CandidateControlError("candidate root cannot be a symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.resolve(strict=True) != self.root:
            raise CandidateControlError("candidate root traverses a symlink")

    def create_attempt(self, handoff: Mapping[str, Any]) -> Path:
        value = _validate_handoff(dict(handoff))
        attempt = self.root / value["attempt_id"]
        try:
            attempt.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise CandidateControlError("candidate attempt already exists") from exc
        _write_new(attempt / "candidate-handoff.json", value)
        return attempt

    def _validate_runner_result(self, result: Any, request: Mapping[str, Any]) -> str:
        if not isinstance(result, dict) or set(result) != _RUNNER_RESULT_KEYS:
            raise CandidateControlError("runner result key closure failed")
        if result["schema"] != "endpoint_agnostic_runner_v1.result":
            raise CandidateControlError("runner result schema drift")
        for key in ("request_id", "provider", "protocol", "model"):
            if result[key] != request[key]:
                raise CandidateControlError("runner result identity drift")
        if (
            result["status"] != "success"
            or result["attempts"] != 1
            or result["fallback_used"] is not False
            or result["promotion_authority"] is not False
            or result["capability_receipt"] != request["capabilities"]
        ):
            raise CandidateControlError("runner status/authority drift")
        if not isinstance(result["diagnostics"], dict) or result["diagnostics"].get("stderr_tail"):
            raise CandidateControlError("runner diagnostics are not clean")
        output = result["structured_output"]
        if not isinstance(output, dict) or not isinstance(output.get("text"), str):
            raise CandidateControlError("runner produced no candidate text")
        _finite_tree(result)
        return output["text"]

    def run(self, attempt_id: str, *, generator, compiler, evaluator) -> dict[str, Any]:
        attempt = self.root / _identity(attempt_id, "attempt_id")
        if not attempt.is_dir() or attempt.is_symlink():
            raise CandidateControlError("attempt directory is missing or invalid")
        result_path = attempt / "candidate-result.json"
        if result_path.exists():
            result = strict_json_file(result_path)
            if not isinstance(result, dict) or set(result) != _RESULT_KEYS:
                raise CandidateControlError("persisted candidate result drift")
            return result
        handoff_path = attempt / "candidate-handoff.json"
        handoff = _validate_handoff(strict_json_file(handoff_path))
        request = build_agent_request(
            handoff,
            messages=[
                {"role": "system", "content": "Author one bounded candidate only."},
                {"role": "user", "content": handoff["tracelens"]["prompt_excerpt"]},
            ],
            sandbox_root=handoff["source"]["allowed_root"],
        )
        runner_result = generator(request)
        candidate_text = self._validate_runner_result(runner_result, request)
        compile_result = compiler(candidate_text, handoff["compiler"])
        if not isinstance(compile_result, dict) or compile_result.get("status") != "PASS":
            raise CandidateControlError("candidate compilation did not pass")
        _digest(compile_result.get("binary_sha256"), "candidate binary")
        evaluation = evaluator(compile_result["binary_sha256"], handoff["evaluation_plan"])
        if not isinstance(evaluation, dict) or set(evaluation) != {"correctness", "performance", "abba"}:
            raise CandidateControlError("evaluator result key closure failed")
        correctness = evaluation["correctness"]
        accepted = (
            correctness.get("status") == "PASS"
            and correctness.get("mismatches") == 0
            and evaluation["performance"].get("status") == "PASS"
            and evaluation["abba"].get("status") == "PASS"
            and evaluation["abba"].get("minimum_ratio", 0)
            >= 1 + handoff["evaluation_plan"]["promotion_margin"]
        )
        result = {
            "schema": "hyperloom.candidate-result.v3",
            "status": "PASS" if accepted else "REJECT",
            "attempt_id": attempt_id,
            "handoff": _meta(handoff_path),
            "agent": {
                "provider": request["provider"],
                "protocol": request["protocol"],
                "model": request["model"],
                "fallback_used": False,
            },
            "generated_source": {
                "bytes": len(candidate_text.encode("utf-8")),
                "sha256": hashlib.sha256(candidate_text.encode("utf-8")).hexdigest(),
            },
            "compile": compile_result,
            "correctness": correctness,
            "performance": evaluation["performance"],
            "abba": evaluation["abba"],
            "decision": "candidate_accept" if accepted else "candidate_reject",
            "promotion_authority": False,
        }
        if set(result) != _RESULT_KEYS:
            raise CandidateControlError("internal result key closure failed")
        _finite_tree(result)
        _write_new(result_path, result)
        return result
