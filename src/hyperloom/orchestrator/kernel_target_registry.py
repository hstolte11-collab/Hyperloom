# SPDX-License-Identifier: MIT
"""Source-bound framework-aware kernel target registry."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class KernelTargetRegistryError(RuntimeError):
    """The kernel target registry or a selection violates its contract."""


_ID = re.compile(r"[A-Za-z0-9_.-]+")
_SHA = re.compile(r"[0-9a-f]{64}")
_SUPPORT_STATES = {
    "native",
    "adapted",
    "shared",
    "control_only",
    "pending_task_10_18",
    "unavailable_vllm_private",
    "unavailable_sglang_private",
}
_USABLE_SUPPORT = {"native", "adapted", "shared", "control_only"}


@dataclass(frozen=True)
class KernelTarget:
    id: str
    framework: str
    family: str
    role: str
    status: str
    source_lineage: str
    source_files: frozenset[str]
    allowed_edit_paths: tuple[str, ...]
    symbols: tuple[str, ...]
    domain: dict[str, Any]
    materialization: dict[str, Any]
    correctness: dict[str, Any]
    framework_support: dict[str, str]
    fallback: str
    promotion_authority: bool


@dataclass(frozen=True)
class KernelTargetRegistry:
    schema: str
    registry_id: str
    platform: dict[str, str]
    source_lineages: dict[str, dict[str, str]]
    files: dict[str, dict[str, Any]]
    targets: dict[str, KernelTarget]
    closed_donors: tuple[dict[str, Any], ...]
    fallback: str
    promotion_authority: bool
    manifest_path: str
    manifest_bytes: int
    manifest_sha256: str


def _pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise KernelTargetRegistryError("duplicate JSON key")
        out[key] = value
    return out


def _strict_json(path: Path) -> tuple[dict[str, Any], bytes]:
    path = path.absolute()
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise KernelTargetRegistryError("registry manifest is missing") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise KernelTargetRegistryError("registry manifest must be a regular file")
    data = path.read_bytes()
    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                KernelTargetRegistryError(f"non-finite JSON token: {token}")
            ),
        )
    except KernelTargetRegistryError:
        raise
    except (UnicodeDecodeError, ValueError) as exc:
        raise KernelTargetRegistryError("registry manifest is invalid JSON") from exc
    if not isinstance(value, dict):
        raise KernelTargetRegistryError("registry manifest must be an object")
    return value, data


def _finite_tree(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise KernelTargetRegistryError("non-finite registry value")
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise KernelTargetRegistryError("registry keys must be strings")
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)


def _identity(value: Any, name: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise KernelTargetRegistryError(f"invalid {name}")
    return value


def _regular_source(root: Path, relative: str) -> bytes:
    rel = Path(relative)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts:
        raise KernelTargetRegistryError("invalid source-relative path")
    root = root.absolute()
    if root.resolve(strict=True) != root:
        raise KernelTargetRegistryError("source root traverses a symlink")
    path = root / rel
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise KernelTargetRegistryError(f"missing source file: {relative}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise KernelTargetRegistryError(f"source file is not regular: {relative}")
    if path.resolve(strict=True) != path:
        raise KernelTargetRegistryError(f"source file traverses a symlink: {relative}")
    return path.read_bytes()


def _git_identity(root: Path) -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise KernelTargetRegistryError("cannot resolve source Git identity") from exc
        return completed.stdout.rstrip("\n")

    return {
        "repository_commit": run("rev-parse", "HEAD"),
        "repository_tree": run("rev-parse", "HEAD^{tree}"),
    }


def _target(value: Any, files: Mapping[str, Any]) -> KernelTarget:
    keys = {
        "id",
        "framework",
        "family",
        "role",
        "status",
        "source_lineage",
        "source_files",
        "allowed_edit_paths",
        "symbols",
        "domain",
        "materialization",
        "correctness",
        "framework_support",
        "fallback",
        "promotion_authority",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise KernelTargetRegistryError("target key closure failed")
    target_id = _identity(value["id"], "target id")
    framework = _identity(value["framework"], "framework")
    family = _identity(value["family"], "family")
    role = _identity(value["role"], "role")
    if value["status"] not in {"available", "control_only"}:
        raise KernelTargetRegistryError("invalid target status")
    source_files = value["source_files"]
    edit_paths = value["allowed_edit_paths"]
    symbols = value["symbols"]
    if (
        not isinstance(source_files, list)
        or not source_files
        or any(path not in files for path in source_files)
        or len(set(source_files)) != len(source_files)
    ):
        raise KernelTargetRegistryError("target source-file closure failed")
    if (
        not isinstance(edit_paths, list)
        or any(path not in source_files for path in edit_paths)
        or len(set(edit_paths)) != len(edit_paths)
    ):
        raise KernelTargetRegistryError("target edit-path closure failed")
    if value["status"] == "available" and not edit_paths:
        raise KernelTargetRegistryError("available target requires edit paths")
    if value["status"] == "control_only" and edit_paths:
        raise KernelTargetRegistryError("control-only target cannot expose edit paths")
    if (
        not isinstance(symbols, list)
        or not symbols
        or any(not isinstance(symbol, str) or not symbol for symbol in symbols)
        or len(set(symbols)) != len(symbols)
    ):
        raise KernelTargetRegistryError("target symbol closure failed")
    materialization = value["materialization"]
    if (
        not isinstance(materialization, dict)
        or set(materialization) != {"kind", "entrypoints", "commands"}
        or not isinstance(materialization["kind"], str)
        or not materialization["kind"]
        or not isinstance(materialization["entrypoints"], list)
        or not materialization["entrypoints"]
        or not isinstance(materialization["commands"], list)
    ):
        raise KernelTargetRegistryError("materialization contract drift")
    correctness = value["correctness"]
    if (
        not isinstance(correctness, dict)
        or set(correctness) != {"oracle", "files", "requires_gpu", "evidence_state"}
        or not isinstance(correctness["oracle"], str)
        or not correctness["oracle"]
        or not isinstance(correctness["files"], list)
        or not correctness["files"]
        or any(path not in files for path in correctness["files"])
        or not isinstance(correctness["requires_gpu"], bool)
        or correctness["evidence_state"]
        not in {"qualified", "source_available_pending_functional_proof", "control_only"}
    ):
        raise KernelTargetRegistryError("correctness contract drift")
    support = value["framework_support"]
    if (
        not isinstance(support, dict)
        or set(support) != {"vllm", "sglang"}
        or any(state not in _SUPPORT_STATES for state in support.values())
    ):
        raise KernelTargetRegistryError("framework support matrix drift")
    if support[framework] not in _USABLE_SUPPORT:
        raise KernelTargetRegistryError("native framework is not usable")
    if value["fallback"] != "none" or value["promotion_authority"] is not False:
        raise KernelTargetRegistryError("target fallback or authority drift")
    if not isinstance(value["domain"], dict) or not value["domain"]:
        raise KernelTargetRegistryError("target domain is missing")
    _finite_tree(value)
    return KernelTarget(
        id=target_id,
        framework=framework,
        family=family,
        role=role,
        status=value["status"],
        source_lineage=_identity(value["source_lineage"], "source lineage"),
        source_files=frozenset(source_files),
        allowed_edit_paths=tuple(edit_paths),
        symbols=tuple(symbols),
        domain=dict(value["domain"]),
        materialization=dict(materialization),
        correctness=dict(correctness),
        framework_support=dict(support),
        fallback="none",
        promotion_authority=False,
    )


def load_registry(
    manifest_path: str | os.PathLike[str],
    *,
    source_roots: Mapping[str, str | os.PathLike[str]],
) -> KernelTargetRegistry:
    manifest, manifest_bytes = _strict_json(Path(manifest_path))
    keys = {
        "schema",
        "registry_id",
        "platform",
        "source_lineages",
        "files",
        "targets",
        "closed_donors",
        "fallback",
        "promotion_authority",
    }
    if set(manifest) != keys:
        raise KernelTargetRegistryError("registry key closure failed")
    if manifest["schema"] != "hyperloom.kernel-target-registry.v1":
        raise KernelTargetRegistryError("registry schema drift")
    registry_id = _identity(manifest["registry_id"], "registry id")
    platform = manifest["platform"]
    if platform != {
        "isa": "gfx1151",
        "board": "strix-halo-radeon-8060s",
        "rocm_root": "/opt/rocm/core-10.0",
    }:
        raise KernelTargetRegistryError("registry platform drift")
    lineages = manifest["source_lineages"]
    roots = {name: Path(path).absolute() for name, path in source_roots.items()}
    if not isinstance(lineages, dict) or set(lineages) != set(roots):
        raise KernelTargetRegistryError("source lineage/root closure failed")
    for name, expected in lineages.items():
        if (
            not isinstance(expected, dict)
            or set(expected) != {"repository_commit", "repository_tree"}
            or _git_identity(roots[name]) != expected
        ):
            raise KernelTargetRegistryError(f"source lineage drift: {name}")
    files = manifest["files"]
    if not isinstance(files, dict) or not files:
        raise KernelTargetRegistryError("source file roster missing")
    for relative, metadata in files.items():
        if (
            not isinstance(metadata, dict)
            or set(metadata) != {"source_lineage", "bytes", "sha256"}
            or metadata["source_lineage"] not in roots
            or isinstance(metadata["bytes"], bool)
            or not isinstance(metadata["bytes"], int)
            or metadata["bytes"] < 0
            or not isinstance(metadata["sha256"], str)
            or _SHA.fullmatch(metadata["sha256"]) is None
        ):
            raise KernelTargetRegistryError("source file metadata drift")
        data = _regular_source(roots[metadata["source_lineage"]], relative)
        if len(data) != metadata["bytes"] or hashlib.sha256(data).hexdigest() != metadata["sha256"]:
            raise KernelTargetRegistryError(f"source file drift: {relative}")
    values = manifest["targets"]
    if not isinstance(values, list) or not values:
        raise KernelTargetRegistryError("target roster missing")
    targets = {}
    for value in values:
        target = _target(value, files)
        if target.id in targets:
            raise KernelTargetRegistryError("duplicate target id")
        if target.source_lineage not in lineages:
            raise KernelTargetRegistryError("target source lineage is unknown")
        targets[target.id] = target
    closed = manifest["closed_donors"]
    if not isinstance(closed, list):
        raise KernelTargetRegistryError("closed donor roster drift")
    seen_closed = set()
    for row in closed:
        if (
            not isinstance(row, dict)
            or set(row) != {"id", "reason", "source_files"}
            or _identity(row["id"], "closed donor id") in seen_closed
            or not isinstance(row["reason"], str)
            or not row["reason"]
            or not isinstance(row["source_files"], list)
            or any(path not in files for path in row["source_files"])
        ):
            raise KernelTargetRegistryError("closed donor contract drift")
        seen_closed.add(row["id"])
    if manifest["fallback"] != "none" or manifest["promotion_authority"] is not False:
        raise KernelTargetRegistryError("registry fallback or authority drift")
    _finite_tree(manifest)
    return KernelTargetRegistry(
        schema=manifest["schema"],
        registry_id=registry_id,
        platform=dict(platform),
        source_lineages={name: dict(value) for name, value in lineages.items()},
        files={name: dict(value) for name, value in files.items()},
        targets=targets,
        closed_donors=tuple(dict(row) for row in closed),
        fallback="none",
        promotion_authority=False,
        manifest_path=str(Path(manifest_path).absolute()),
        manifest_bytes=len(manifest_bytes),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )


def select_target(
    registry: KernelTargetRegistry,
    *,
    target_id: str,
    framework: str,
) -> KernelTarget:
    target_id = _identity(target_id, "target id")
    framework = _identity(framework, "framework")
    target = registry.targets.get(target_id)
    if target is None:
        raise KernelTargetRegistryError("target id is not registered")
    support = target.framework_support.get(framework)
    if support not in _USABLE_SUPPORT:
        raise KernelTargetRegistryError(
            f"target {target_id!r} is not usable for framework {framework!r}: {support}"
        )
    return target


def candidate_contract(target: KernelTarget) -> dict[str, Any]:
    if target.status == "control_only":
        raise KernelTargetRegistryError("control-only target is not candidate-editable")
    return {
        "schema": "hyperloom.kernel-candidate-contract.v1",
        "target_id": target.id,
        "framework": target.framework,
        "family": target.family,
        "role": target.role,
        "source_lineage": target.source_lineage,
        "source_files": sorted(target.source_files),
        "allowed_edit_paths": list(target.allowed_edit_paths),
        "symbols": list(target.symbols),
        "domain": dict(target.domain),
        "materialization": dict(target.materialization),
        "correctness": dict(target.correctness),
        "fallback": "none",
        "promotion_authority": False,
    }


def bind_kernel_target(
    registry: KernelTargetRegistry,
    *,
    target_id: str,
    framework: str,
) -> dict[str, Any]:
    target = select_target(registry, target_id=target_id, framework=framework)
    contract = candidate_contract(target)
    contract_bytes = (
        json.dumps(contract, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()
    return {
        "registry_id": registry.registry_id,
        "registry_path": registry.manifest_path,
        "registry_bytes": registry.manifest_bytes,
        "registry_sha256": registry.manifest_sha256,
        "target_id": target.id,
        "framework": framework,
        "source_lineage": target.source_lineage,
        "candidate_contract": contract,
        "candidate_contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "fallback": "none",
        "promotion_authority": False,
    }
