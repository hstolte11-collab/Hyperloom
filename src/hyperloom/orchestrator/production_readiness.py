# SPDX-License-Identifier: MIT
"""Durable, fail-closed production-readiness control plane for gfx1151."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping

from hyperloom.orchestrator.candidate_control import (
    CandidateControlError,
    CandidateControlPlane,
    strict_json_file,
)
from hyperloom.orchestrator.kernel_target_registry import (
    KernelTargetRegistryError,
    load_registry,
)


class ProductionReadinessError(RuntimeError):
    """Readiness state, identity, or transition failed closed."""


def _canonical(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ProductionReadinessError("value is not strict JSON") from exc


def _meta(path: Path) -> dict[str, Any]:
    path = path.absolute()
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ProductionReadinessError(f"missing file: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ProductionReadinessError(f"file must be regular and non-symlink: {path}")
    if path.resolve(strict=True) != path:
        raise ProductionReadinessError(f"file traverses symlink: {path}")
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path = path.absolute()
    if path.exists() or path.is_symlink():
        raise ProductionReadinessError(f"refusing to overwrite {path}")
    payload = _canonical(dict(value))
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
                raise ProductionReadinessError("short readiness write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)


def _providers(path: Path) -> tuple[dict[str, Any], ...]:
    value = strict_json_file(path)
    keys = {
        "schema",
        "status",
        "providers",
        "claude_wiring",
        "provider_fallback",
        "model_fallback",
        "promotion_authority",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ProductionReadinessError("provider roster key closure failed")
    if (
        value["schema"] != "hyperloom.gfx1151-production-provider-roster.v1"
        or value["status"] != "QUALIFIED"
        or value["provider_fallback"] != "none"
        or value["model_fallback"] != "none"
        or value["promotion_authority"] is not False
    ):
        raise ProductionReadinessError("provider roster identity drift")
    rows = value["providers"]
    if not isinstance(rows, list) or [row.get("id") for row in rows] != [
        "codex-native-oauth",
        "hermes-openai-codex",
    ]:
        raise ProductionReadinessError("provider roster cardinality/order drift")
    expected = {
        "codex-native-oauth": ("codex", "codex_app_server", "gpt-5.6-sol"),
        "hermes-openai-codex": ("hermes", "hermes_oneshot", "gpt-5.6-sol"),
    }
    for row in rows:
        if (
            (row.get("provider"), row.get("protocol"), row.get("model"))
            != expected[row["id"]]
            or row.get("capabilities") != ["coder", "structured_output"]
            or row.get("sandbox") != "read_only"
            or row.get("tools") != []
            or row.get("egress") is not True
            or row.get("max_attempts") != 1
            or row.get("fallback") != "none"
        ):
            raise ProductionReadinessError("provider roster profile drift")
    return tuple(dict(row) for row in rows)


class ProductionReadinessControl:
    """Write-once readiness epoch and immutable candidate-attempt lifecycle."""

    def __init__(
        self,
        *,
        root: str | os.PathLike[str],
        registry_path: str | os.PathLike[str],
        source_roots: Mapping[str, str | os.PathLike[str]],
        provider_roster_path: str | os.PathLike[str],
        functional_proof_path: str | os.PathLike[str],
    ) -> None:
        self.root = Path(root).absolute()
        if self.root.is_symlink():
            raise ProductionReadinessError("readiness root cannot be a symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.resolve(strict=True) != self.root:
            raise ProductionReadinessError("readiness root traverses a symlink")
        self.registry_path = Path(registry_path).absolute()
        self.source_roots = {name: Path(path).absolute() for name, path in source_roots.items()}
        self.provider_roster_path = Path(provider_roster_path).absolute()
        self.functional_proof_path = Path(functional_proof_path).absolute()
        self.epoch_path = self.root / "READINESS-EPOCH.json"
        self.attempt_root = self.root / "attempts"
        self.attempt_root.mkdir(exist_ok=True)
        self.lock_path = self.root / ".control.lock"
        self.candidates = CandidateControlPlane(self.attempt_root)

    @contextlib.contextmanager
    def _locked(self):
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _registry(self):
        try:
            return load_registry(self.registry_path, source_roots=self.source_roots)
        except (KernelTargetRegistryError, CandidateControlError) as exc:
            raise ProductionReadinessError("kernel registry validation failed") from exc

    def _provider_rows(self) -> tuple[dict[str, Any], ...]:
        try:
            return _providers(self.provider_roster_path)
        except CandidateControlError as exc:
            raise ProductionReadinessError("provider roster parse failed") from exc

    def _functional_proof(self, registry) -> dict[str, Any]:
        try:
            value = strict_json_file(self.functional_proof_path)
        except CandidateControlError as exc:
            raise ProductionReadinessError("functional proof parse failed") from exc
        keys = {
            "schema",
            "status",
            "registry_id",
            "registry_sha256",
            "target_count",
            "targets",
            "promotion_authority",
        }
        if not isinstance(value, dict) or set(value) != keys:
            raise ProductionReadinessError("functional proof key closure failed")
        if (
            value["schema"] != "hyperloom.gfx1151-functional-target-matrix.v3"
            or value["status"] != "PASS"
            or value["registry_id"] != registry.registry_id
            or value["registry_sha256"] != registry.manifest_sha256
            or value["target_count"] != len(registry.targets)
            or value["promotion_authority"] is not False
        ):
            raise ProductionReadinessError("functional proof identity drift")
        rows = value["targets"]
        if not isinstance(rows, list) or len(rows) != len(registry.targets):
            raise ProductionReadinessError("functional proof target count drift")
        observed = {}
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                "target_id", "status", "fallback_used", "promotion_authority"
            }:
                raise ProductionReadinessError("functional proof row closure failed")
            target_id = row["target_id"]
            if target_id in observed or target_id not in registry.targets:
                raise ProductionReadinessError("functional proof target identity drift")
            expected_status = (
                "CONTROL_ONLY"
                if registry.targets[target_id].status == "control_only"
                else "PASS"
            )
            if (
                row["status"] != expected_status
                or row["fallback_used"] is not False
                or row["promotion_authority"] is not False
            ):
                raise ProductionReadinessError("functional proof target verdict drift")
            observed[target_id] = row
        if set(observed) != set(registry.targets):
            raise ProductionReadinessError("functional proof roster drift")
        return value

    def _epoch_value(self) -> dict[str, Any]:
        registry = self._registry()
        providers = self._provider_rows()
        functional_proof = self._functional_proof(registry)
        return {
            "schema": "hyperloom.gfx1151-production-readiness-epoch.v1",
            "status": "READY_NOT_ACTIVATED",
            "registry": {
                "registry_id": registry.registry_id,
                "path": str(self.registry_path),
                "bytes": registry.manifest_bytes,
                "sha256": registry.manifest_sha256,
                "target_count": len(registry.targets),
            },
            "provider_roster": _meta(self.provider_roster_path),
            "functional_proof": {
                **_meta(self.functional_proof_path),
                "target_count": functional_proof["target_count"],
            },
            "providers": list(providers),
            "platform": dict(registry.platform),
            "gpu_lock": "/run/lock/hermes-vllm-gfx1151.lock",
            "provider_fallback": "none",
            "model_fallback": "none",
            "kernel_fallback": "none",
            "activation": "manual_external_approval_required",
            "production_activated": False,
            "promotion_authority": False,
        }

    def initialize(self) -> dict[str, Any]:
        with self._locked():
            expected = self._epoch_value()
            if self.epoch_path.exists():
                observed = strict_json_file(self.epoch_path)
                if observed != expected:
                    raise ProductionReadinessError("readiness epoch drift")
                return observed
            _write_new(self.epoch_path, expected)
            return expected

    def _require_epoch(self) -> dict[str, Any]:
        if not self.epoch_path.is_file() or self.epoch_path.is_symlink():
            raise ProductionReadinessError("readiness epoch is not initialized")
        observed = strict_json_file(self.epoch_path)
        if observed != self._epoch_value():
            raise ProductionReadinessError("readiness epoch drift")
        return observed

    def _provider_match(self, handoff: Mapping[str, Any]) -> bool:
        provider = handoff.get("provider")
        if not isinstance(provider, Mapping):
            return False
        return any(
            provider.get("provider") == row["provider"]
            and provider.get("protocol") == row["protocol"]
            and provider.get("model") == row["model"]
            and provider.get("fallback") == "none"
            for row in self._provider_rows()
        )

    def create_attempt(self, handoff: Mapping[str, Any]) -> Path:
        with self._locked():
            epoch = self._require_epoch()
            if not self._provider_match(handoff):
                raise ProductionReadinessError("handoff does not match provider roster")
            target = handoff.get("kernel_target")
            if not isinstance(target, Mapping):
                raise ProductionReadinessError("kernel target binding is missing")
            if (
                target.get("registry_id") != epoch["registry"]["registry_id"]
                or target.get("registry_sha256") != epoch["registry"]["sha256"]
                or target.get("fallback") != "none"
                or target.get("promotion_authority") is not False
            ):
                raise ProductionReadinessError("kernel target binding drift")
            registry = self._registry()
            selected = registry.targets.get(target.get("target_id"))
            if selected is None or selected.framework != target.get("framework"):
                raise ProductionReadinessError("kernel target is not exactly selected")
            try:
                return self.candidates.create_attempt(handoff)
            except CandidateControlError as exc:
                raise ProductionReadinessError("candidate attempt creation failed") from exc

    def run(self, attempt_id: str, *, generator, compiler, evaluator) -> dict[str, Any]:
        self._require_epoch()
        try:
            return self.candidates.run(
                attempt_id,
                generator=generator,
                compiler=compiler,
                evaluator=evaluator,
            )
        except CandidateControlError as exc:
            raise ProductionReadinessError("candidate execution failed") from exc

    def _attempt(self, attempt_id: str) -> Path:
        if not isinstance(attempt_id, str) or not attempt_id or "/" in attempt_id or ".." in attempt_id:
            raise ProductionReadinessError("invalid attempt id")
        path = self.attempt_root / attempt_id
        if not path.is_dir() or path.is_symlink():
            raise ProductionReadinessError("attempt is missing or invalid")
        return path

    def record_failure(self, attempt_id: str, *, category: str, message: str) -> dict[str, Any]:
        with self._locked():
            self._require_epoch()
            attempt = self._attempt(attempt_id)
            if (attempt / "candidate-result.json").exists():
                raise ProductionReadinessError("completed attempt cannot fail")
            value = {
                "schema": "hyperloom.production-attempt-failure.v1",
                "status": "RETRYABLE",
                "attempt_id": attempt_id,
                "category": category,
                "message": message,
                "fallback_used": False,
                "production_mutated": False,
                "promotion_authority": False,
            }
            _write_new(attempt / "failure.json", value)
            return value

    def resume(self, failed_attempt_id: str, new_attempt_id: str) -> Path:
        with self._locked():
            self._require_epoch()
            failed = self._attempt(failed_attempt_id)
            if not (failed / "failure.json").is_file() or (failed / "candidate-result.json").exists():
                raise ProductionReadinessError("attempt is not retryable")
            handoff = strict_json_file(failed / "candidate-handoff.json")
            handoff["attempt_id"] = new_attempt_id
            handoff["parent_attempt_id"] = failed_attempt_id
            if not self._provider_match(handoff):
                raise ProductionReadinessError("resumed provider binding drift")
            try:
                return self.candidates.create_attempt(handoff)
            except CandidateControlError as exc:
                raise ProductionReadinessError("resume attempt creation failed") from exc

    def record_cleanup(self, attempt_id: str, receipt: Mapping[str, Any]) -> dict[str, Any]:
        expected = {
            "kfd_clear": True,
            "containers_absent": True,
            "listeners_clear": True,
            "lock_released": True,
            "production_mutated": False,
            "promotion_authority": False,
        }
        with self._locked():
            self._require_epoch()
            attempt = self._attempt(attempt_id)
            if not (attempt / "candidate-result.json").is_file() or (attempt / "failure.json").exists():
                raise ProductionReadinessError("cleanup requires a completed nonfailed attempt")
            if dict(receipt) != expected:
                raise ProductionReadinessError("cleanup receipt drift")
            value = {
                "schema": "hyperloom.production-attempt-cleanup.v1",
                "status": "PASS",
                "attempt_id": attempt_id,
                **expected,
            }
            _write_new(attempt / "cleanup.json", value)
            return value

    def status(self) -> dict[str, Any]:
        epoch = self._require_epoch()
        counts = {"pending": 0, "retryable": 0, "complete": 0}
        attempts = []
        for path in sorted(self.attempt_root.iterdir()):
            if not path.is_dir() or path.is_symlink():
                raise ProductionReadinessError("attempt roster contains invalid entry")
            if (path / "failure.json").exists():
                state = "retryable"
            elif (path / "candidate-result.json").exists() and (path / "cleanup.json").exists():
                state = "complete"
            else:
                state = "pending"
            counts[state] += 1
            attempts.append({"attempt_id": path.name, "status": state.upper()})
        return {
            "schema": "hyperloom.gfx1151-production-readiness-status.v1",
            "healthy": True,
            "epoch": _meta(self.epoch_path),
            "registry_id": epoch["registry"]["registry_id"],
            "counts": counts,
            "attempts": attempts,
            "production_activated": False,
            "activation": "manual_external_approval_required",
            "promotion_authority": False,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--vllm-root", required=True)
    parser.add_argument("--sglang-root", required=True)
    parser.add_argument("--providers", required=True)
    parser.add_argument("--functional-proof", required=True)
    parser.add_argument("command", choices=("initialize", "status"))
    args = parser.parse_args(argv)
    control = ProductionReadinessControl(
        root=args.root,
        registry_path=args.registry,
        source_roots={"vllm_rocm10": args.vllm_root, "sglang_async_v7": args.sglang_root},
        provider_roster_path=args.providers,
        functional_proof_path=args.functional_proof,
    )
    result = control.initialize() if args.command == "initialize" else control.status()
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
