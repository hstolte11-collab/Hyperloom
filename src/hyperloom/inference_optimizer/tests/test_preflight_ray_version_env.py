# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``_ensure_ray`` must honour the same ``RAY_VERSION`` / ``RAY_CLI_CLICK_MAX_VERSION``
environment overrides as ``agents/kernel/scripts/install.sh``.

A container image can legitimately ship a newer Ray (for example on a Python
whose interpreter has no 2.44.1 wheel). ``install.sh`` already accepts that via
``RAY_VERSION``; the Python preflight hard-pinned the default and would try to
``pip install`` a downgrade into a pinned image and then fail.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest


def _reload_preflight(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import hyperloom.inference_optimizer.cli.preflight as pf

    return importlib.reload(pf)


def test_default_pins_are_unchanged(monkeypatch):
    monkeypatch.delenv("RAY_VERSION", raising=False)
    monkeypatch.delenv("RAY_CLI_CLICK_MAX_VERSION", raising=False)
    pf = _reload_preflight(monkeypatch)
    assert pf._RAY_VERSION == "2.44.1"
    assert pf._RAY_CLI_CLICK_MAX_VERSION == "8.3.0"
    assert pf._RAY_INSTALL_SPEC == "ray[default]==2.44.1"
    assert pf._CLICK_INSTALL_SPEC == "click<8.3.0"


def test_env_overrides_flow_into_smoke_and_install_spec(monkeypatch):
    pf = _reload_preflight(monkeypatch, RAY_VERSION="2.55.0", RAY_CLI_CLICK_MAX_VERSION="9.0.0")
    assert pf._RAY_VERSION == "2.55.0"
    assert pf._RAY_INSTALL_SPEC == "ray[default]==2.55.0"
    assert pf._CLICK_INSTALL_SPEC == "click<9.0.0"
    assert 'RAY_VERSION = "2.55.0"' in pf._RAY_SMOKE
    assert 'RAY_CLI_CLICK_MAX_VERSION = "9.0.0"' in pf._RAY_SMOKE


def test_ensure_ray_does_not_pip_install_when_env_matches_installed(monkeypatch):
    """With the env pointing at the installed Ray, the smoke passes and pip is never invoked."""
    # Exercised for real inside the runtime image. Other tests in this suite
    # register stub ``ray`` modules in ``sys.modules``; require a genuine,
    # importable-in-a-subprocess Ray (the smoke runs in a subprocess) or skip.
    probe = subprocess.run(
        [sys.executable, "-c", "import ray, sys; sys.stdout.write(ray.__version__)"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        pytest.skip("real ray not installed in this interpreter")
    installed_ray_version = probe.stdout.strip()

    pf = _reload_preflight(monkeypatch, RAY_VERSION=installed_ray_version, RAY_CLI_CLICK_MAX_VERSION="99.0.0")
    calls: list[list[str]] = []
    real_run = subprocess.run

    def spy(cmd, *a, **k):
        calls.append(list(cmd))
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(pf.subprocess, "run", spy)
    pf._ensure_ray(sys.executable, [])
    assert not any("install" in c for c in calls), calls


def test_ray_version_env_values_cannot_inject_code(monkeypatch):
    """The overrides are interpolated into a ``python -c`` program; a hostile
    value must be rejected as a non-version, never executed."""
    hostile = '"; __import__("builtins").print("INJECTED"); #'
    with pytest.raises(ValueError):
        _reload_preflight(monkeypatch, RAY_VERSION=hostile, RAY_CLI_CLICK_MAX_VERSION="8.3.0")
    with pytest.raises(ValueError):
        _reload_preflight(monkeypatch, RAY_VERSION="2.55.0", RAY_CLI_CLICK_MAX_VERSION=hostile)
    # Well-formed versions are accepted and appear only as quoted literals.
    pf = _reload_preflight(monkeypatch, RAY_VERSION="2.55.0", RAY_CLI_CLICK_MAX_VERSION="9.0.0")
    assert 'RAY_VERSION = "2.55.0"' in pf._RAY_SMOKE
    assert "INJECTED" not in pf._RAY_SMOKE
