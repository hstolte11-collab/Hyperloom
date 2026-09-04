# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Opt-in ``INFERENCE_OPTIMIZER_CODEX_AUTH_MODE=native_oauth`` for the Orchestrator.

The Codex Orchestrator role then authenticates through the operator's own
``codex login`` state (``INFERENCE_OPTIMIZER_CODEX_HOME``), the same way
KernelForge's Codex backend already does, instead of an API key + gateway URL.
Default users (mode unset) see byte-identical behaviour.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

_PROVIDER_ENV = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_CUSTOM_HEADERS",
    "CODEX_API_KEY",
    "LLM_GATEWAY_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "INFERENCE_OPTIMIZER_CODEX_AUTH_MODE",
    "INFERENCE_OPTIMIZER_CODEX_HOME",
    "INFERENCE_OPTIMIZER_CLAUDE_FOLLOWS_CODEX",
)


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)


def _codex_home(tmp_path: Path, *, with_auth: bool = True) -> Path:
    home = tmp_path / "codex-home"
    home.mkdir()
    if with_auth:
        auth = home / "auth.json"
        auth.write_text(json.dumps({"tokens": {"access_token": "not-a-real-token"}}))
        auth.chmod(0o600)
    return home


def _enable(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CODEX_AUTH_MODE", "native_oauth")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CODEX_HOME", str(home))


# --- credentials preflight ----------------------------------------------------


def test_native_oauth_satisfies_credentials_without_any_api_key(monkeypatch, tmp_path):
    from hyperloom.inference_optimizer.cli import credentials

    _clear(monkeypatch)
    _enable(monkeypatch, _codex_home(tmp_path))
    credentials._validate_credentials()  # must not SystemExit


def test_native_oauth_requires_regular_0600_auth_json(monkeypatch, tmp_path):
    from hyperloom.inference_optimizer.cli import credentials

    _clear(monkeypatch)
    home = _codex_home(tmp_path, with_auth=False)
    _enable(monkeypatch, home)
    with pytest.raises(SystemExit):
        credentials._validate_credentials()
    auth = home / "auth.json"
    auth.write_text("{}")
    auth.chmod(0o644)
    with pytest.raises(SystemExit):
        credentials._validate_credentials()


def test_native_oauth_rejects_gateway_key_coexistence(monkeypatch, tmp_path):
    from hyperloom.inference_optimizer.cli import credentials

    _clear(monkeypatch)
    _enable(monkeypatch, _codex_home(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "would-shadow-the-login")
    with pytest.raises(SystemExit):
        credentials._validate_credentials()


@pytest.mark.parametrize(
    "name",
    [
        # Every variable that can authenticate or redirect the gateway path
        # (codex_session._API_KEY_ENV_FALLBACKS + endpoint/header knobs) or
        # drag in the Anthropic side. None may coexist with the CLI login.
        "OPENAI_API_KEY",
        "LLM_GATEWAY_KEY",
        "CODEX_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_CUSTOM_HEADERS",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ],
)
def test_native_oauth_rejects_every_gateway_or_anthropic_variable(monkeypatch, tmp_path, name):
    from hyperloom.inference_optimizer.cli import credentials

    _clear(monkeypatch)
    _enable(monkeypatch, _codex_home(tmp_path))
    monkeypatch.setenv(name, "set")
    with pytest.raises(SystemExit) as excinfo:
        credentials._validate_credentials()
    assert excinfo.value.code == 2


def test_native_oauth_conflict_list_covers_upstream_codex_key_fallbacks():
    """The conflict list must never drift behind the keys the gateway path accepts."""
    from hyperloom.common import codex_session as cs
    from hyperloom.inference_optimizer.cli import credentials

    for name in cs._API_KEY_ENV_FALLBACKS:
        assert name in credentials._NATIVE_OAUTH_CONFLICTING_ENV, name


def test_default_mode_still_requires_a_key(monkeypatch, tmp_path):
    from hyperloom.inference_optimizer.cli import credentials

    _clear(monkeypatch)
    with pytest.raises(SystemExit):
        credentials._validate_credentials()


# --- backend selection --------------------------------------------------------


def test_build_backends_native_oauth_selects_codex_orchestrator_with_home(monkeypatch, tmp_path):
    from hyperloom.inference_optimizer.cli import backends

    _clear(monkeypatch)
    home = _codex_home(tmp_path)
    _enable(monkeypatch, home)
    built = backends._build_backends(
        claude_model="unused",
        codex_model="gpt-5.5",
        critic_choice="mock",
        session_dir=tmp_path / "session",
        robustness_choice="mock",
    )
    orch = built["orchestration"]
    assert orch.name == "codex"
    assert orch.auth_mode == "native_oauth"
    assert orch.codex_home == str(home.resolve())
    assert orch.model == "gpt-5.5"


def test_build_backends_default_is_unchanged(monkeypatch, tmp_path):
    from hyperloom.inference_optimizer.cli import backends

    _clear(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    built = backends._build_backends(
        claude_model="unused",
        codex_model="gpt-5.5",
        critic_choice="mock",
        session_dir=tmp_path / "session",
        robustness_choice="mock",
    )
    orch = built["orchestration"]
    assert orch.name == "codex"
    assert orch.auth_mode == "gateway"
    assert orch.codex_home == ""


# --- model resolution ---------------------------------------------------------


def test_model_resolution_native_oauth_skips_bearer_catalog(monkeypatch, tmp_path):
    import hyperloom.inference_optimizer.cli as cli

    _clear(monkeypatch)
    _enable(monkeypatch, _codex_home(tmp_path))

    def boom(**_):
        raise AssertionError("bearer catalog probe must not run without a key")

    monkeypatch.setattr(cli, "_probe_llm_catalog", boom)
    args = argparse.Namespace(claude_model="unused", codex_model="gpt-5.5", critic_backend="mock")
    cli._resolve_models_for_run(args, None, claude_follows_codex=True, codex_follows_claude=False)
    assert args.claude_model == args.codex_model == "gpt-5.5"
