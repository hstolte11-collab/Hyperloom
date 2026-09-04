# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CodexSession ``native_oauth``: mirror KernelForge's already-qualified mode.

KernelForge's Codex backend (``kernelforge.agent_backends.codex``) supports
``auth_mode="native_oauth"``: no gateway ``model_providers`` override, a
caller-supplied pre-existing ``CODEX_HOME`` that owns the ChatGPT-subscription
login, and the API/gateway variables scrubbed from the child so they cannot
silently select a different transport. The Orchestrator's ``CodexSession``
must offer the same contract with identical defaults for gateway mode.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.common import codex_session as cs


def _env_without_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_CUSTOM_HEADERS", "CODEX_API_KEY", "LLM_GATEWAY_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_default_auth_mode_is_gateway_and_unchanged() -> None:
    session = cs.CodexSession(cwd=Path("/tmp"), model="gpt-5.5")
    assert session.auth_mode == "gateway"
    assert session.codex_home == ""


def test_unknown_auth_mode_is_rejected() -> None:
    with pytest.raises(cs.CodexSessionUnavailableError, match="auth_mode"):
        cs.CodexSession(cwd=Path("/tmp"), model="gpt-5.5", auth_mode="magic")


def test_native_oauth_requires_absolute_existing_nonsymlink_home(tmp_path: Path) -> None:
    with pytest.raises(cs.CodexSessionUnavailableError, match="CODEX_HOME"):
        cs.CodexSession(cwd=tmp_path, model="gpt-5.5", auth_mode="native_oauth")
    with pytest.raises(cs.CodexSessionUnavailableError, match="absolute"):
        cs.CodexSession(cwd=tmp_path, model="gpt-5.5", auth_mode="native_oauth", codex_home="relative")
    missing = tmp_path / "missing"
    with pytest.raises(cs.CodexSessionUnavailableError, match="directory"):
        cs.CodexSession(cwd=tmp_path, model="gpt-5.5", auth_mode="native_oauth", codex_home=str(missing))
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(cs.CodexSessionUnavailableError, match="symlink"):
        cs.CodexSession(cwd=tmp_path, model="gpt-5.5", auth_mode="native_oauth", codex_home=str(link))


def test_native_oauth_child_env_uses_home_and_strips_gateway(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _env_without_gateway(monkeypatch)
    home = tmp_path / "codex-oauth"
    home.mkdir()
    session = cs.CodexSession(cwd=tmp_path, model="gpt-5.5", auth_mode="native_oauth", codex_home=str(home))
    child, overrides = session._native_oauth_launch(
        {
            "PATH": "/usr/bin",
            "OPENAI_API_KEY": "must-not-leak",
            "OPENAI_BASE_URL": "https://gateway.example/v1",
            "OPENAI_CUSTOM_HEADERS": "X: y",
            "CODEX_API_KEY": "nope",
            "LLM_GATEWAY_KEY": "also-nope",
        }
    )
    assert child["CODEX_HOME"] == str(home.resolve())
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_CUSTOM_HEADERS", "CODEX_API_KEY", "LLM_GATEWAY_KEY"):
        assert name not in child
    # Every name the gateway path could authenticate with is scrubbed.
    assert set(cs._API_KEY_ENV_FALLBACKS) <= set(cs._NATIVE_OAUTH_SCRUBBED_ENV)
    assert child["PATH"] == "/usr/bin"
    # No gateway model_provider is forced: the CLI's own login/provider applies.
    assert not any(o.startswith("model_provider") or o.startswith("model_providers.") for o in overrides)
    assert "features.memories=false" in overrides


def test_gateway_mode_still_requires_key_and_base_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _env_without_gateway(monkeypatch)
    session = cs.CodexSession(cwd=tmp_path, model="gpt-5.5")
    with pytest.raises(cs.CodexSessionUnavailableError, match="Codex gateway credential is missing"):
        cs._resolve_codex_provider_config(api_key_env=session.api_key_env, base_url_env=session.base_url_env, source={})


# --- end-to-end through the fake SDK (start -> thread_start -> turn -> close) --


def _fake_sdk(monkeypatch: pytest.MonkeyPatch):
    # Reuse the established fake SDK/sandbox fixtures so this file cannot drift
    # from the gateway tests' notion of what the SDK receives.
    from hyperloom.agents.kernel.tests import test_codex_session as base

    return base._install_fake_sdk(monkeypatch), base


def test_native_oauth_thread_start_has_no_model_provider_and_keeps_operator_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import asyncio

    _env_without_gateway(monkeypatch)
    record, _ = _fake_sdk(monkeypatch)
    home = tmp_path / "codex-oauth"
    home.mkdir()
    (home / "auth.json").write_text("{}")
    session = cs.CodexSession(
        cwd=tmp_path, model="gpt-5.5", auth_mode="native_oauth", codex_home=str(home), env={"PATH": "/usr/bin"}
    )

    async def go() -> None:
        await session.start()
        await session.turn("hello", timeout_sec=5.0)
        await session.aclose()

    asyncio.run(go())
    # KernelForge contract: model_provider is only set in gateway mode.
    assert "model_provider" not in record["thread_options"]
    assert record["thread_options"]["model"] == "gpt-5.5"
    # The operator's home is used as-is and is NOT removed on close.
    assert record["codex_home_at_enter"] == str(home.resolve())
    assert record["codex_home_exists_when_closed"] is True
    assert home.is_dir() and (home / "auth.json").is_file()
    # No gateway provider config reached the CLI; env was scrubbed.
    overrides = record["config"].kwargs["config_overrides"]
    assert not any(o.startswith("model_provider") for o in overrides)
    assert "OPENAI_API_KEY" not in record["config"].kwargs["env"]


def test_gateway_thread_start_still_pins_the_hyperloom_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import asyncio

    record, base = _fake_sdk(monkeypatch)
    session = cs.CodexSession(cwd=tmp_path, model="gpt-5.5", env=base._gateway_env())

    async def go() -> None:
        await session.start()
        await session.turn("hello", timeout_sec=5.0)
        await session.aclose()

    asyncio.run(go())
    assert record["thread_options"]["model_provider"] == cs.CODEX_PROVIDER_NAME
    assert any(
        o.startswith(f"model_providers.{cs.CODEX_PROVIDER_NAME}.") for o in record["config"].kwargs["config_overrides"]
    )
    # Gateway mode still uses a private throwaway home, removed once aclose() returns.
    private_home = Path(record["codex_home_at_enter"])
    assert private_home != tmp_path and not private_home.exists()
