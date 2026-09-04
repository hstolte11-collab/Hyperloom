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


# --- framework specialist (CodexAgentBackend) -----------------------------------


def test_specialist_codex_backend_threads_native_oauth_into_run_codex_turn(monkeypatch, tmp_path):
    import asyncio

    from hyperloom.common import codex_session as cs
    from hyperloom.orchestrator.roles import codex_agent as ca

    home = _codex_home(tmp_path)
    seen: dict = {}

    envelope = json.dumps(
        {"intents": [{"intent_type": "send_message", "payload": {"topic": "heartbeat", "body_md": "ok"}}]}
    )

    async def fake_turn(**kwargs):
        seen.update(kwargs)
        return cs.CodexSessionResult(text=envelope, usage={"input_tokens": 1, "output_tokens": 1}, thread_id="t")

    monkeypatch.setattr(ca, "run_codex_turn", fake_turn)
    monkeypatch.setattr(ca, "resolve_codex_sandbox_mode", lambda **_: "workspace-write")
    backend = ca.CodexAgentBackend(
        model="gpt-5.5",
        cwd=tmp_path / "rt",
        writable_roots=(tmp_path / "rt",),
        auth_mode="native_oauth",
        codex_home=str(home),
    )
    asyncio.run(backend.run("do the task", system_prompt="you are a specialist"))
    assert seen["auth_mode"] == "native_oauth" and seen["codex_home"] == str(home)

    # Default is the prior gateway contract.
    seen.clear()
    backend = ca.CodexAgentBackend(model="gpt-5.5", cwd=tmp_path / "rt2", writable_roots=(tmp_path / "rt2",))
    asyncio.run(backend.run("do the task", system_prompt="s"))
    assert seen["auth_mode"] == "gateway" and seen["codex_home"] == ""


def test_specialist_executor_selects_codex_with_native_oauth_and_no_key(monkeypatch, tmp_path):
    """With native_oauth and no OPENAI_* set, the specialist must still resolve to the
    Codex agent backend (not silently fall to Claude) and carry the operator home."""
    from hyperloom.orchestrator.specialists import subprocess_ as sp

    _clear(monkeypatch)
    _enable(monkeypatch, _codex_home(tmp_path))
    assert sp.resolve_specialist_agent_backend() == sp.AGENT_BACKEND_CODEX


# --- robustness RCA engine ----------------------------------------------------


def test_robustness_rca_engine_is_configured_under_native_oauth_without_key(monkeypatch, tmp_path):
    """Mirror of AnthropicRcaEngine: a subscription login reaches this process as
    neither base_url nor api_key, so the configured test must defer to the
    transport probe instead of `base_url and api_key`."""
    import asyncio

    from hyperloom.agents.robustness.decision import rca_engine as rca
    from hyperloom.common import codex_oneshot as co

    _clear(monkeypatch)
    _enable(monkeypatch, _codex_home(tmp_path))
    engine = rca.LlmRcaEngine(base_url="", api_key="", model="gpt-5.5")
    assert engine._is_configured() is True
    client = engine._ensure_client()
    assert isinstance(client, co.CodexOneShotClient)
    assert client.component == "robustness"
    asyncio.run(engine.aclose()) if hasattr(engine, "aclose") else None


def test_robustness_rca_engine_default_still_requires_key_and_url(monkeypatch):
    from hyperloom.agents.robustness.decision import rca_engine as rca

    _clear(monkeypatch)
    engine = rca.LlmRcaEngine(base_url="", api_key="", model="gpt-5.5")
    assert engine._is_configured() is False
    engine = rca.LlmRcaEngine(base_url="https://x/v1", api_key="k", model="gpt-5.5")
    assert engine._is_configured() is True


def test_robustness_config_discovers_native_oauth_as_openai_provider(monkeypatch, tmp_path):
    from hyperloom.agents.robustness import config as rcfg

    _clear(monkeypatch)
    _enable(monkeypatch, _codex_home(tmp_path))
    base_url, api_key, provider = rcfg._discover_llm_credentials()
    assert provider == "openai" and api_key == "" and base_url == ""


# --- critic agent --------------------------------------------------------------


def test_critic_agent_backend_uses_codex_oneshot_client_under_native_oauth(monkeypatch, tmp_path):
    """The critic's `protocol=openai` review transport is upstream's
    get_async_openai_client(); under native_oauth that yields the Codex CLI
    client and construction must succeed with no key in the environment."""
    from hyperloom.common import codex_oneshot as co
    from hyperloom.orchestrator.roles import critic_agent as ca

    _clear(monkeypatch)
    _enable(monkeypatch, _codex_home(tmp_path))
    root = tmp_path / "critic-agent"
    (root / "runtime").mkdir(parents=True)
    (root / "runtime" / "cli.py").write_text("# stub\n")
    backend = ca.CriticAgentBackend(
        critic_agent_root=root,
        session_dir=tmp_path / "session",
        codex_model="gpt-5.5",
        runtime_caller_factory=lambda: lambda call: None,
    )
    assert backend.protocol == "openai"
    assert isinstance(backend._client, co.CodexOneShotClient)
    assert backend._review_model == "gpt-5.5"


def test_build_backends_native_oauth_critic_agent_protocol_is_openai(monkeypatch, tmp_path):
    from hyperloom.inference_optimizer.cli import backends
    from hyperloom.common import codex_oneshot as co

    _clear(monkeypatch)
    _enable(monkeypatch, _codex_home(tmp_path))
    root = tmp_path / "critic-agent"
    (root / "runtime").mkdir(parents=True)
    (root / "runtime" / "cli.py").write_text("# stub\n")
    built = backends._build_backends(
        claude_model="unused",
        codex_model="gpt-5.5",
        critic_choice="agent",
        session_dir=tmp_path / "session",
        robustness_choice="mock",
        critic_agent_root=root,
    )
    critic = built["critic"]
    assert critic.protocol == "openai"
    assert isinstance(critic._client, co.CodexOneShotClient)


def test_robustness_factory_credentials_present_under_native_oauth(monkeypatch, tmp_path):
    """The robustness-agent child process builds its engine from Config.discover();
    its credential gate must accept native_oauth like it accepts a Claude token."""
    from hyperloom.agents.robustness import factory as rf
    from hyperloom.agents.robustness.config import Config

    _clear(monkeypatch)
    _enable(monkeypatch, _codex_home(tmp_path))
    cfg = Config(
        session_dir=tmp_path / "s", llm_base_url="", llm_api_key="", llm_provider="openai", llm_model="gpt-5.5"
    )
    assert rf._llm_credentials_ready(cfg) is True
    _clear(monkeypatch)
    assert rf._llm_credentials_ready(cfg) is False


def test_framework_audit_refine_reaches_client_under_native_oauth(monkeypatch, tmp_path):
    """audit.py's gateway pre-check (resolve_openai_client_config) must not skip
    the refine step when native_oauth is the credential."""
    import hyperloom.common.llm_config as llm_cfg
    from hyperloom.agents.framework import audit
    from hyperloom.common import codex_oneshot as co

    _clear(monkeypatch)
    _enable(monkeypatch, _codex_home(tmp_path))
    seen = {}

    def fake_stream(client, **params):
        seen["client"] = client
        return ('{"semantic_status": "not_present", "risks": [], "layer": "llm"}', None)

    monkeypatch.setattr(llm_cfg, "stream_chat_completion_text", fake_stream)
    static = {"risks": [], "semantic_status": "not_present", "applicability": "applicable"}
    result = audit._maybe_llm_refine({"use_llm": True, "model": "gpt-5.5"}, static, "diff --git a b")
    assert isinstance(seen.get("client"), co.SyncCodexOneShotClient), result
    assert not any("llm refine skipped" in r for r in result.get("risks", [])), result


# --- proposal scorer --------------------------------------------------------------


def test_proposal_scorer_scores_through_codex_oneshot_under_native_oauth(monkeypatch, tmp_path):
    """The scorer's only transport is astream_chat_completion_text over
    get_async_openai_client(); under native_oauth that must reach the Codex CLI
    client and return a usable score, not refuse streaming."""
    import asyncio

    from hyperloom.common import codex_oneshot as co
    from hyperloom.common import codex_session as cs
    from hyperloom.orchestrator.scoring import proposal_scorer as ps

    _clear(monkeypatch)
    _enable(monkeypatch, _codex_home(tmp_path))

    async def fake_turn(**kwargs):
        return cs.CodexSessionResult(
            text='{"score": 0.8, "rationale": "fine"}', usage={"input_tokens": 5, "output_tokens": 4}, thread_id="t"
        )

    monkeypatch.setattr(co, "run_codex_turn", fake_turn)
    scorer = (
        ps.ProposalScorer(model="gpt-5.5")
        if "model" in ps.ProposalScorer.__init__.__code__.co_varnames
        else ps.ProposalScorer()
    )
    client = scorer._client if getattr(scorer, "_client", None) is not None else None
    if client is None and hasattr(scorer, "_ensure_client"):
        client = scorer._ensure_client()
    assert isinstance(client, co.CodexOneShotClient), type(client)
    from hyperloom.common import llm_config

    text, usage = asyncio.run(
        llm_config.astream_chat_completion_text(
            client,
            component="proposal_scorer",
            operation="score_proposal",
            model="gpt-5.5",
            messages=[{"role": "user", "content": "p"}],
        )
    )
    assert "0.8" in text and usage.completion_tokens == 4


def test_framework_audit_broken_native_oauth_does_not_fall_through_to_stale_gateway_creds(monkeypatch, tmp_path):
    """If native_oauth is selected but unavailable, stale gateway credentials must
    not let the best-effort audit path proceed as though the selected transport
    were ready. It should degrade to the static verdict without calling the LLM."""
    import hyperloom.common.llm_config as llm_cfg
    from hyperloom.agents.framework import audit
    from hyperloom.common import codex_session as cs

    home = _codex_home(tmp_path)
    monkeypatch.setenv(cs.CODEX_AUTH_MODE_ENV, "native_oauth")
    monkeypatch.setenv(cs.CODEX_HOME_ENV, str(home))
    monkeypatch.setenv("OPENAI_API_KEY", "stale")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.invalid/v1")

    def no_sdk():
        raise cs.CodexSessionUnavailableError("no sdk")

    monkeypatch.setattr(cs, "load_codex_sdk", no_sdk)

    def should_not_stream(*_args, **_kwargs):
        raise AssertionError("audit refine must not call stale gateway or broken native transport")

    monkeypatch.setattr(llm_cfg, "stream_chat_completion_text", should_not_stream)
    static = {"risks": [], "semantic_status": "not_present", "applicability": "applicable"}
    result = audit._maybe_llm_refine({"use_llm": True, "model": "gpt-5.5"}, static, "diff --git a b")
    assert result is static
    assert any("native_oauth" in r for r in result["risks"]), result
