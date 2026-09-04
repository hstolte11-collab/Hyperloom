# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``CodexOneShotClient``: an ``AsyncOpenAI``-shaped client over the Codex SDK.

Mirror of :mod:`hyperloom.common.claude_oneshot` for the OpenAI side. The
Anthropic HTTP path rejects a Claude subscription token, so upstream drives
single-shot Claude calls through the Claude CLI behind a client that keeps the
``AnthropicMessageResult`` shape. The OpenAI ``chat.completions`` HTTP path
likewise rejects a Codex subscription login, so this client drives single-shot
calls through the Codex CLI (``run_codex_turn`` under ``native_oauth``) behind
the ``chat.completions.create`` shape every Hyperloom role already consumes:
critic reasoning, robustness RCA, proposal scorer, framework audit.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hyperloom.common import codex_oneshot as co
from hyperloom.common import codex_session as cs
from hyperloom.common import llm_config


def _home(tmp_path: Path) -> Path:
    home = tmp_path / "codex-oauth"
    home.mkdir()
    (home / "auth.json").write_text("{}")
    (home / "auth.json").chmod(0o600)
    return home


def test_client_is_openai_shaped_and_flattens_via_achat_completion(monkeypatch, tmp_path):
    home = _home(tmp_path)
    seen: dict = {}

    async def fake_turn(**kwargs):
        seen.update(kwargs)
        return cs.CodexSessionResult(
            text="REVIEW: approve",
            usage={
                "input_tokens": 120,
                "output_tokens": 7,
                "cache_read_input_tokens": 30,
                "reasoning_output_tokens": 2,
            },
            thread_id="t-1",
        )

    monkeypatch.setattr(co, "run_codex_turn", fake_turn)
    client = co.CodexOneShotClient(
        codex_home=str(home), cwd=tmp_path, timeout_s=42.0, component="critic", operation="review"
    )
    result = asyncio.run(
        llm_config.achat_completion(
            client,
            component="critic",
            operation="review",
            model="gpt-5.5",
            messages=[{"role": "system", "content": "You judge."}, {"role": "user", "content": "Proposal X"}],
            max_completion_tokens=256,
        )
    )
    # Same flattened contract every existing caller consumes.
    assert result.text == "REVIEW: approve"
    assert result.finish_reason == "stop"
    # OpenAI-spelled usage so critic/robustness accumulators keep working unchanged.
    assert result.usage.prompt_tokens == 120 and result.usage.completion_tokens == 7
    assert result.usage.total_tokens == 127
    # The turn was the native_oauth one-shot, tool-free/read-only, with the operator home.
    assert seen["auth_mode"] == "native_oauth" and seen["codex_home"] == str(home.resolve())
    assert seen["model"] == "gpt-5.5" and seen["timeout_sec"] == 42.0
    # Sandbox preset defers to the deployment (like CodexBackend); never any writable root.
    assert seen["sandbox_mode"] == "" and seen["writable_roots"] == ()
    assert seen["developer_instructions"] == "You judge." and seen["prompt"] == "Proposal X"
    assert seen["component"] == "critic" and seen["operation"] == "review"


def test_client_maps_length_stop_and_error(monkeypatch, tmp_path):
    home = _home(tmp_path)

    async def truncated(**_):
        return cs.CodexSessionResult(text="partial", usage={"output_tokens": 256}, thread_id="t")

    monkeypatch.setattr(co, "run_codex_turn", truncated)
    client = co.CodexOneShotClient(codex_home=str(home), cwd=tmp_path)
    r = asyncio.run(
        client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "x"}], max_completion_tokens=256
        )
    )
    assert r.choices[0].finish_reason == "length"

    async def failed(**_):
        return cs.CodexSessionResult(text="", usage={}, thread_id="t", error="rate limited")

    monkeypatch.setattr(co, "run_codex_turn", failed)
    with pytest.raises(co.CodexOneShotError, match="rate limited"):
        asyncio.run(client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}]))


def test_client_rejects_tools(monkeypatch, tmp_path):
    home = _home(tmp_path)
    client = co.CodexOneShotClient(codex_home=str(home), cwd=tmp_path)
    with pytest.raises(co.CodexOneShotError, match="tools"):
        asyncio.run(client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}], tools=[{}]))


def test_async_client_serves_astream_chat_completion_text(monkeypatch, tmp_path):
    """The proposal scorer streams (astream_chat_completion_text sets stream=True);
    the async client must replay the completed turn as an async chunk stream
    that yields the full text and the usage-only trailer."""
    home = _home(tmp_path)

    async def fake_turn(**kwargs):
        return cs.CodexSessionResult(text="score: 7", usage={"input_tokens": 9, "output_tokens": 3}, thread_id="t")

    monkeypatch.setattr(co, "run_codex_turn", fake_turn)
    client = co.CodexOneShotClient(codex_home=str(home), cwd=tmp_path)
    text, usage = asyncio.run(
        llm_config.astream_chat_completion_text(
            client,
            component="proposal_scorer",
            operation="score_proposal",
            model="m",
            messages=[{"role": "user", "content": "x"}],
            max_completion_tokens=64,
        )
    )
    assert text == "score: 7" and usage.completion_tokens == 3 and usage.prompt_tokens == 9


def test_factory_switches_every_openai_side_client_under_native_oauth(monkeypatch, tmp_path):
    """The seam every role uses: get_async_openai_client() returns the CLI client
    under native_oauth and the unchanged AsyncOpenAI factory otherwise."""
    home = _home(tmp_path)
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "LLM_GATEWAY_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(cs.CODEX_AUTH_MODE_ENV, "native_oauth")
    monkeypatch.setenv(cs.CODEX_HOME_ENV, str(home))
    client = llm_config.get_async_openai_client()
    assert isinstance(client, co.CodexOneShotClient)
    assert client.codex_home == str(home.resolve())
    # And the sync twin (framework audit / breakdown reporter use it).
    assert isinstance(llm_config.get_openai_client(), co.CodexOneShotClient)

    monkeypatch.delenv(cs.CODEX_AUTH_MODE_ENV)
    with pytest.raises(llm_config.LLMConfigError):
        llm_config.get_async_openai_client()  # default path: still needs a key


def test_run_codex_turn_threads_auth_mode_to_the_session(monkeypatch, tmp_path):
    home = _home(tmp_path)
    captured: dict = {}

    class FakeSession:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def start(self):
            return None

        async def aclose(self):
            return None

        async def turn(self, prompt, *, timeout_sec, output_schema=None):
            return cs.CodexSessionResult(text="ok")

    monkeypatch.setattr(cs, "CodexSession", FakeSession)
    asyncio.run(
        cs.run_codex_turn(
            prompt="p",
            developer_instructions="d",
            cwd=tmp_path,
            model="m",
            timeout_sec=1.0,
            auth_mode="native_oauth",
            codex_home=str(home),
        )
    )
    assert captured["auth_mode"] == "native_oauth" and captured["codex_home"] == str(home)
    captured.clear()
    asyncio.run(cs.run_codex_turn(prompt="p", developer_instructions="d", cwd=tmp_path, model="m", timeout_sec=1.0))
    assert captured["auth_mode"] == "gateway" and captured["codex_home"] == ""


def test_sync_client_serves_chat_completion_and_stream_helpers(monkeypatch, tmp_path):
    """The sync twin must satisfy llm_config.chat_completion and
    stream_chat_completion_text (framework audit / breakdown reporter)."""
    home = _home(tmp_path)

    async def fake_turn(**kwargs):
        return cs.CodexSessionResult(text="audit ok", usage={"input_tokens": 5, "output_tokens": 2}, thread_id="t")

    monkeypatch.setattr(co, "run_codex_turn", fake_turn)
    client = co.SyncCodexOneShotClient(codex_home=str(home), cwd=tmp_path)
    r = llm_config.chat_completion(
        client, component="framework", operation="audit", model="m", messages=[{"role": "user", "content": "x"}]
    )
    assert r.text == "audit ok" and r.usage.prompt_tokens == 5
    text, usage = llm_config.stream_chat_completion_text(
        client, component="framework", operation="audit", model="m", messages=[{"role": "user", "content": "x"}]
    )
    assert text == "audit ok" and usage.completion_tokens == 2
    with client:  # sync context-manager shape
        pass


def test_sync_client_refuses_running_loop(monkeypatch, tmp_path):
    home = _home(tmp_path)
    client = co.SyncCodexOneShotClient(codex_home=str(home), cwd=tmp_path)

    async def inside():
        with pytest.raises(co.CodexOneShotError, match="running event loop"):
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])

    asyncio.run(inside())


# --- transport readiness probe (mirror of anthropic_transport_ready) ----------------


def test_codex_transport_ready_requires_mode_valid_home_and_sdk(monkeypatch, tmp_path):
    """Readiness must check the TRANSPORT, not the mode string: a configured
    mode with a missing/unsafe home or an unimportable SDK is not ready."""
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "LLM_GATEWAY_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(cs.CODEX_AUTH_MODE_ENV, raising=False)
    monkeypatch.delenv(cs.CODEX_HOME_ENV, raising=False)
    # mode unset -> not a native transport
    assert llm_config.codex_transport_ready() is False

    monkeypatch.setenv(cs.CODEX_AUTH_MODE_ENV, "native_oauth")
    # mode set but no home
    assert llm_config.codex_transport_ready() is False
    # home without auth.json
    bare = tmp_path / "bare"
    bare.mkdir()
    monkeypatch.setenv(cs.CODEX_HOME_ENV, str(bare))
    assert llm_config.codex_transport_ready() is False
    # auth.json with loose mode
    (bare / "auth.json").write_text("{}")
    (bare / "auth.json").chmod(0o644)
    assert llm_config.codex_transport_ready() is False
    # valid home + SDK importable -> ready
    (bare / "auth.json").chmod(0o600)
    monkeypatch.setattr(cs, "load_codex_sdk", lambda: object())
    assert llm_config.codex_transport_ready() is True

    # valid home but SDK missing -> not ready
    def no_sdk():
        raise cs.CodexSessionUnavailableError("no sdk")

    monkeypatch.setattr(cs, "load_codex_sdk", no_sdk)
    assert llm_config.codex_transport_ready() is False


def test_role_gates_defer_to_codex_transport_ready(monkeypatch, tmp_path):
    """RCA engine and robustness factory gates must be the transport probe, so an
    unusable native transport is not treated as configured (the Claude precedent)."""
    from hyperloom.agents.robustness import factory as rf
    from hyperloom.agents.robustness.config import Config
    from hyperloom.agents.robustness.decision import rca_engine as rca

    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "LLM_GATEWAY_KEY"):
        monkeypatch.delenv(name, raising=False)
    home = _home(tmp_path)
    monkeypatch.setenv(cs.CODEX_AUTH_MODE_ENV, "native_oauth")
    monkeypatch.setenv(cs.CODEX_HOME_ENV, str(home))
    cfg = Config(session_dir=tmp_path / "s", llm_base_url="", llm_api_key="", llm_provider="openai", llm_model="m")
    eng = rca.LlmRcaEngine(base_url="", api_key="", model="m")

    monkeypatch.setattr(cs, "load_codex_sdk", lambda: object())
    assert eng._is_configured() is True and rf._llm_credentials_ready(cfg) is True

    def no_sdk():
        raise cs.CodexSessionUnavailableError("no sdk")

    monkeypatch.setattr(cs, "load_codex_sdk", no_sdk)
    assert eng._is_configured() is False and rf._llm_credentials_ready(cfg) is False

    # And an invalid home is not ready even with the SDK present.
    monkeypatch.setattr(cs, "load_codex_sdk", lambda: object())
    monkeypatch.setenv(cs.CODEX_HOME_ENV, str(tmp_path / "missing"))
    assert eng._is_configured() is False and rf._llm_credentials_ready(cfg) is False


def test_role_gates_branch_on_mode_not_fall_through_to_gateway_creds(monkeypatch, tmp_path):
    """Reviewer reproduction: native_oauth selected but home broken, AND stale
    gateway key/URL present. The gates must report NOT configured, because
    client construction under native_oauth always selects the (broken) native
    transport; stale gateway credentials must not mask it. Mirror of
    AnthropicRcaEngine: branch on the selected transport, then defer entirely."""
    from hyperloom.agents.robustness import factory as rf
    from hyperloom.agents.robustness.config import Config
    from hyperloom.agents.robustness.decision import rca_engine as rca

    monkeypatch.setenv(cs.CODEX_AUTH_MODE_ENV, "native_oauth")
    monkeypatch.setenv(cs.CODEX_HOME_ENV, str(tmp_path / "missing"))
    monkeypatch.setattr(cs, "load_codex_sdk", lambda: object())
    assert llm_config.codex_transport_ready() is False

    cfg = Config(
        session_dir=tmp_path / "s",
        llm_base_url="https://gw/v1",
        llm_api_key="stale",
        llm_provider="openai",
        llm_model="m",
    )
    eng = rca.LlmRcaEngine(base_url="https://gw/v1", api_key="stale", model="m")
    assert eng._is_configured() is False
    assert rf._llm_credentials_ready(cfg) is False

    # Gateway mode (unset) with the same creds: unchanged upstream behaviour.
    monkeypatch.delenv(cs.CODEX_AUTH_MODE_ENV)
    monkeypatch.delenv(cs.CODEX_HOME_ENV)
    assert eng._is_configured() is True
    assert rf._llm_credentials_ready(cfg) is True
