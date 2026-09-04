"""CLI contract regressions for the quantization agent."""

from __future__ import annotations

import pytest

from hyperloom.agents.quantization.cli import _parse_args


@pytest.mark.parametrize("provider", ["codex", "hermes"])
def test_parse_args_accepts_faithful_provider_adapters(provider: str) -> None:
    args = _parse_args(
        [
            "--prompt",
            "quantize the model",
            "--workspace",
            "/tmp/quark-workspace",
            "--provider",
            provider,
        ]
    )

    assert args.provider == provider
