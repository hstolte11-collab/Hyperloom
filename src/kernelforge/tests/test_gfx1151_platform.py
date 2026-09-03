# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RED contract tests for the strict KernelForge gfx1151 lane."""

from dataclasses import asdict

import pytest

from kernelforge.config import Config
from kernelforge.platforms import (
    PlatformContractError,
    get_platform_profile,
    strict_gfx1151_runtime,
    validate_backend,
)


EXPECTED = {
    "isa_target": "gfx1151",
    "board_identity": "strix-halo-radeon-8060s",
    "gpu_name": "AMD Radeon 8060S",
    "architecture_class": "RDNA3.5",
    "platform_class": "RocmPlatform",
    "runtime_lane": "rocm10-isolated",
    "rocm_root": "/opt/rocm/core-10.0",
    "offload_arch": "gfx1151",
}


def test_exact_gfx1151_profile_is_explicit_and_immutable():
    profile = get_platform_profile("gfx1151", "strix-halo-radeon-8060s")
    values = asdict(profile) if hasattr(profile, "__dataclass_fields__") else vars(profile)
    assert {key: values[key] for key in EXPECTED} == EXPECTED
    assert values["hsa_override_gfx_version"] == "forbidden"


@pytest.mark.parametrize(
    "isa,board",
    [("gfx1151", ""), ("gfx1151", "unknown"), ("gfx942", "strix-halo-radeon-8060s"),
     ("gfx950", "strix-halo-radeon-8060s"), ("gfx1151", "mi355x"), ("mi355x", "mi355x")],
)
def test_bare_unknown_foreign_and_architecture_spoofed_profiles_rejected(isa, board):
    with pytest.raises(PlatformContractError):
        get_platform_profile(isa, board)


@pytest.mark.parametrize("backend", ["aiter", "ck", "flydsl", "fusion", "gluon", "hipblaslt"])
def test_unqualified_gfx1151_backend_rejected_before_execution(backend):
    profile = get_platform_profile("gfx1151", "strix-halo-radeon-8060s")
    with pytest.raises(PlatformContractError):
        validate_backend(profile, backend)


@pytest.mark.parametrize("backend", ["hip", "triton"])
def test_hip_and_triton_are_candidates_not_proven_supported(backend):
    profile = get_platform_profile("gfx1151", "strix-halo-radeon-8060s")
    with pytest.raises(PlatformContractError, match="qualif"):
        validate_backend(profile, backend)


def test_strict_config_rejects_source_current_defaults_and_hsa_override():
    config = Config(gpu_target="gfx1151", gpu_type="strix-halo-radeon-8060s")
    config.agent_backend = "auto"
    config.agent_model = ""
    config.agent_fallback_provider = "claude"
    config.agent_options = {"hsa_override_gfx_version": "11.0.0"}
    with pytest.raises(PlatformContractError):
        strict_gfx1151_runtime(config)


@pytest.mark.parametrize("kwargs", [
    {"agent_backend": "endpoint_agnostic", "agent_model": "m", "agent_fallback_provider": "", "agent_options": {}},
    {"agent_backend": "endpoint_agnostic", "agent_model": "m", "agent_fallback_provider": "none", "agent_options": {"protocol": "p", "provider": "q"}},
])
def test_strict_config_requires_explicit_runner_and_exact_rocm_root(kwargs):
    config = Config(gpu_target="gfx1151", gpu_type="strix-halo-radeon-8060s", **kwargs)
    with pytest.raises(PlatformContractError):
        strict_gfx1151_runtime(config)


def test_strict_config_accepts_one_exact_endpoint_runner_profile():
    config = Config(
        gpu_target="gfx1151",
        gpu_type="strix-halo-radeon-8060s",
        agent_backend="endpoint_agnostic",
        agent_model="explicit-model",
        agent_fallback_provider="",
        agent_options={
            "provider": "openai_compatible",
            "protocol": "openai_compatible",
            "api_key_env": None,
            "base_url": "http://127.0.0.1:18090/v1",
            "capabilities": ["coder", "tools", "structured_output", "session_resume"],
            "fallback": "none",
            "rocm_root": "/opt/rocm/core-10.0",
        },
    )
    validated = strict_gfx1151_runtime(config)
    assert validated.gpu_target == "gfx1151"
    assert validated.gpu_type == "strix-halo-radeon-8060s"
    assert validated.agent_backend == "endpoint_agnostic"
    assert validated.agent_fallback_provider in {"", "none"}


def test_strict_resolver_does_not_change_legacy_gfx942_or_gfx950_controls():
    assert get_platform_profile("gfx942", "mi300x") is not None
    assert get_platform_profile("gfx950", "mi355x") is not None
