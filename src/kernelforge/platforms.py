# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Explicit platform profiles and fail-closed backend/runtime validation."""
from dataclasses import dataclass

class PlatformContractError(ValueError):
    """The requested platform or runtime violates its frozen contract."""

@dataclass(frozen=True)
class PlatformProfile:
    isa_target: str
    board_identity: str
    gpu_name: str
    architecture_class: str
    platform_class: str
    runtime_lane: str
    rocm_root: str
    offload_arch: str
    hsa_override_gfx_version: str

_PROFILES = {
    ("gfx942", "mi300x"): PlatformProfile("gfx942", "mi300x", "AMD Instinct MI300X", "CDNA3", "RocmPlatform", "rocm", "/opt/rocm", "gfx942", "forbidden"),
    ("gfx950", "mi355x"): PlatformProfile("gfx950", "mi355x", "AMD Instinct MI355X", "CDNA4", "RocmPlatform", "rocm", "/opt/rocm", "gfx950", "forbidden"),
    ("gfx1151", "strix-halo-radeon-8060s"): PlatformProfile("gfx1151", "strix-halo-radeon-8060s", "AMD Radeon 8060S", "RDNA3.5", "RocmPlatform", "rocm10-isolated", "/opt/rocm/core-10.0", "gfx1151", "forbidden"),
}
_UNSUPPORTED = {"aiter", "ck", "flydsl", "fusion", "gluon", "hipblaslt"}

def get_platform_profile(isa_target, board_identity):
    try: return _PROFILES[(isa_target.strip().lower(), board_identity.strip().lower())]
    except (AttributeError, KeyError): raise PlatformContractError("explicit platform profile is required") from None

def validate_backend(profile, backend):
    backend = (backend or "").strip().lower()
    if profile.isa_target == "gfx1151":
        if backend in _UNSUPPORTED: raise PlatformContractError(f"backend {backend!r} is unsupported for gfx1151")
        if backend in {"hip", "triton"}: raise PlatformContractError(f"backend {backend!r} requires qualification")
        raise PlatformContractError("backend must be explicitly qualified for gfx1151")
    return backend

def strict_gfx1151_runtime(config):
    profile = get_platform_profile(config.gpu_target, config.gpu_type)
    if config.agent_backend != "endpoint_agnostic" or not config.agent_model.strip(): raise PlatformContractError("explicit endpoint_agnostic backend and model required")
    if config.agent_fallback_provider not in {"", "none"}: raise PlatformContractError("provider fallback must be none")
    o = config.agent_options
    required = ("provider", "protocol", "capabilities", "fallback")
    if any(not o.get(k) for k in required) or o.get("fallback") != "none": raise PlatformContractError("explicit provider, protocol, capabilities and fallback none required")
    if o.get("rocm_root", profile.rocm_root) != profile.rocm_root: raise PlatformContractError("exact ROCm root required")
    if "hsa_override_gfx_version" in o or o.get("auto") or o.get("claude_fallback"): raise PlatformContractError("HSA override/auto/fallback forbidden")
    return config
