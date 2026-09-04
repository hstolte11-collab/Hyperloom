# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Radeon 8060S (Strix Halo, gfx1151, 40 CU) is a first-class board.

Magpie ships ``vllm_radeon8060s.sh`` / ``sglang_radeon8060s.sh``. Without a
board entry here, ``--gpu-type`` rejects it, autodetect returns ``None`` on the
hardware, and the optimizer never reaches those runners.
"""

from __future__ import annotations

from hyperloom.common.gpu_identity import AMD_GPU_DISPATCH_IDENTITIES, gfx_arch_for_gpu_type
from hyperloom.inference_optimizer import gpu_types


def test_radeon8060s_has_dispatch_identity():
    assert AMD_GPU_DISPATCH_IDENTITIES["radeon8060s"] == ("gfx1151", 40)
    assert gfx_arch_for_gpu_type("radeon8060s") == "gfx1151"
    assert gpu_types.amd_gpu_dispatch_identity("radeon8060s") == ("gfx1151", 40)


def test_gfx1151_arch_selects_the_radeon8060s_runner():
    assert gpu_types._GFX_TO_RUNNER["gfx1151"] == "radeon8060s"
    assert gpu_types._gpu_runner_type("radeon8060s") == "radeon8060s"


def test_autodetect_from_rocm_smi_product_name(monkeypatch):
    """rocm-smi on Strix Halo prints ``Card Series: Radeon 8060S Graphics``."""
    import subprocess

    class _Done:
        stdout = "GPU[0]\t\t: Card Series: \t\tRadeon 8060S Graphics\nGPU[0]\t\t: GFX Version: \t\tgfx1151\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Done())
    assert gpu_types._autodetect_gpu_type() == "radeon8060s"


def test_autodetect_from_torch_gcn_arch_when_rocm_smi_absent(monkeypatch):
    import subprocess
    import types

    def _missing(*a, **k):
        raise FileNotFoundError("rocm-smi")

    monkeypatch.setattr(subprocess, "run", _missing)
    props = types.SimpleNamespace(gcnArchName="gfx1151:sramecc-:xnack-")
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_properties=lambda i: props))
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    assert gpu_types._autodetect_gpu_type() == "radeon8060s"


def test_product_tag_resolution_is_shared_with_the_remote_probe():
    """One parser for rocm-smi text, used by local autodetect AND the multi-node probe.

    The remote probe (rayjob/infera) used to carry its own copy of the tag loop
    and therefore missed the alias -- the Radeon 8060S resolved locally but not
    on a handed-over cluster.
    """
    from hyperloom.inference_optimizer.multi_node._internal import gpu_probe

    smi = "GPU[0]\t\t: Card Series: \t\tRadeon 8060S Graphics\nGPU[0]\t\t: Card Model: \t\t0x1586\n"
    assert gpu_types.gpu_type_from_product_text(smi) == "radeon8060s"
    assert gpu_probe._parse_gpu_type(smi) == "radeon8060s"
    # Instinct still resolves via its own name, longer tag first.
    assert gpu_probe._parse_gpu_type("Card Series: AMD Instinct MI300X OAM") == "mi300x"
    assert gpu_probe._parse_gpu_type("Card Series: Instinct MI355X") == "mi355x"
    # gcnArchName fallback still works for both consumers.
    assert gpu_probe._parse_gpu_type("gfx1151:sramecc-:xnack-") == "radeon8060s"
    assert gpu_probe._parse_gpu_type("nothing here") is None
