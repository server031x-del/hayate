from __future__ import annotations

import subprocess

from hayate.hardware import HardwareProfiler


def test_hardware_profiler_parses_two_gpus_and_selects_gpu_zero(monkeypatch):
    def runner(command, **kwargs):
        if command[0] == "nvidia-smi" and len(command) > 1 and "--query-gpu" in command[1]:
            stdout = (
                "0, NVIDIA GeForce RTX 3060, 12288, 8.6, 591.59\n"
                "1, NVIDIA GeForce GTX 1660 SUPER, 6144, 7.5, 591.59\n"
            )
        elif command == ["nvidia-smi"]:
            stdout = "NVIDIA-SMI 591.59 CUDA UMD Version: 13.0"
        else:
            stdout = ""
        return subprocess.CompletedProcess(command, 0, stdout, "")

    # Keep this unit independent from whichever torch happens to be installed.
    monkeypatch.setattr(
        HardwareProfiler,
        "_torch_info",
        lambda self: (None, None, None, []),
    )
    profile = HardwareProfiler(runner=runner).profile()
    assert [gpu.name for gpu in profile.gpus] == [
        "NVIDIA GeForce RTX 3060",
        "NVIDIA GeForce GTX 1660 SUPER",
    ]
    assert profile.primary_gpu is not None
    assert profile.primary_gpu.index == 0
    assert not profile.gpus[1].selected_for_inference
    assert profile.cuda_driver_api_version == "13.0"
