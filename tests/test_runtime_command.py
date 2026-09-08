from __future__ import annotations

import subprocess
from types import SimpleNamespace

from hayate.backends.minimax_h3.fasth3 import probe_fastvideo_runtime
from hayate.backends.minimax_h3.runtime_command import (
    build_runtime_command,
    parse_runtime_spec,
    update_runtime_command_environment,
    windows_to_wsl_path,
)


def test_windows_path_maps_to_wsl_mount():
    assert windows_to_wsl_path(r"M:\Project\HAYATE\models\fastvideo") == "/mnt/m/Project/HAYATE/models/fastvideo"


def test_wsl_runtime_descriptor_builds_explicit_distribution_command(tmp_path):
    spec = parse_runtime_spec("wsl://Ubuntu/home/ano031/hayate-fasth3-venv/bin/python")
    command, cwd = build_runtime_command(
        spec,
        ["-c", "print(1)"],
        environment={"CUDA_VISIBLE_DEVICES": "0", "PYTHONIOENCODING": "utf-8"},
        cwd=tmp_path / "models",
        project_root=r"M:\Project\HAYATE",
    )

    assert command[:5] == [
        "wsl.exe",
        "--distribution",
        "Ubuntu",
        "--cd",
        windows_to_wsl_path(tmp_path / "models"),
    ]
    assert "/usr/bin/env" in command
    assert "CUDA_VISIBLE_DEVICES=0" in command
    assert "PYTHONIOENCODING=utf-8" in command
    assert "PYTHONPATH=/mnt/m/Project/HAYATE" in command
    assert command[-3:] == ["/home/ano031/hayate-fasth3-venv/bin/python", "-c", "print(1)"]
    assert cwd == windows_to_wsl_path(tmp_path / "models")


def test_wsl_probe_accepts_json_after_runtime_log(monkeypatch):
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout='INFO runtime initialized\n{"fastvideo":true,"api":true,"minimax_h3":true,"vsa_kernel":true,"cuda":true,"available":true}\n',
            stderr="",
        )

    result = probe_fastvideo_runtime(
        "wsl://Ubuntu/home/ano031/hayate-fasth3-venv/bin/python",
        runner=runner,
    )

    assert result["available"] is True
    assert calls[0][0][0] == "wsl.exe"
    assert calls[0][0][2] == "Ubuntu"


def test_wsl_probe_retries_once_after_cold_start_timeout():
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return SimpleNamespace(
            returncode=0,
            stdout='{"fastvideo":true,"api":true,"minimax_h3":true,"cuda":true,"available":true}\n',
            stderr="",
        )

    result = probe_fastvideo_runtime(
        "wsl://Ubuntu/home/ano031/hayate-fasth3-venv/bin/python",
        runner=runner,
    )

    assert result["available"] is True
    assert len(calls) == 2


def test_wsl_command_refreshes_scheduler_gpu_assignment(tmp_path):
    command, _ = build_runtime_command(
        "wsl://Ubuntu/home/ano031/hayate-fasth3-venv/bin/python",
        ["-m", "hayate.backends.minimax_h3.fasth3_entrypoint"],
        environment={"CUDA_VISIBLE_DEVICES": "0", "HAYATE_GPU_INDEX": "0"},
        cwd=tmp_path,
        project_root=r"M:\Project\HAYATE",
    )

    refreshed = update_runtime_command_environment(
        command,
        {
            "CUDA_VISIBLE_DEVICES": "GPU-222",
            "HAYATE_GPU_INDEX": "1",
            "HAYATE_GPU_UUID": "GPU-222",
            "PYTHONIOENCODING": "utf-8",
        },
        project_root=r"M:\Project\HAYATE",
    )

    assert "CUDA_VISIBLE_DEVICES=GPU-222" in refreshed
    assert "HAYATE_GPU_INDEX=1" in refreshed
    assert "HAYATE_GPU_UUID=GPU-222" in refreshed
    assert "CUDA_VISIBLE_DEVICES=0" not in refreshed
    assert refreshed[-3:] == [
        "/home/ano031/hayate-fasth3-venv/bin/python",
        "-m",
        "hayate.backends.minimax_h3.fasth3_entrypoint",
    ]
