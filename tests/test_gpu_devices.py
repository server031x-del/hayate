from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from hayate.backends.minimax_h3.generation import GenerationPlan, GenerationRequest
from hayate.backends.minimax_h3.upstream import AUDITED_COMMIT, UpstreamValidation
from hayate.runtime.gpu_devices import (
    GPUDevice,
    allowed_gpu_devices,
    discover_gpu_devices,
    eligible_gpu_devices,
    normalize_gpu_selector,
    resolve_gpu_selector,
)
from hayate.runtime.gpu_lease import GPULease
from hayate.webui.jobs import FINAL_STATUSES, JobManager, JobStore


def test_discover_gpu_devices_keeps_uuid_and_marks_sm75_ineligible():
    def runner(command, **kwargs):
        if command[0] == "nvidia-smi" and any(
            item.startswith("--query-gpu") for item in command
        ):
            return subprocess.CompletedProcess(
                command,
                0,
                "0, GPU-AAA, RTX, 12288, 8.6, 610.88\n"
                "1, N/A, Legacy, 6144, 7.5, 610.88\n",
                "",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    devices = discover_gpu_devices(runner)
    assert devices[0].uuid == "GPU-AAA"
    assert devices[0].h3_eligible is True
    assert devices[1].uuid is None
    assert devices[1].h3_eligible is False
    assert [device.index for device in eligible_gpu_devices(devices)] == [0]


def test_gpu_selector_normalization_and_visibility_allowlist():
    devices = (
        GPUDevice(0, "A", 12, "8.6", uuid="GPU-A"),
        GPUDevice(1, "B", 12, "8.9", uuid="GPU-B"),
    )
    assert normalize_gpu_selector("cuda:01") == "1"
    assert normalize_gpu_selector("gpu-b") == "GPU-B"
    assert [device.index for device in allowed_gpu_devices(devices, "GPU-B")] == [1]
    assert resolve_gpu_selector("1", devices) == devices[1]


def test_uuid_scoped_leases_allow_different_adapters(tmp_path, monkeypatch):
    monkeypatch.delenv("HAYATE_GPU_LEASE_PATH", raising=False)
    first = GPULease(gpu_id="GPU-test-a", namespace="scheduler")
    second = GPULease(gpu_id="GPU-test-b", namespace="scheduler")
    assert first.path != second.path
    assert first.acquire() is True
    assert second.acquire() is True
    second.release()
    first.release()


def test_job_manager_can_assign_two_auto_jobs_to_two_uuid_gpus(tmp_path, monkeypatch):
    monkeypatch.delenv("HAYATE_GPU_LEASE_PATH", raising=False)
    devices = (
        GPUDevice(0, "A", 12, "8.6", uuid="GPU-TEST-A"),
        GPUDevice(1, "B", 12, "8.9", uuid="GPU-TEST-B"),
    )

    def fake_plan(output: Path) -> GenerationPlan:
        code = (
            "import os,time; from pathlib import Path; "
            f"Path({str(output)!r}).write_text(os.environ.get('CUDA_VISIBLE_DEVICES','')); "
            "time.sleep(0.25); "
            f"Path({str(output)!r}).write_bytes(b'fake-mp4')"
        )
        validation = UpstreamValidation(
            tmp_path, True, AUDITED_COMMIT, AUDITED_COMMIT, (), ()
        )
        return GenerationPlan(
            GenerationRequest("test", tmp_path, output),
            (sys.executable, "-c", code),
            {},
            validation,
            (),
            (),
        )

    store = JobStore(tmp_path / "multi-gpu.sqlite3")
    manager = JobManager(store, worker_count=2, gpu_discovery=lambda: devices)
    try:
        first = manager.submit(fake_plan(tmp_path / "first.mp4"), {"prompt": "first"})
        second = manager.submit(fake_plan(tmp_path / "second.mp4"), {"prompt": "second"})
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            current = [store.get(first["id"]), store.get(second["id"])]
            if all(job and job["status"] in FINAL_STATUSES for job in current):
                break
            time.sleep(0.05)
        assert current[0] is not None and current[1] is not None
        assert [job["status"] for job in current] == ["succeeded", "succeeded"]
        assert {job["assigned_gpu_uuid"] for job in current} == {
            "GPU-TEST-A",
            "GPU-TEST-B",
        }
    finally:
        manager.shutdown()
