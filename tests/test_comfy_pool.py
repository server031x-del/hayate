from __future__ import annotations

import io
import sys
import time
from pathlib import Path

from hayate.backends.minimax_h3.generation import GenerationPlan, GenerationRequest
from hayate.runtime.gpu_devices import GPUDevice
from hayate.webui import comfy_pool
from hayate.webui.jobs import JobManager, JobStore


def test_warm_pool_reuses_server_and_holds_runtime_lease(tmp_path, monkeypatch):
    processes = []
    leases = []

    class FakeProcess:
        def __init__(self, args, **kwargs):
            self.args = args
            self.environment = kwargs["env"]
            self.pid = 1000 + len(processes)
            self.returncode = None
            processes.append(self)

        def poll(self):
            return self.returncode

        def wait(self, **kwargs):
            return self.returncode

        def kill(self):
            self.returncode = -9

    class FakeLease:
        def __init__(self, **kwargs):
            self.released = False
            leases.append(self)

        def acquire(self):
            return True

        def release(self):
            self.released = True

    class FakeParent:
        def __init__(self, pid):
            self.process = next(item for item in processes if item.pid == pid)

        def children(self, **kwargs):
            return []

        def terminate(self):
            self.process.returncode = 0

    monkeypatch.setattr(comfy_pool.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(comfy_pool, "GPULease", FakeLease)
    monkeypatch.setattr(comfy_pool.psutil, "Process", FakeParent)
    monkeypatch.setattr(comfy_pool.psutil, "wait_procs", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(comfy_pool.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"{}"))

    pool = comfy_pool.WarmComfyPool()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    first = pool.ensure("GPU-test-a", runtime, "python", tmp_path, {"CUDA_VISIBLE_DEVICES": "GPU-test-a"})
    second = pool.ensure("GPU-test-a", runtime, "python", tmp_path, {"CUDA_VISIBLE_DEVICES": "GPU-test-a"})
    assert len(processes) == len(leases) == 1
    assert first["HAYATE_COMFY_REUSED"] == "0"
    assert second["HAYATE_COMFY_REUSED"] == "1"
    assert first["HAYATE_COMFY_BASE_URL"] == second["HAYATE_COMFY_BASE_URL"]
    assert leases[0].released is False
    assert processes[0].environment["CUDA_VISIBLE_DEVICES"] == "GPU-test-a"
    assert processes[0].args[processes[0].args.index("--listen") + 1] == "127.0.0.1"
    processes[0].returncode = 1
    assert pool.has_live("GPU-test-a") is False
    assert leases[0].released is True
    third = pool.ensure("GPU-test-a", runtime, "python", tmp_path, {"CUDA_VISIBLE_DEVICES": "GPU-test-a"})
    assert third["HAYATE_COMFY_REUSED"] == "0"
    assert len(processes) == len(leases) == 2
    pool.discard("GPU-test-a")
    assert processes[1].poll() is not None
    assert leases[1].released is True
    pool.close()


def test_webui_reuses_warm_pool_for_two_fast_h3_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("HAYATE_GPU_LEASE_PATH", str(tmp_path / "gpu.lock"))
    device = GPUDevice(0, "A100", 40 * 1024**3, "8.0", uuid="GPU-test-a")

    class FakePool:
        def __init__(self):
            self.alive = False
            self.reused = []
            self.discarded = []
            self.closed = False

        def has_live(self, gpu_id):
            return self.alive

        def ensure(self, gpu_id, runtime, python, root, environment):
            self.reused.append(self.alive)
            self.alive = True
            return {"HAYATE_COMFY_BASE_URL": "http://127.0.0.1:8189",
                    "HAYATE_COMFY_REUSED": "1" if self.reused[-1] else "0"}

        def discard(self, gpu_id):
            self.discarded.append(gpu_id)
            self.alive = False

        def close(self):
            self.closed = True

    def plan(output: Path, fail=False):
        code = "raise SystemExit(1)" if fail else (
            "import os; from pathlib import Path; "
            f"Path({str(output.with_suffix('.reuse'))!r}).write_text(os.environ['HAYATE_COMFY_REUSED']); "
            f"Path({str(output)!r}).write_bytes(b'fake-mp4')"
        )
        return GenerationPlan(
            GenerationRequest("car", tmp_path, output),
            (sys.executable, "-c", code, "--runtime", str(tmp_path / "runtime")),
            {}, None, (), (), working_directory=tmp_path, backend="comfy_fasth3",
        )

    pool = FakePool()
    store = JobStore(tmp_path / "jobs.sqlite")
    manager = JobManager(store, gpu_discovery=lambda: (device,))
    manager._comfy_pool = pool
    try:
        for name in ("first", "second"):
            output = tmp_path / f"{name}.mp4"
            job = manager.submit(plan(output), {"prompt": "car"})
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                current = store.get(job["id"])
                if current and current["status"] in {"succeeded", "failed"}:
                    break
                time.sleep(0.02)
            assert current["status"] == "succeeded", current
        assert (tmp_path / "first.reuse").read_text() == "0"
        assert (tmp_path / "second.reuse").read_text() == "1"
        assert pool.reused == [False, True]
        assert pool.discarded == []
        failed = manager.submit(plan(tmp_path / "failed.mp4", fail=True), {"prompt": "car"})
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            current = store.get(failed["id"])
            if current and current["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.02)
        assert current["status"] == "failed", current
        assert pool.discarded == ["GPU-test-a"]
    finally:
        manager.shutdown()
    assert pool.closed is True
