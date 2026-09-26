from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from hayate.backends.minimax_h3.generation import (
    GenerationPlan,
    GenerationRequest,
    generation_artifact_paths,
)
from hayate.backends.minimax_h3.upstream import AUDITED_COMMIT, UpstreamValidation
from hayate.runtime.gpu_devices import GPUDevice
from hayate.webui.jobs import JobManager, JobStore


class _Lease:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


class _NativePool:
    def __init__(self, output: Path):
        self.output = output
        self.process = SimpleNamespace(pid=123456789, poll=lambda: None)
        self.ensure_calls = 0
        self.run_calls = 0
        self.discard_calls = 0
        self.closed = False

    def ensure(self, _gpu_id, _command, _environment, _cwd, startup_log):
        self.ensure_calls += 1
        startup_log.write('HAYATE_WORKER_READY {"pid":123456789}\n')
        return SimpleNamespace(process=self.process)

    def run_job(self, _session, _job_id, _argv, _log, on_line):
        self.run_calls += 1
        self.output.write_bytes(b"mock-video")
        on_line(
            'HAYATE_EVENT '
            + json.dumps(
                {
                    "schema": 1,
                    "type": "progress",
                    "progress": 64,
                    "stage": "動画生成",
                    "detail": "デノイズ",
                }
            )
            + "\n"
        )
        metrics = {"cuda_peak_allocated_bytes": 123}
        on_line("HAYATE_RUNTIME_METRICS " + json.dumps(metrics) + "\n")
        return 0, metrics

    def discard(self, _gpu_id, *, force=True):
        self.discard_calls += 1

    def close(self):
        self.closed = True


def test_a100_detail_job_runs_through_and_retains_native_worker(tmp_path):
    output = tmp_path / "warm.mp4"
    upstream = tmp_path / "h3"
    request = GenerationRequest(
        "test prompt",
        tmp_path,
        output,
        steps=20,
        keep_model_warm=True,
    )
    validation = UpstreamValidation(
        upstream, True, AUDITED_COMMIT, AUDITED_COMMIT, (), ()
    )
    plan = GenerationPlan(
        request,
        (
            sys.executable,
            "-m",
            "hayate.backends.minimax_h3.entrypoint",
            "--upstream",
            str(upstream),
            "--",
            "--prompt",
            request.prompt,
        ),
        {},
        validation,
        (),
        (),
    )
    store = JobStore(tmp_path / "jobs.sqlite3")
    manager = JobManager(store)
    fake_pool = _NativePool(output)
    manager._native_pool = fake_pool
    assigned_gpu = GPUDevice(0, "NVIDIA A100", 40 * 1024**3, "8.0", uuid="GPU-a100")
    lease = _Lease()
    manager._acquire_gpu_lease = lambda *_args: (lease, assigned_gpu, None)
    job_id = "native-warm-job"
    log_path, _manifest_path = generation_artifact_paths(output)
    store.create(
        {
            "id": job_id,
            "status": "queued",
            "source": "webui",
            "created_at": "2026-09-26T00:00:00+00:00",
            "started_at": None,
            "completed_at": None,
            "progress": 0.0,
            "stage": "待機中",
            "detail": "queued",
            "eta_seconds": None,
            "duration_seconds": None,
            "request": {"prompt": request.prompt, "profile": "a100_detail"},
            "plan": plan.to_dict(),
            "runtime_metrics": None,
            "output_path": str(output),
            "log_path": str(log_path),
            "error": None,
            "pid": None,
            "requested_gpu_selector": "auto",
            "assigned_gpu_uuid": None,
            "assigned_gpu_index": None,
            "assigned_gpu_name": None,
            "assigned_gpu_compute_capability": None,
            "assigned_at": None,
        }
    )
    manager._plans[job_id] = plan

    try:
        manager._run(job_id, plan)

        result = store.get(job_id)
        assert result is not None
        assert result["status"] == "succeeded"
        assert result["runtime_metrics"]["cuda_peak_allocated_bytes"] == 123
        assert output.read_bytes() == b"mock-video"
        assert fake_pool.ensure_calls == 1
        assert fake_pool.run_calls == 1
        assert fake_pool.discard_calls == 0
        assert lease.released
    finally:
        manager.shutdown()

    assert fake_pool.closed
