from __future__ import annotations

import json
import queue
import sys
from io import StringIO

from hayate.webui.native_pool import RESULT_PREFIX, WarmNativePool


class _QueueTextStream:
    def __init__(self):
        self.lines: queue.Queue[str | None] = queue.Queue()

    def __iter__(self):
        while True:
            line = self.lines.get()
            if line is None:
                return
            yield line


class _FakeProcess:
    _next_pid = 100_000_000

    def __init__(self):
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.stdout = _QueueTextStream()
        self.returncode = None
        self.stdin = _FakeStdin(self)
        self.stdout.lines.put('HAYATE_WORKER_READY {"pid":1}\n')

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self):
        self.returncode = -9
        self.stdout.lines.put(None)


class _FakeStdin:
    def __init__(self, process):
        self.process = process

    def write(self, text):
        request = json.loads(text)
        if request.get("command") == "shutdown":
            self.process.returncode = 0
            self.process.stdout.lines.put("HAYATE_WORKER_STOPPED\n")
            self.process.stdout.lines.put(None)
        else:
            self.process.stdout.lines.put("generation log\n")
            self.process.stdout.lines.put(
                RESULT_PREFIX
                + json.dumps({"returncode": 0, "runtime_metrics": {"cuda_peak_allocated_bytes": 42}})
                + "\n"
            )
        return len(text)

    @staticmethod
    def flush():
        return None


def test_native_pool_reuses_worker_and_restarts_for_runtime_changes(tmp_path, monkeypatch):
    processes = []

    def fake_popen(*_args, **_kwargs):
        process = _FakeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr("hayate.webui.native_pool.subprocess.Popen", fake_popen)
    pool = WarmNativePool(startup_timeout=1)
    command = [
        sys.executable,
        "-m",
        "hayate.backends.minimax_h3.entrypoint",
        "--upstream",
        str(tmp_path / "upstream"),
        "--",
        "--prompt",
        "first",
    ]
    environment = {"HAYATE_GPU_UUID": "GPU-test", "HAYATE_EASYCACHE": "1"}
    startup_log = StringIO()

    session = pool.ensure("GPU-test", command, environment, tmp_path, startup_log)
    seen = []
    returncode, metrics = pool.run_job(
        session, "job-1", ["--prompt", "first"], StringIO(), seen.append
    )
    reused_log = StringIO()
    reused = pool.ensure("GPU-test", command, environment, tmp_path, reused_log)

    assert returncode == 0
    assert metrics == {"cuda_peak_allocated_bytes": 42}
    assert seen == ["generation log\n"]
    assert reused is session
    assert len(processes) == 1
    assert "HAYATE_WORKER_REUSED" in reused_log.getvalue()

    changed_environment = {**environment, "HAYATE_EASYCACHE": "0"}
    replacement = pool.ensure(
        "GPU-test", command, changed_environment, tmp_path, StringIO()
    )
    assert replacement is not session
    assert len(processes) == 2

    pool.close()
    assert not pool.has_live("GPU-test")
