from __future__ import annotations

import io
import json
from types import SimpleNamespace

from hayate.backends.minimax_h3.entrypoint import (
    _collect_runtime_metrics,
    _persistent_worker_loop,
)


def test_cuda_metrics_failure_is_reported_without_raising():
    class Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def synchronize():
            raise RuntimeError("device failed")

    torch = SimpleNamespace(cuda=Cuda())
    memory = SimpleNamespace(rss=1, vms=2, peak_wset=3, private=4, peak_pagefile=5)
    psutil = SimpleNamespace(Process=lambda: SimpleNamespace(memory_info=lambda: memory))
    metrics = _collect_runtime_metrics(SimpleNamespace(), psutil, torch)
    assert metrics["process_peak_private_bytes"] == 5
    assert metrics["cuda_metrics_error"] == "RuntimeError: device failed"


def test_persistent_worker_protocol_ready_and_shutdown():
    input_stream = io.StringIO('{"command":"shutdown"}\n')
    output_stream = io.StringIO()

    result = _persistent_worker_loop(
        SimpleNamespace(), "generate.py", input_stream, output_stream
    )

    assert result == 0
    assert "HAYATE_WORKER_READY" in output_stream.getvalue()
    assert "HAYATE_WORKER_STOPPED" in output_stream.getvalue()


def test_persistent_worker_rejects_malformed_requests():
    input_stream = io.StringIO("not-json\n{\"command\":\"shutdown\"}\n")
    output_stream = io.StringIO()

    _persistent_worker_loop(SimpleNamespace(), "generate.py", input_stream, output_stream)
    lines = output_stream.getvalue().splitlines()
    result = json.loads(next(line.split(" ", 1)[1] for line in lines if line.startswith("HAYATE_WORKER_RESULT ")))

    assert result["returncode"] == 2
    assert result["error"] == "invalid worker request"
