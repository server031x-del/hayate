from __future__ import annotations

from types import SimpleNamespace

from hayate.backends.minimax_h3.entrypoint import _collect_runtime_metrics


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
