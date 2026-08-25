from __future__ import annotations

import json
import subprocess

from hayate.benchmark import Benchmark
from hayate.memory import MemoryManager


def test_memory_snapshot_and_reset_peak():
    def runner(command, **kwargs):
        stdout = "0, NVIDIA GeForce RTX 3060, 1000, 11288, 12288\n"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    manager = MemoryManager(runner=runner)
    first = manager.snapshot()
    reset = manager.reset_peak()
    assert first.process_rss_bytes > 0
    assert first.system_available_bytes > 0
    assert first.gpus[0].used_bytes == 1000 * 1024 * 1024
    assert reset.peak_vram_bytes == reset.current_vram_bytes


def test_benchmark_timer_saves_json(tmp_path):
    def runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, "", "no gpu")

    benchmark = Benchmark(MemoryManager(runner=runner))
    with benchmark.stage("model_inspection"):
        sum(range(100))
    output = benchmark.save_json(tmp_path)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["stages"][0]["name"] == "model_inspection"
    assert payload["stages"][0]["duration_seconds"] >= 0
    assert payload["quality_impact"] == "NOT_MEASURED_IN_V0.1_INSPECTION"

