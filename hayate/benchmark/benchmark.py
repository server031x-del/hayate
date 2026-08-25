from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from hayate.memory.manager import MemoryManager


@dataclass(frozen=True)
class BenchmarkStageResult:
    name: str
    duration_seconds: float
    ram_before_bytes: int
    ram_after_bytes: int
    ram_peak_bytes: int
    vram_before_bytes: int
    vram_after_bytes: int
    vram_peak_bytes: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class Benchmark:
    def __init__(self, memory: MemoryManager | None = None):
        self.memory = memory or MemoryManager()
        self.started_at = datetime.now(timezone.utc)
        self.stages: list[BenchmarkStageResult] = []
        self._active: str | None = None

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if self._active is not None:
            raise RuntimeError(f"benchmark stage {self._active!r} is already active")
        self._active = name
        before = self.memory.snapshot()
        started = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - started
            after = self.memory.snapshot()
            self.stages.append(
                BenchmarkStageResult(
                    name=name,
                    duration_seconds=duration,
                    ram_before_bytes=before.process_rss_bytes,
                    ram_after_bytes=after.process_rss_bytes,
                    ram_peak_bytes=max(before.process_peak_rss_bytes, after.process_peak_rss_bytes),
                    vram_before_bytes=before.current_vram_bytes,
                    vram_after_bytes=after.current_vram_bytes,
                    vram_peak_bytes=max(before.peak_vram_bytes, after.peak_vram_bytes),
                )
            )
            self._active = None

    def to_dict(self) -> dict:
        peak_ram = max((stage.ram_peak_bytes for stage in self.stages), default=0)
        peak_vram = max((stage.vram_peak_bytes for stage in self.stages), default=0)
        return {
            "schema_version": 1,
            "started_at": self.started_at.isoformat(),
            "stages": [stage.to_dict() for stage in self.stages],
            "peak_ram_bytes": peak_ram,
            "peak_vram_bytes": peak_vram,
            "quality_impact": "NOT_MEASURED_IN_V0.1_INSPECTION",
        }

    def save_json(self, directory: str | Path = "benchmarks") -> Path:
        target_dir = Path(directory).expanduser().resolve(strict=False)
        target_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        target = target_dir / f"run_{stamp}.json"
        temporary = target.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
        return target

