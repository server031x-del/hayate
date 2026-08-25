from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import psutil


class MemoryTier(str, Enum):
    GPU = "GPU_VRAM"
    RAM = "SYSTEM_RAM"
    NVME = "NVME_FUTURE"


@dataclass(frozen=True)
class GPUMemorySnapshot:
    index: int
    name: str
    used_bytes: int
    free_bytes: int
    total_bytes: int
    sampled_peak_used_bytes: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass(frozen=True)
class MemorySnapshot:
    timestamp: float
    process_rss_bytes: int
    process_peak_rss_bytes: int
    system_used_bytes: int
    system_available_bytes: int
    system_total_bytes: int
    gpus: tuple[GPUMemorySnapshot, ...]
    vram_scope: str = "GPU_TOTAL_SAMPLED"

    @property
    def current_vram_bytes(self) -> int:
        return sum(gpu.used_bytes for gpu in self.gpus)

    @property
    def peak_vram_bytes(self) -> int:
        return sum(gpu.sampled_peak_used_bytes for gpu in self.gpus)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "process_rss_bytes": self.process_rss_bytes,
            "process_peak_rss_bytes": self.process_peak_rss_bytes,
            "system_used_bytes": self.system_used_bytes,
            "system_available_bytes": self.system_available_bytes,
            "system_total_bytes": self.system_total_bytes,
            "current_vram_bytes": self.current_vram_bytes,
            "peak_vram_bytes": self.peak_vram_bytes,
            "vram_scope": self.vram_scope,
            "gpus": [gpu.to_dict() for gpu in self.gpus],
        }


class MemoryManager:
    """Sample process RAM and total GPU VRAM with explicit peak semantics."""

    def __init__(self, runner: Callable[..., subprocess.CompletedProcess[str]] | None = None):
        self._runner = runner or subprocess.run
        self._process = psutil.Process()
        self._peak_rss = 0
        self._gpu_peaks: dict[int, int] = {}

    def _gpu_memory(self) -> list[tuple[int, str, int, int, int]]:
        try:
            result = self._runner(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.used,memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        if result.returncode != 0:
            return []
        rows = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 5:
                continue
            try:
                index = int(parts[0])
                used, free, total = (int(float(value) * 1024 * 1024) for value in parts[2:])
            except ValueError:
                continue
            rows.append((index, parts[1], used, free, total))
        return rows

    def snapshot(self) -> MemorySnapshot:
        process_info = self._process.memory_info()
        rss = process_info.rss
        os_peak = getattr(process_info, "peak_wset", rss)
        self._peak_rss = max(self._peak_rss, rss, os_peak)
        virtual = psutil.virtual_memory()
        gpu_snapshots = []
        for index, name, used, free, total in self._gpu_memory():
            peak = max(self._gpu_peaks.get(index, 0), used)
            self._gpu_peaks[index] = peak
            gpu_snapshots.append(GPUMemorySnapshot(index, name, used, free, total, peak))
        return MemorySnapshot(
            timestamp=time.time(),
            process_rss_bytes=rss,
            process_peak_rss_bytes=self._peak_rss,
            system_used_bytes=virtual.used,
            system_available_bytes=virtual.available,
            system_total_bytes=virtual.total,
            gpus=tuple(gpu_snapshots),
        )

    def reset_peak(self) -> MemorySnapshot:
        self._peak_rss = 0
        self._gpu_peaks.clear()
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                if torch.cuda.is_initialized():
                    torch.cuda.reset_peak_memory_stats()
            except (AttributeError, RuntimeError):
                pass
        return self.snapshot()

