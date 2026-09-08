"""Small, dependency-free helpers for selecting physical NVIDIA GPUs.

The MiniMax-H3 launcher intentionally runs with a single CUDA device per
process.  A process can still be assigned to any physical adapter by masking
the process with ``CUDA_VISIBLE_DEVICES=<UUID>`` and keeping the upstream
device argument at ``cuda:0``.  This module keeps that policy in one place so
the WebUI, CLI and cross-process lease use the same identity.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import Callable, Sequence

GPU_UUID_RE = re.compile(r"^GPU-[0-9A-Fa-f-]+$", re.IGNORECASE)
GPU_INDEX_RE = re.compile(r"^(?:cuda:)?[0-9]+$", re.IGNORECASE)
AUTO_GPU = "auto"


@dataclass(frozen=True)
class GPUDevice:
    """A physical adapter as reported by ``nvidia-smi``.

    ``index`` is useful for display and legacy command lines only.  UUID is
    preferred for process masking because CUDA visible ordinals can be
    reordered by the caller or by another launcher.
    """

    index: int
    name: str
    vram_total_bytes: int
    compute_capability: str | None = None
    driver_version: str | None = None
    uuid: str | None = None

    @property
    def identity(self) -> str:
        return self.uuid or f"index:{self.index}"

    @property
    def visible_id(self) -> str:
        return self.uuid or str(self.index)

    @property
    def label(self) -> str:
        return f"GPU {self.index} · {self.name} · {self.vram_total_bytes / 1024**3:.1f} GB"

    @property
    def h3_eligible(self) -> bool:
        """Whether the native HAYATE W4A8 path can target this adapter.

        The packed H3 kernel requires an Ampere-or-newer CUDA capability
        (SM 8.0+).  Unknown capability is treated as ineligible for automatic
        assignment so a detected legacy adapter is never selected silently.
        """

        if not self.compute_capability:
            return False
        try:
            major = int(str(self.compute_capability).split(".", 1)[0])
        except (TypeError, ValueError):
            return False
        return major >= 8

    @property
    def auto_assignable(self) -> bool:
        """Whether this adapter has a stable identity for automatic routing.

        An index-only report can still be used with an explicit index, but it
        must not enter a multi-process auto scheduler because CUDA visibility
        order can change between launches.
        """

        return self.h3_eligible and bool(self.uuid)

    @property
    def eligibility_reason(self) -> str:
        if self.h3_eligible and not self.uuid:
            return "GPU UUID unavailable; automatic assignment is disabled (select the physical index)"
        if self.h3_eligible:
            return "SM 8.0+"
        if not self.compute_capability:
            return "compute capability unavailable"
        return f"SM {self.compute_capability} is below the W4A8 SM 8.0 requirement"

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "vram_total_bytes": self.vram_total_bytes,
            "compute_capability": self.compute_capability,
            "driver_version": self.driver_version,
            "uuid": self.uuid,
            "identity": self.identity,
            "h3_eligible": self.h3_eligible,
            "auto_assignable": self.auto_assignable,
            "eligibility_reason": self.eligibility_reason,
        }


def normalize_gpu_selector(value: str | int | None) -> str:
    """Normalize the public ``auto``/index/UUID selector syntax.

    Arbitrary environment values are deliberately not accepted.  Callers can
    select ``auto``, a physical index (``0`` or ``cuda:0``), or an NVIDIA UUID.
    """

    if value is None:
        return AUTO_GPU
    text = str(value).strip()
    if not text or text.lower() == AUTO_GPU:
        return AUTO_GPU
    if GPU_INDEX_RE.fullmatch(text):
        return str(int(text.split(":")[-1]))
    if GPU_UUID_RE.fullmatch(text):
        return text.upper()
    raise ValueError("gpu selector must be auto, a numeric GPU index, or an NVIDIA GPU UUID")


def _run_nvidia_query(
    runner: Callable[..., subprocess.CompletedProcess[str]],
    query: Sequence[str],
) -> subprocess.CompletedProcess[str] | None:
    try:
        return runner(
            list(query),
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _memory_bytes(value: str) -> int:
    return int(float(value.strip()) * 1024 * 1024)


def discover_gpu_devices(
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> tuple[GPUDevice, ...]:
    """Return physical NVIDIA adapters without importing PyTorch.

    Older drivers or test doubles may omit UUID/compute capability.  The
    parser accepts both the current six-column query and the historical
    five-column response, retaining a usable index-based fallback.
    """

    run = runner or subprocess.run
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ]
    result = _run_nvidia_query(run, command)
    if result is None or result.returncode != 0:
        fallback = _run_nvidia_query(
            run,
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
        )
        if fallback is None or fallback.returncode != 0:
            return ()
        lines = fallback.stdout.splitlines()
    else:
        lines = result.stdout.splitlines()

    devices: list[GPUDevice] = []
    for line in lines:
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        try:
            if len(parts) >= 6:
                index = int(parts[0])
                uuid = parts[1].upper() if GPU_UUID_RE.fullmatch(parts[1]) else None
                name = parts[2]
                memory = _memory_bytes(parts[3])
                compute = parts[4] or None
                driver = parts[5] or None
            elif len(parts) == 5:
                # Compatibility with the profiler's pre-UUID query and old
                # nvidia-smi fixtures: index,name,memory,compute,driver.
                index = int(parts[0])
                uuid = None
                name = parts[1]
                memory = _memory_bytes(parts[2])
                compute = parts[3] or None
                driver = parts[4] or None
            elif len(parts) == 4:
                index = int(parts[0])
                uuid = None
                name = parts[1]
                memory = _memory_bytes(parts[2])
                compute = None
                driver = parts[3] or None
            else:
                continue
        except (TypeError, ValueError):
            continue
        devices.append(
            GPUDevice(
                index=index,
                name=name,
                vram_total_bytes=memory,
                compute_capability=compute,
                driver_version=driver,
                uuid=uuid,
            )
        )
    return tuple(sorted(devices, key=lambda item: item.index))


def allowed_gpu_devices(
    devices: Sequence[GPUDevice],
    visible_devices: str | None = None,
) -> tuple[GPUDevice, ...]:
    """Filter physical devices by an existing CUDA visibility allow-list."""

    raw = os.environ.get("CUDA_VISIBLE_DEVICES") if visible_devices is None else visible_devices
    if raw is None:
        return tuple(devices)
    tokens = tuple(token.strip().upper() for token in raw.split(",") if token.strip())
    if not tokens or "-1" in tokens:
        return ()
    allowed: list[GPUDevice] = []
    for device in devices:
        if str(device.index).upper() in tokens or (device.uuid or "").upper() in tokens:
            allowed.append(device)
    return tuple(allowed)


def eligible_gpu_devices(devices: Sequence[GPUDevice]) -> tuple[GPUDevice, ...]:
    """Return adapters safe for automatic MiniMax-H3 assignment."""

    return tuple(device for device in devices if device.auto_assignable)


def resolve_gpu_selector(
    selector: str | int | None,
    devices: Sequence[GPUDevice],
) -> GPUDevice | None:
    """Resolve a selector to one physical adapter; ``auto`` returns ``None``."""

    normalized = normalize_gpu_selector(selector)
    if normalized == AUTO_GPU:
        return None
    for device in devices:
        if normalized == str(device.index) or normalized.casefold() == (device.uuid or "").casefold():
            return device
    return None


def choose_auto_gpu(devices: Sequence[GPUDevice]) -> GPUDevice | None:
    """Pick a deterministic candidate; the lease arbitrates actual availability."""

    candidates = eligible_gpu_devices(allowed_gpu_devices(devices))
    return candidates[0] if candidates else None


__all__ = [
    "AUTO_GPU",
    "GPUDevice",
    "allowed_gpu_devices",
    "choose_auto_gpu",
    "discover_gpu_devices",
    "eligible_gpu_devices",
    "normalize_gpu_selector",
    "resolve_gpu_selector",
]
