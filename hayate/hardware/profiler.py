from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Sequence

import psutil


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    vram_total_bytes: int
    compute_capability: str | None
    driver_version: str | None
    selected_for_inference: bool
    # Physical identity is stable across CUDA visible-device reordering.  It
    # is optional for older drivers and mocked nvidia-smi output.
    uuid: str | None = None

    @property
    def h3_eligible(self) -> bool:
        if not self.compute_capability:
            return False
        try:
            return int(str(self.compute_capability).split(".", 1)[0]) >= 8
        except (TypeError, ValueError):
            return False

    @property
    def auto_assignable(self) -> bool:
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

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "vram_total_bytes": self.vram_total_bytes,
            "compute_capability": self.compute_capability,
            "driver_version": self.driver_version,
            "selected_for_inference": self.selected_for_inference,
            "uuid": self.uuid,
            "h3_eligible": self.h3_eligible,
            "auto_assignable": self.auto_assignable,
            "eligibility_reason": self.eligibility_reason,
        }


@dataclass(frozen=True)
class HardwareProfile:
    cpu: str
    logical_cpu_count: int
    physical_cpu_count: int | None
    system_ram_bytes: int
    cuda_available: bool
    cuda_driver_api_version: str | None
    pytorch_version: str | None
    pytorch_cuda_version: str | None
    torch_cuda_available: bool | None
    gpus: tuple[GPUInfo, ...]
    warnings: tuple[str, ...] = ()

    @property
    def primary_gpu(self) -> GPUInfo | None:
        return next((gpu for gpu in self.gpus if gpu.selected_for_inference), None)

    @property
    def inference_gpus(self) -> tuple[GPUInfo, ...]:
        """All detected adapters available for explicit GPU assignment.

        ``selected_for_inference`` remains the v0.1 primary marker for
        backwards-compatible reports; it no longer means that other adapters
        are unusable.  The scheduler uses the physical UUID when present.
        """

        return self.gpus

    @property
    def eligible_gpus(self) -> tuple[GPUInfo, ...]:
        return tuple(gpu for gpu in self.gpus if gpu.h3_eligible)

    @property
    def auto_assignable_gpus(self) -> tuple[GPUInfo, ...]:
        return tuple(gpu for gpu in self.gpus if gpu.auto_assignable)

    def to_dict(self) -> dict:
        return {
            "cpu": self.cpu,
            "logical_cpu_count": self.logical_cpu_count,
            "physical_cpu_count": self.physical_cpu_count,
            "system_ram_bytes": self.system_ram_bytes,
            "cuda_available": self.cuda_available,
            "cuda_driver_api_version": self.cuda_driver_api_version,
            "pytorch_version": self.pytorch_version,
            "pytorch_cuda_version": self.pytorch_cuda_version,
            "torch_cuda_available": self.torch_cuda_available,
            "gpus": [gpu.to_dict() for gpu in self.gpus],
            "gpu_count": len(self.gpus),
            "multi_gpu_supported": len(self.gpus) > 1,
            "eligible_gpu_count": len(self.eligible_gpus),
            "auto_assignable_gpu_count": len(self.auto_assignable_gpus),
            "warnings": list(self.warnings),
        }


class HardwareProfiler:
    """Profile hardware without importing torch into the HAYATE process."""

    def __init__(self, runner: Callable[..., subprocess.CompletedProcess[str]] | None = None):
        self._runner = runner or subprocess.run

    @staticmethod
    def _cpu_name() -> str:
        if os.name == "nt":
            try:
                import winreg

                key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                    value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                    if value.strip():
                        return value.strip()
            except OSError:
                pass
        return platform.processor().strip() or platform.uname().processor or "Unknown CPU"

    def _run(self, command: Sequence[str], timeout: float = 8.0) -> subprocess.CompletedProcess[str] | None:
        try:
            return self._runner(
                list(command), capture_output=True, text=True, timeout=timeout, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None

    def _nvidia_gpus(self) -> tuple[tuple[GPUInfo, ...], str | None, list[str]]:
        warnings: list[str] = []
        query = [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ]
        result = self._run(query)
        if result is None or result.returncode != 0:
            fallback = self._run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ]
            )
            if fallback is None or fallback.returncode != 0:
                return (), None, warnings
            warnings.append("nvidia-smi did not expose compute capability")
            rows = []
            for line in fallback.stdout.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) == 4:
                    rows.append(", ".join((parts[0], parts[1], parts[2], "", parts[3])))
            result = subprocess.CompletedProcess(fallback.args, 0, "\n".join(rows), fallback.stderr)
        gpus: list[GPUInfo] = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(",")]
            # Current nvidia-smi query: index,uuid,name,memory,compute,driver.
            # Accept the old five-column shape used by older drivers/tests.
            if len(parts) not in {5, 6}:
                warnings.append(f"could not parse nvidia-smi GPU row: {line}")
                continue
            try:
                index = int(parts[0])
                if len(parts) == 6:
                    uuid = parts[1].upper() if parts[1].upper().startswith("GPU-") else None
                    name = parts[2]
                    memory_bytes = int(float(parts[3]) * 1024 * 1024)
                    compute_capability = parts[4] or None
                    driver_version = parts[5] or None
                else:
                    uuid = None
                    name = parts[1]
                    memory_bytes = int(float(parts[2]) * 1024 * 1024)
                    compute_capability = parts[3] or None
                    driver_version = parts[4] or None
            except ValueError:
                warnings.append(f"invalid nvidia-smi numeric field: {line}")
                continue
            gpus.append(
                GPUInfo(
                    index=index,
                    name=name,
                    vram_total_bytes=memory_bytes,
                    compute_capability=compute_capability,
                    driver_version=driver_version,
                    selected_for_inference=index == 0,
                    uuid=uuid,
                )
            )
        full = self._run(["nvidia-smi"])
        driver_cuda = None
        if full is not None:
            # Newer Windows drivers label this "CUDA UMD Version" while the
            # classic Linux/Windows table uses "CUDA Version".
            match = re.search(r"CUDA(?: UMD)? Version:\s*([0-9.]+)", full.stdout)
            if match:
                driver_cuda = match.group(1)
        return tuple(sorted(gpus, key=lambda gpu: gpu.index)), driver_cuda, warnings

    def _torch_info(self) -> tuple[str | None, str | None, bool | None, list[str]]:
        warnings: list[str] = []
        try:
            installed_version = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            return None, None, None, warnings
        script = (
            "import json, torch; "
            "print(json.dumps({'version': torch.__version__, "
            "'cuda': torch.version.cuda, 'available': torch.cuda.is_available()}))"
        )
        result = self._run([sys.executable, "-c", script], timeout=20)
        if result is None or result.returncode != 0:
            warnings.append("PyTorch is installed but its CUDA probe failed in an isolated process")
            return installed_version, None, None, warnings
        try:
            payload = json.loads(result.stdout.strip().splitlines()[-1])
            return (
                str(payload.get("version") or installed_version),
                payload.get("cuda"),
                bool(payload.get("available")),
                warnings,
            )
        except (json.JSONDecodeError, IndexError):
            warnings.append("PyTorch CUDA probe returned invalid output")
            return installed_version, None, None, warnings

    def profile(self) -> HardwareProfile:
        gpus, driver_cuda, warnings = self._nvidia_gpus()
        torch_version, torch_cuda, torch_available, torch_warnings = self._torch_info()
        warnings.extend(torch_warnings)
        if len(gpus) > 1:
            warnings.append(
                "複数GPUを検出しました。生成ごとに物理GPUを割り当てられます（既定の並列数は1）"
            )
        if gpus and not any(gpu.h3_eligible for gpu in gpus):
            warnings.append(
                "H3 W4A8を自動割り当てできるSM 8.0以上のGPUが見つかりません"
            )
        elif gpus and not any(gpu.auto_assignable for gpu in gpus):
            warnings.append(
                "H3対応GPUのUUIDを取得できないため、自動GPU割り当ては無効です。物理indexを明示してください"
            )
        return HardwareProfile(
            cpu=self._cpu_name(),
            logical_cpu_count=psutil.cpu_count(logical=True) or 0,
            physical_cpu_count=psutil.cpu_count(logical=False),
            system_ram_bytes=psutil.virtual_memory().total,
            cuda_available=bool(gpus),
            cuda_driver_api_version=driver_cuda,
            pytorch_version=torch_version,
            pytorch_cuda_version=torch_cuda,
            torch_cuda_available=torch_available,
            gpus=gpus,
            warnings=tuple(warnings),
        )
