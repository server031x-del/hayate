"""Optional FastH3/VSA generation adapter.

This adapter is deliberately separate from :class:`ExternalH3GenerationBackend`.
It launches an operator-provided FastVideo installation and only accepts an
official FastVideo *directory* checkpoint.  The Kijai single-file conversion is
registered and inspected by :mod:`fasth3`, but is rejected here because the
FastVideo component loader cannot consume it directly.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

from hayate.backends.minimax_h3.fasth3 import (
    BLACKWELL_FASTVIDEO_CAPABILITIES,
    FAST_PROFILE_MIN_VRAM_BYTES,
    FASTH3_PROFILE_FAST,
    FASTH3_PROFILE_STRICT,
    fast_profile_issues,
    probe_fastvideo_runtime,
    validate_fastvideo_model_directory,
)
from hayate.backends.minimax_h3.generation import (
    GenerationPlan,
    GenerationRequest,
    GenerationResult,
    generation_artifact_paths,
    runtime_metrics_from_log,
    write_generation_manifest,
)
from hayate.backends.minimax_h3.runtime_command import (
    build_runtime_command,
    parse_runtime_spec,
    runtime_path,
)
from hayate.errors import GenerationPreflightError
from hayate.runtime.gpu_devices import (
    AUTO_GPU,
    allowed_gpu_devices,
    choose_auto_gpu,
    discover_gpu_devices,
    normalize_gpu_selector,
    resolve_gpu_selector,
)
from hayate.runtime.gpu_lease import GPULease


class FastH3GenerationBackend:
    """Build a HAYATE job around FastVideo's official FastH3 API example."""

    def __init__(
        self,
        model_directory: str | Path,
        *,
        python: str | Path = sys.executable,
        performance_profile: str = FASTH3_PROFILE_STRICT,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ):
        if performance_profile not in {FASTH3_PROFILE_STRICT, FASTH3_PROFILE_FAST}:
            raise ValueError(f"unknown FastH3 performance profile: {performance_profile}")
        self.model_directory = Path(model_directory).expanduser().resolve(strict=False)
        # Keep the descriptor as text: ``wsl://Ubuntu/...`` is intentionally
        # not a Windows Path.  Native paths are normalized by the command
        # helper at the point of execution.
        self.python = str(python).strip()
        self.performance_profile = performance_profile
        self._runner = runner

    @staticmethod
    def _validate_request(request: GenerationRequest) -> list[str]:
        issues: list[str] = []
        if not request.prompt.strip():
            issues.append("prompt must not be empty")
        if request.task not in {"auto", "t2va"}:
            issues.append("FastH3 VSA preview is currently validated for T2VA only")
        if request.image_path is not None or request.last_image_path is not None or request.references:
            issues.append("FastH3 VSA preview does not accept I2V/reference images")
        if request.height is not None and (
            request.height <= 0 or request.width is None or request.width <= 0
            or request.height % 32 or request.width % 32
        ):
            issues.append("height and width must be positive multiples of 32")
        if request.frames < 5:
            issues.append("frames must be at least 5")
        if request.steps != 5:
            issues.append("FastH3's trained schedule uses 5 sigma points (4 DiT forwards)")
        if request.easycache:
            issues.append("FastH3 VSA has its own distilled schedule; EasyCache is disabled")
        if request.pdd_checkpoint is not None:
            issues.append("FastH3 VSA cannot be combined with the H3 PDD adapter")
        try:
            normalize_gpu_selector(request.gpu_device)
        except ValueError as exc:
            issues.append(str(exc))
        return issues

    def _select_gpu(self, selector: str) -> tuple[dict[str, str], list[str]]:
        issues: list[str] = []
        environment: dict[str, str] = {}
        try:
            normalized = normalize_gpu_selector(selector)
        except ValueError as exc:
            return environment, [str(exc)]
        environment["HAYATE_GPU_SELECTOR"] = normalized
        environment["HAYATE_FASTH3_PROFILE"] = self.performance_profile
        if self.performance_profile == FASTH3_PROFILE_FAST:
            environment["HAYATE_FASTH3_REQUIRE_BLACKWELL"] = "1"
            environment["HAYATE_FASTH3_MIN_VRAM_BYTES"] = str(FAST_PROFILE_MIN_VRAM_BYTES)
        if normalized == AUTO_GPU:
            return environment, issues
        selected = resolve_gpu_selector(
            normalized,
            allowed_gpu_devices(discover_gpu_devices(self._runner)),
        )
        if selected is None:
            issues.append(f"selected GPU was not found: {selector}")
            return environment, issues
        if not selected.h3_eligible:
            issues.append(
                f"selected GPU is not eligible for VSA INT8 execution: "
                f"{selected.name} ({selected.eligibility_reason})"
            )
            return environment, issues
        if (
            self.performance_profile == FASTH3_PROFILE_FAST
            and str(selected.compute_capability) not in BLACKWELL_FASTVIDEO_CAPABILITIES
        ):
            issues.append(
                "FastH3最速プロファイルの選択GPUはBlackwell（compute capability 10.0/10.3）ではありません"
            )
            return environment, issues
        if (
            self.performance_profile == FASTH3_PROFILE_FAST
            and selected.vram_total_bytes < FAST_PROFILE_MIN_VRAM_BYTES
        ):
            issues.append(
                "FastH3最速プロファイルの選択GPUは80 GiB以上のVRAMを必要とします"
            )
            return environment, issues
        environment.update(
            CUDA_VISIBLE_DEVICES=selected.visible_id,
            HAYATE_GPU_UUID=selected.uuid or "",
            HAYATE_GPU_INDEX=str(selected.index),
        )
        return environment, issues

    def probe_cli(self, timeout: float = 30.0) -> tuple[bool, str]:
        runtime = probe_fastvideo_runtime(
            self.python,
            runner=self._runner,
            timeout=timeout,
        )
        if not runtime["available"]:
            return False, str(runtime["reason"])
        issues = validate_fastvideo_model_directory(
            self.model_directory,
            strict=(self.model_directory / "modular_model_index.json").is_file(),
        )
        if issues:
            return False, "; ".join(issues)
        return True, "FastVideo runtime and model directory are available"

    def plan(self, request: GenerationRequest) -> GenerationPlan:
        issues = self._validate_request(request)
        issues.extend(
            validate_fastvideo_model_directory(
                self.model_directory,
                strict=(self.model_directory / "modular_model_index.json").is_file(),
            )
        )
        runtime = probe_fastvideo_runtime(
            self.python,
            runner=self._runner,
        )
        if not runtime["available"]:
            issues.append(f"FastVideo runtime is unavailable: {runtime['reason']}")
        elif self.performance_profile == FASTH3_PROFILE_FAST:
            issues.extend(fast_profile_issues(runtime))
        environment, gpu_issues = self._select_gpu(request.gpu_device)
        issues.extend(gpu_issues)
        warnings = [
            "FastH3 VSA is an experimental alternate backend; compare against the standard H3 profile",
            "the linked Kijai ComfyUI single-file checkpoint is not accepted as a FastVideo directory",
        ]
        try:
            runtime_spec = parse_runtime_spec(self.python)
        except ValueError:
            runtime_spec = None
        raw_args = [
            "-m",
            "hayate.backends.minimax_h3.fasth3_entrypoint",
            "--model-path",
            runtime_path(self.model_directory, runtime_spec) if runtime_spec else str(self.model_directory),
            "--prompt",
            request.prompt,
            "--output",
            runtime_path(request.output.resolve(strict=False), runtime_spec)
            if runtime_spec
            else str(request.output.resolve(strict=False)),
            "--height",
            str(request.height or 768),
            "--width",
            str(request.width or 1344),
            "--num-frames",
            str(request.frames),
            "--steps",
            str(request.steps),
            "--seed",
            str(request.seed),
            "--num-gpus",
            "1",
        ]
        if self.performance_profile == FASTH3_PROFILE_FAST:
            # This is the official Blackwell performance recipe: tile-64
            # sm100a VSA, FA4, regional DiT compile, parallel VAE decode and
            # the non-parity H3 fusions.  Preflight rejects other GPUs so the
            # WebUI never silently runs a slower profile under this label.
            raw_args.extend(
                [
                    "--vsa-kernel",
                    "sm100a",
                    "--profile",
                    "all",
                    "--fa4",
                    "--inference-torch-compile",
                    "--parallel-vae",
                    "--no-low-memory",
                    "--replicated-dit",
                ]
            )
            warnings.extend(
                [
                    "Blackwell最速プロファイルはH3 fusionsとregional compileを有効にするため、strict経路と数値が一致しません",
                    "最速プロファイルは1ジョブ1 GPUで起動します。公式4 GPU測定値はWebUIの単一GPU実行値ではありません",
                ]
            )
        else:
            raw_args.extend(
                [
                    "--vsa-kernel",
                    "triton",
                    "--profile",
                    "strict",
                    "--no-fa4",
                    "--no-inference-torch-compile",
                    "--no-parallel-vae",
                    # A 12GB consumer GPU cannot keep the 35B DiT resident.
                    # The layerwise path is the supported single-GPU
                    # low-memory mode.
                    "--low-memory",
                ]
            )
            warnings.append(
                "VSA Triton strict経路を使用します。sm100a/FA4はBlackwell最速プロファイルでのみ有効です"
            )
        if environment.get("CUDA_VISIBLE_DEVICES"):
            environment["HAYATE_GPU_RUNTIME_LEASE"] = "1"
        environment.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        environment.setdefault("PYTHONIOENCODING", "utf-8")
        command, _ = build_runtime_command(
            self.python,
            raw_args,
            environment=environment,
            cwd=self.model_directory,
            project_root=Path(__file__).resolve().parents[3],
        )
        return GenerationPlan(
            request=request,
            command=tuple(command),
            environment=environment,
            upstream=None,
            issues=tuple(dict.fromkeys(issues)),
            warnings=tuple(dict.fromkeys(warnings)),
            # WSL receives the mapped ``--cd`` from build_runtime_command; the
            # host subprocess still needs a native cwd.
            working_directory=(
                self.model_directory
                if runtime_spec is None or not runtime_spec.is_wsl
                else Path(__file__).resolve().parents[3]
            ),
            backend="fastvideo_vsa",
        )

    def execute(self, plan: GenerationPlan):
        if not plan.executable:
            raise GenerationPreflightError(
                "FastH3 generation preflight failed: " + "; ".join(plan.issues)
            )
        plan.request.output.parent.mkdir(parents=True, exist_ok=True)
        log_path, _ = generation_artifact_paths(plan.request.output)
        import time

        started = time.perf_counter()
        environment = dict(plan.environment)
        environment["PYTHONIOENCODING"] = "utf-8"
        inherited = {**os.environ, **environment}
        selected_gpu = None
        gpu_id = inherited.get("HAYATE_GPU_UUID") or None
        if gpu_id is None and inherited.get("HAYATE_GPU_INDEX"):
            gpu_id = f"index:{inherited['HAYATE_GPU_INDEX']}"
        devices = allowed_gpu_devices(
            discover_gpu_devices(self._runner),
            visible_devices=inherited.get("CUDA_VISIBLE_DEVICES"),
        )
        if gpu_id is not None:
            try:
                selected_gpu = resolve_gpu_selector(
                    inherited.get("HAYATE_GPU_UUID")
                    or inherited.get("HAYATE_GPU_INDEX")
                    or gpu_id,
                    devices,
                )
            except ValueError:
                selected_gpu = None
        if gpu_id is None and plan.request.gpu_device != AUTO_GPU:
            try:
                selected_gpu = resolve_gpu_selector(plan.request.gpu_device, devices)
            except ValueError:
                selected_gpu = None
        if gpu_id is None and selected_gpu is None and plan.request.gpu_device == AUTO_GPU:
            selected_gpu = choose_auto_gpu(devices)
        if selected_gpu is not None:
            gpu_id = selected_gpu.identity
            inherited["CUDA_VISIBLE_DEVICES"] = selected_gpu.visible_id
            inherited["HAYATE_GPU_UUID"] = selected_gpu.uuid or ""
            inherited["HAYATE_GPU_INDEX"] = str(selected_gpu.index)
            inherited["HAYATE_GPU_RUNTIME_LEASE"] = "1"

        lease = GPULease(
            gpu_id=gpu_id,
            namespace="scheduler",
            owner={
                "pid": os.getpid(),
                "kind": "fasth3-generation",
                "output": str(plan.request.output),
                "gpu_uuid": inherited.get("HAYATE_GPU_UUID") or None,
                "gpu_index": inherited.get("HAYATE_GPU_INDEX"),
            },
        )
        if not lease.acquire():
            owner = lease.busy_owner() or {}
            raise GenerationPreflightError(
                f"選択したGPUはHAYATEで使用中です (pid={owner.get('pid', '?')})"
            )
        runtime_probe = None
        if gpu_id is not None:
            runtime_probe = GPULease(
                gpu_id=gpu_id,
                namespace="runtime",
                owner={"pid": os.getpid(), "kind": "fasth3-runtime-probe"},
            )
            if not runtime_probe.acquire():
                lease.release()
                owner = runtime_probe.busy_owner() or {}
                raise GenerationPreflightError(
                    f"選択したGPUの実行ロックが残っています (pid={owner.get('pid', '?')})"
                )
            runtime_probe.release()

        command = list(plan.command)
        try:
            # A WSL command carries its allow-listed environment as argv to
            # wsl.exe, so an auto-selected GPU must be inserted after the
            # direct CLI resolves it.  WebUI jobs are already assigned by the
            # parent scheduler before they spawn and do not call this method.
            runtime_spec = parse_runtime_spec(self.python)
            if runtime_spec.is_wsl:
                interpreter_index = command.index(str(runtime_spec.executable))
                raw_args = command[interpreter_index + 1 :]
                command, _ = build_runtime_command(
                    runtime_spec,
                    raw_args,
                    environment=environment | {
                        key: value
                        for key, value in inherited.items()
                        if key.startswith("HAYATE_") or key.startswith("CUDA_")
                    },
                    cwd=self.model_directory,
                    project_root=Path(__file__).resolve().parents[3],
                )
            with log_path.open("w", encoding="utf-8") as log:
                result = self._runner(
                    command,
                    cwd=str(plan.working_directory or self.model_directory),
                    env=inherited,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
        finally:
            lease.release()

        runtime_metrics = runtime_metrics_from_log(log_path)
        if selected_gpu is not None:
            runtime_metrics = dict(runtime_metrics or {})
            runtime_metrics["gpu"] = selected_gpu.to_dict()
        duration = time.perf_counter() - started
        write_generation_manifest(
            plan,
            returncode=result.returncode,
            duration_seconds=duration,
            log_path=log_path,
            runtime_metrics=runtime_metrics,
            gpu_assignment=selected_gpu.to_dict() if selected_gpu is not None else None,
        )
        return GenerationResult(
            result.returncode,
            duration,
            tuple(command),
            log_path,
            plan.request.output,
            runtime_metrics,
        )
