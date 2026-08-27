from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from hayate.backends.minimax_h3.upstream import H3UpstreamAdapter, UpstreamValidation
from hayate.backends.minimax_h3.pdd import (
    PDDCheckpointConfig,
    validate_adaln_affine,
    validate_pruned_adaln_coordinates,
)
from hayate.backends.minimax_h3.vae_tiling import validate_vae_tile_size
from hayate.errors import GenerationPreflightError
from hayate.loaders.base import LoaderStatus
from hayate.loaders.factory import LoaderFactory
from hayate.models import ModelRegistry
from hayate.models.types import ModelRole
from hayate.runtime.gpu_lease import GPULease


def _aligned_h3_frames(frames: int) -> int:
    """Return the upstream video-VAE frame count (17*n + 5) for a request."""
    return frames + ((5 - frames) % 17)


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    checkpoint_dir: Path
    output: Path
    task: str = "auto"
    image_path: Path | None = None
    last_image_path: Path | None = None
    references: tuple[Path, ...] = ()
    height: int | None = None
    width: int | None = None
    frames: int = 124
    steps: int = 50
    seed: int = 20260825
    blocks_to_swap: int = 49
    activation_chunk_rows: int = 32768
    prompt_cache: Path | None = None
    easycache: bool = False
    easycache_threshold: float = 0.2
    easycache_start: float = 0.15
    easycache_end: float = 0.95
    easycache_max_consecutive_skips: int = 2
    vae_tile_size: int = 256
    attention_backend: str = "sdpa"
    pdd_checkpoint: Path | None = None
    pdd_adaln_affine: Path | None = None


@dataclass(frozen=True)
class GenerationPlan:
    request: GenerationRequest
    command: tuple[str, ...]
    environment: dict[str, str]
    upstream: UpstreamValidation
    issues: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def executable(self) -> bool:
        return self.upstream.valid and not self.issues

    def to_dict(self) -> dict:
        return {
            "executable": self.executable,
            "command": list(self.command),
            "environment": self.environment,
            "upstream": self.upstream.to_dict(),
            "issues": list(self.issues),
            "warnings": list(self.warnings),
            "request": {
                "prompt": self.request.prompt,
                "checkpoint_dir": str(self.request.checkpoint_dir),
                "output": str(self.request.output),
                "task": self.request.task,
                "height": self.request.height,
                "width": self.request.width,
                "frames": self.request.frames,
                "steps": self.request.steps,
                "seed": self.request.seed,
                "blocks_to_swap": self.request.blocks_to_swap,
                "activation_chunk_rows": self.request.activation_chunk_rows,
                "prompt_cache": (
                    str(self.request.prompt_cache.resolve(strict=False))
                    if self.request.prompt_cache is not None
                    else None
                ),
                "easycache": self.request.easycache,
                "easycache_threshold": self.request.easycache_threshold,
                "easycache_start": self.request.easycache_start,
                "easycache_end": self.request.easycache_end,
                "easycache_max_consecutive_skips": self.request.easycache_max_consecutive_skips,
                "vae_tile_size": self.request.vae_tile_size,
                "attention_backend": self.request.attention_backend,
                "pdd_checkpoint": (
                    str(self.request.pdd_checkpoint.resolve(strict=False))
                    if self.request.pdd_checkpoint is not None
                    else None
                ),
                "pdd_adaln_affine": (
                    str(self.request.pdd_adaln_affine.resolve(strict=False))
                    if self.request.pdd_adaln_affine is not None
                    else None
                ),
            },
        }


@dataclass(frozen=True)
class GenerationResult:
    returncode: int
    duration_seconds: float
    command: tuple[str, ...]
    log_path: Path
    output: Path
    runtime_metrics: dict | None = None


def generation_artifact_paths(output: Path) -> tuple[Path, Path]:
    return (
        output.with_suffix(output.suffix + ".hayate.log"),
        output.with_suffix(output.suffix + ".hayate.json"),
    )


def runtime_metrics_from_log(log_path: Path) -> dict | None:
    runtime_metrics = None
    if not log_path.is_file():
        return None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("HAYATE_RUNTIME_METRICS "):
            try:
                runtime_metrics = json.loads(line.removeprefix("HAYATE_RUNTIME_METRICS "))
            except json.JSONDecodeError:
                runtime_metrics = None
    return runtime_metrics


def write_generation_manifest(
    plan: GenerationPlan,
    *,
    returncode: int,
    duration_seconds: float,
    log_path: Path,
    runtime_metrics: dict | None,
    job_id: str | None = None,
) -> Path:
    _, manifest_path = generation_artifact_paths(plan.request.output)
    payload = {
        "schema_version": 1,
        "plan": plan.to_dict(),
        "returncode": returncode,
        "duration_seconds": duration_seconds,
        "log_path": str(log_path),
        "runtime_metrics": runtime_metrics,
    }
    if job_id is not None:
        payload["job_id"] = job_id
    temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest_path


class ExternalH3GenerationBackend:
    """Build and execute maybleMyers/h3's generation CLI without copying it."""

    def __init__(
        self,
        checkout: str | Path,
        registry: ModelRegistry,
        *,
        python: str | Path = sys.executable,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ):
        self.upstream = H3UpstreamAdapter(checkout)
        self.registry = registry
        self.python = Path(python).expanduser().resolve(strict=False)
        self._runner = runner

    @staticmethod
    def _checkpoint_requirements(checkpoint_dir: Path, task: str) -> tuple[Path, ...]:
        transformer = "transformer_ref" if task == "ref2va" else "transformer"
        return (
            checkpoint_dir / transformer / "config.json",
            checkpoint_dir / "vae" / "config.json",
            checkpoint_dir / "audio_vae" / "config.json",
            checkpoint_dir / "text_encoder" / "config.json",
            checkpoint_dir / "scheduler" / "scheduler_config.json",
            checkpoint_dir / "audio_scheduler" / "scheduler_config.json",
            checkpoint_dir / "tokenizer",
            checkpoint_dir / "processor",
        )

    def _model_paths(self) -> dict[ModelRole, Path]:
        return {
            role: self.registry.get("minimax_h3", role).path
            for role in (
                ModelRole.TRANSFORMER,
                ModelRole.TEXT_ENCODER,
                ModelRole.VIDEO_VAE,
                ModelRole.AUDIO_VAE,
            )
        }

    def _probe_python_module(self, module: str, timeout: float = 30.0) -> tuple[bool, str]:
        try:
            result = self._runner(
                [str(self.python), "-c", f"import {module}"],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return False, str(exc)
        if result.returncode:
            detail = result.stderr.strip().splitlines()
            return False, detail[-1] if detail else f"exit code {result.returncode}"
        return True, "available"

    @staticmethod
    def _validate_request(request: GenerationRequest) -> list[str]:
        issues: list[str] = []
        if not request.prompt.strip():
            issues.append("prompt must not be empty")
        if request.task not in {"auto", "t2va", "fl2va", "ref2va"}:
            issues.append(f"unsupported task: {request.task}")
        if (request.height is None) != (request.width is None):
            issues.append("height and width must be supplied together")
        if request.height is not None and (
            request.height <= 0
            or request.width is None
            or request.width <= 0
            or request.height % 32
            or request.width % 32
        ):
            issues.append("height and width must be positive multiples of 32")
        if request.frames < 5:
            issues.append("frames must be at least 5")
        if request.steps < 2:
            issues.append("steps must be at least 2 (the terminal zero is included)")
        if not 0 <= request.blocks_to_swap <= 49:
            issues.append("blocks_to_swap must be in the range 0..49")
        if request.activation_chunk_rows < 0:
            issues.append("activation_chunk_rows must be non-negative")
        if request.easycache_threshold < 0:
            issues.append("easycache_threshold must be non-negative")
        if not 0 <= request.easycache_start < request.easycache_end <= 1:
            issues.append("easycache range must satisfy 0 <= start < end <= 1")
        if request.easycache_max_consecutive_skips < 1:
            issues.append("easycache_max_consecutive_skips must be at least one")
        try:
            validate_vae_tile_size(request.vae_tile_size, label="vae_tile_size")
        except ValueError as exc:
            issues.append(str(exc))
        if request.attention_backend not in {"sdpa", "sageattn"}:
            issues.append(f"unsupported attention backend: {request.attention_backend}")
        if request.pdd_checkpoint is not None and request.easycache:
            issues.append("PDD and EasyCache are mutually exclusive; select only one acceleration mode")
        if request.pdd_checkpoint is not None and request.task == "ref2va":
            issues.append("the FL2VA PDD checkpoint cannot be used with ref2va")
        if (
            request.pdd_checkpoint is not None
            and request.attention_backend == "sageattn"
            and _aligned_h3_frames(request.frames) < 243
        ):
            issues.append(
                "PDD with SageAttention is disabled below 243 frames on the validated "
                "RTX 3060 path because short clips can produce non-finite latents; select SDPA"
            )
        for label, path in (
            ("first image", request.image_path),
            ("last image", request.last_image_path),
        ):
            if path is not None and not path.is_file():
                issues.append(f"{label} does not exist: {path}")
        for path in request.references:
            if not path.is_file():
                issues.append(f"reference does not exist: {path}")
        return issues

    def plan(self, request: GenerationRequest) -> GenerationPlan:
        checkout_validation = self.upstream.validate(require_audited_commit=True)
        issues = self._validate_request(request)
        warnings: list[str] = []
        if not self.python.is_file():
            issues.append(f"Python interpreter does not exist: {self.python}")
        elif request.attention_backend == "sageattn":
            available, reason = self._probe_python_module("sageattention")
            if not available:
                issues.append(f"SageAttention is unavailable in {self.python}: {reason}")
        checkpoint_dir = request.checkpoint_dir.expanduser().resolve(strict=False)
        if not checkpoint_dir.is_dir():
            issues.append(f"checkpoint directory does not exist: {checkpoint_dir}")
        else:
            selected_task = "t2va" if request.task == "auto" else request.task
            for path in self._checkpoint_requirements(checkpoint_dir, selected_task):
                if not path.exists():
                    issues.append(f"checkpoint support file is missing: {path}")

        try:
            model_paths = self._model_paths()
        except Exception as exc:
            issues.append(f"model registry is incomplete: {exc}")
            model_paths = {}
        for role, path in model_paths.items():
            if not path.is_file():
                issues.append(f"{role.value} model is missing: {path}")
                continue
            spec = self.registry.get("minimax_h3", role)
            try:
                validation = LoaderFactory.create(spec).validate()
            except Exception as exc:
                issues.append(f"{role.value} loader validation failed: {exc}")
                continue
            if validation.status is not LoaderStatus.SUPPORTED:
                issues.append(
                    f"{role.value} execution loader is {validation.status.value}: "
                    f"{validation.reason}"
                )

        if request.pdd_checkpoint is not None:
            try:
                pdd_config = PDDCheckpointConfig.inspect(request.pdd_checkpoint)
            except Exception as exc:
                issues.append(f"PDD checkpoint validation failed: {exc}")
            else:
                required_points = pdd_config.nfe + 1
                if request.steps != required_points:
                    issues.append(
                        f"PDD {pdd_config.nfe}-NFE requires {required_points} scheduler points, "
                        f"got {request.steps}"
                    )
                transformer_path = model_paths.get(ModelRole.TRANSFORMER)
                if transformer_path is not None and transformer_path.is_file():
                    try:
                        from hayate.models.safetensors_header import read_safetensors_header

                        transformer_names = {
                            tensor.name for tensor in read_safetensors_header(transformer_path).tensors
                        }
                        if "adaln_t_table" in transformer_names:
                            if request.pdd_adaln_affine is None:
                                issues.append(
                                    "PDD with the pruned transformer requires pdd_adaln_affine"
                                )
                            elif not request.pdd_adaln_affine.is_file():
                                issues.append(
                                    f"PDD AdaLN affine map does not exist: {request.pdd_adaln_affine}"
                                )
                            else:
                                try:
                                    validate_pruned_adaln_coordinates(transformer_path)
                                    validate_adaln_affine(request.pdd_adaln_affine)
                                except Exception as exc:
                                    issues.append(f"PDD AdaLN projection validation failed: {exc}")
                    except Exception as exc:
                        issues.append(f"cannot inspect transformer for PDD compatibility: {exc}")

        if request.steps != 50:
            warnings.append("non-default step count changes the model's quality/speed operating point")
        if request.blocks_to_swap < 40:
            warnings.append("fewer than 40 swapped blocks is unlikely to fit an RTX 3060 12GB")
        if request.task == "ref2va":
            warnings.append("the initial W4A8 target is FL2VA; ref2va needs its matching transformer layout")
        if request.easycache:
            warnings.append("EasyCache trades a small amount of numerical fidelity for generation speed")
        if request.pdd_checkpoint is not None:
            warnings.append(
                "PDD uses a distilled 8-evaluation trajectory; compare motion and prompt fidelity against Quality"
            )
        if request.attention_backend == "sageattn":
            warnings.append("SageAttention is approximate; compare quality against SDPA")

        command = [
            str(self.python),
            "-m",
            "hayate.backends.minimax_h3.entrypoint",
            "--upstream",
            str(self.upstream.checkout),
            "--",
            "--ckpt_dir",
            str(checkpoint_dir),
            "--task",
            request.task,
            "--prompt",
            request.prompt,
            "--video_length",
            str(request.frames),
            "--infer_steps",
            str(request.steps),
            "--seed",
            str(request.seed),
            "--device",
            "cuda:0",
            "--attn_mode",
            request.attention_backend,
            "--blocks_to_swap",
            str(request.blocks_to_swap),
            "--act_chunk_rows",
            str(request.activation_chunk_rows),
            "--text_encoder_gpu_layers",
            "0",
            "--text_encoder_stream",
            "--vae_tiling",
            "--save_path",
            str(request.output.parent.resolve(strict=False)),
            "--output_filename",
            str(request.output.resolve(strict=False)),
        ]
        role_flags = {
            ModelRole.TRANSFORMER: "--dit",
            ModelRole.TEXT_ENCODER: "--text_encoder",
            ModelRole.VIDEO_VAE: "--vae",
            ModelRole.AUDIO_VAE: "--audio_vae",
        }
        for role, flag in role_flags.items():
            if role in model_paths:
                command.extend((flag, str(model_paths[role])))
        if request.height is not None and request.width is not None:
            command.extend(("--video_size", str(request.height), str(request.width)))
        if request.frames < 96:
            command.append("--allow_short")
        if request.image_path is not None:
            command.extend(("--image_path", str(request.image_path.resolve(strict=False))))
        if request.last_image_path is not None:
            command.extend(("--last_image_path", str(request.last_image_path.resolve(strict=False))))
        for reference in request.references:
            command.extend(("--reference", str(reference.resolve(strict=False))))
        if request.prompt_cache is not None:
            command.extend(("--prompt_cache", str(request.prompt_cache.resolve(strict=False))))

        environment = (
            {} if os.name == "nt" else {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        )
        environment["HAYATE_VAE_TILE_SIZE"] = str(request.vae_tile_size)
        if request.easycache:
            environment.update(
                {
                    "HAYATE_EASYCACHE": "1",
                    "HAYATE_EASYCACHE_THRESHOLD": str(request.easycache_threshold),
                    "HAYATE_EASYCACHE_START": str(request.easycache_start),
                    "HAYATE_EASYCACHE_END": str(request.easycache_end),
                    "HAYATE_EASYCACHE_MAX_CONSECUTIVE_SKIPS": str(
                        request.easycache_max_consecutive_skips
                    ),
                }
            )
        if request.pdd_checkpoint is not None:
            environment["HAYATE_PDD_CHECKPOINT"] = str(
                request.pdd_checkpoint.expanduser().resolve(strict=False)
            )
            # Pageable adapter storage is the validated default. Pinning can
            # be enabled explicitly for a measured host with
            # HAYATE_PDD_PIN_LORA=1, but must not silently change allocator
            # behavior on consumer GPUs.
            environment["HAYATE_PDD_PIN_LORA"] = os.environ.get("HAYATE_PDD_PIN_LORA", "0")
            if request.pdd_adaln_affine is not None:
                environment["HAYATE_PDD_ADALN_AFFINE"] = str(
                    request.pdd_adaln_affine.expanduser().resolve(strict=False)
                )
        return GenerationPlan(
            request,
            tuple(command),
            environment,
            checkout_validation,
            tuple(dict.fromkeys(issues)),
            tuple(warnings),
        )

    def probe_cli(self, timeout: float = 60.0) -> tuple[bool, str]:
        validation = self.upstream.validate(require_audited_commit=True)
        if not validation.valid or not self.python.is_file():
            return False, "upstream checkout or Python interpreter is not ready"
        try:
            result = self._runner(
                [str(self.python), str(self.upstream.component_path("generation_cli")), "--help"],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
                cwd=str(self.upstream.checkout),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return False, str(exc)
        required_flags = ("--dit", "--text_encoder", "--vae", "--audio_vae", "--blocks_to_swap")
        missing = [flag for flag in required_flags if flag not in result.stdout]
        if result.returncode or missing:
            reason = result.stderr.strip() or f"generation CLI missing flags: {', '.join(missing)}"
            return False, reason
        return True, "upstream generation CLI contract is available"

    def execute(self, plan: GenerationPlan) -> GenerationResult:
        if not plan.executable:
            raise GenerationPreflightError("generation preflight failed: " + "; ".join(plan.issues))
        ok, reason = self.probe_cli()
        if not ok:
            raise GenerationPreflightError(f"upstream generation CLI probe failed: {reason}")
        plan.request.output.parent.mkdir(parents=True, exist_ok=True)
        log_path, _ = generation_artifact_paths(plan.request.output)
        env = os.environ.copy()
        env.update(plan.environment)
        env["PYTHONIOENCODING"] = "utf-8"
        started = time.perf_counter()
        lease = GPULease(
            owner={"pid": os.getpid(), "kind": "generation", "output": str(plan.request.output)}
        )
        if not lease.acquire():
            owner = lease.busy_owner() or {}
            raise GenerationPreflightError(
                f"GPU 0 is already in use by HAYATE (pid={owner.get('pid', '?')})"
            )
        try:
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.run(
                    list(plan.command),
                    cwd=str(self.upstream.checkout),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
        finally:
            lease.release()
        duration = time.perf_counter() - started
        runtime_metrics = runtime_metrics_from_log(log_path)
        write_generation_manifest(
            plan,
            returncode=process.returncode,
            duration_seconds=duration,
            log_path=log_path,
            runtime_metrics=runtime_metrics,
        )
        return GenerationResult(
            process.returncode,
            duration,
            plan.command,
            log_path,
            plan.request.output,
            runtime_metrics,
        )
