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
from hayate.backends.minimax_h3.vae_tiling import validate_vae_tile_size
from hayate.errors import GenerationPreflightError
from hayate.loaders.base import LoaderStatus
from hayate.loaders.factory import LoaderFactory
from hayate.models import ModelRegistry
from hayate.models.types import ModelRole


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

        if request.steps != 50:
            warnings.append("non-default step count changes the model's quality/speed operating point")
        if request.blocks_to_swap < 40:
            warnings.append("fewer than 40 swapped blocks is unlikely to fit an RTX 3060 12GB")
        if request.task == "ref2va":
            warnings.append("the initial W4A8 target is FL2VA; ref2va needs its matching transformer layout")
        if request.easycache:
            warnings.append("EasyCache trades a small amount of numerical fidelity for generation speed")
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
        log_path = plan.request.output.with_suffix(plan.request.output.suffix + ".hayate.log")
        manifest_path = plan.request.output.with_suffix(plan.request.output.suffix + ".hayate.json")
        env = os.environ.copy()
        env.update(plan.environment)
        started = time.perf_counter()
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
        duration = time.perf_counter() - started
        runtime_metrics = None
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("HAYATE_RUNTIME_METRICS "):
                try:
                    runtime_metrics = json.loads(line.removeprefix("HAYATE_RUNTIME_METRICS "))
                except json.JSONDecodeError:
                    runtime_metrics = None
        manifest_path.write_text(
            json.dumps(
                {
                    "plan": plan.to_dict(),
                    "returncode": process.returncode,
                    "duration_seconds": duration,
                    "log_path": str(log_path),
                    "runtime_metrics": runtime_metrics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return GenerationResult(
            process.returncode,
            duration,
            plan.command,
            log_path,
            plan.request.output,
            runtime_metrics,
        )
