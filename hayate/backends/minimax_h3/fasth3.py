"""FastH3/VSA checkpoint inspection and fail-closed preflight helpers.

The Kijai FastH3 artifact is a ComfyUI single-file conversion.  It is not the
same checkpoint contract as the maybleMyers/h3 W4A8 loader and it must not be
silently routed through that loader.  This module deliberately only inspects
the safetensors header and validates an optional FastVideo model directory;
the actual FastVideo runtime remains an optional integration.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from hayate.models.safetensors_header import read_safetensors_header
from hayate.backends.minimax_h3.runtime_command import (
    build_runtime_command,
    parse_runtime_spec,
    runtime_exists,
)

KIJAI_FASTH3_REPOSITORY = "Kijai/MiniMax-H3-experimental"
KIJAI_FASTH3_REVISION = "f4cac997f880e93cf6940af61ee8d58ef31ff7f7"
KIJAI_FASTH3_FILENAME = (
    "minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors"
)
KIJAI_FASTH3_SIZE_BYTES = 22_898_594_920
KIJAI_FASTH3_SHA256 = (
    "7221ae65d78780354d51e5048d29728d9f1f8fb9baf50b1dd3df85f5101413d"
)
FASTVIDEO_FASTH3_REPOSITORY = (
    "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"
)
# Pin the model snapshot used by the adapter.  The model card is mutable; a
# moving ``main`` ref would make a previously validated directory silently
# change underneath a running HAYATE installation.
FASTVIDEO_FASTH3_REVISION = "5ea076f35b84da4c3c82217112fa733d8eea2ae1"
FASTVIDEO_FASTH3_SOURCE_URL = (
    "https://huggingface.co/" + FASTVIDEO_FASTH3_REPOSITORY
)
# Older snapshots used this name.  The directory validator intentionally does
# not hard-code a repository name so either pinned snapshot can be configured.
FASTVIDEO_FASTH3_LEGACY_REPOSITORY = "FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2"
FASTVIDEO_METADATA_FILES = (
    "model_index.json",
    "modular_model_index.json",
    "fastvideo_inference.json",
)
FASTVIDEO_ATTENTION_BACKEND = "VIDEO_SPARSE_ATTN_H3"
FASTH3_PROFILE_STRICT = "strict"
FASTH3_PROFILE_FAST = "fast"
BLACKWELL_FASTVIDEO_CAPABILITIES = frozenset({"10.0", "10.3"})
FAST_PROFILE_MIN_VRAM_BYTES = 80 * 1024**3

# The preview is a self-contained Diffusers/Modular FastVideo snapshot.  Keep
# this contract in one place so the CLI and WebUI reject a transformer-only
# directory before FastVideo starts importing or allocating model weights.
FASTVIDEO_FASTH3_REQUIRED_FILES: dict[str, str] = {
    "modular_model_index.json": "modular pipeline manifest",
    "transformer/config.json": "transformer config",
    "transformer/diffusion_pytorch_model.safetensors.index.json": "transformer weight index",
    "text_encoder/config.json": "text encoder config",
    "text_encoder/model.safetensors.index.json": "text encoder weight index",
    "text_encoder/tokenizer.json": "text encoder tokenizer",
    "text_encoder/tokenizer_config.json": "text encoder tokenizer config",
    "tokenizer/tokenizer.json": "tokenizer",
    "tokenizer/tokenizer_config.json": "tokenizer config",
    "processor/preprocessor_config.json": "processor config",
    "processor/video_preprocessor_config.json": "video processor config",
    "processor/tokenizer.json": "processor tokenizer",
    "vae/config.json": "video VAE config",
    "vae/diffusion_pytorch_model.safetensors.index.json": "video VAE weight index",
    "audio_vae/config.json": "audio VAE config",
    "audio_vae/diffusion_pytorch_model.safetensors": "audio VAE weights",
    "scheduler/scheduler_config.json": "video scheduler config",
    "audio_scheduler/scheduler_config.json": "audio scheduler config",
}


@dataclass(frozen=True)
class FastH3CheckpointInfo:
    """Header-only facts about a FastH3/VSA checkpoint."""

    path: Path
    exists: bool
    file_size: int | None
    header_size: int | None
    tensor_count: int
    gate_layer_count: int
    comfy_quant_count: int
    int8_weight_count: int
    layout: str
    detected: bool
    issues: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    size_verified: bool | None = None
    sha256_verified: bool | None = None
    identity_verified: bool = False
    gate_scale_count: int = 0
    gate_quant_count: int = 0

    @property
    def is_kijai_fastvideo_single_file(self) -> bool:
        return self.detected and self.layout == "comfy_single_file_vsa"

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "exists": self.exists,
            "file_size": self.file_size,
            "header_size": self.header_size,
            "tensor_count": self.tensor_count,
            "gate_layer_count": self.gate_layer_count,
            "comfy_quant_count": self.comfy_quant_count,
            "int8_weight_count": self.int8_weight_count,
            "layout": self.layout,
            "detected": self.detected,
            "is_kijai_fastvideo_single_file": self.is_kijai_fastvideo_single_file,
            "size_verified": self.size_verified,
            "sha256_verified": self.sha256_verified,
            "identity_verified": self.identity_verified,
            "gate_scale_count": self.gate_scale_count,
            "gate_quant_count": self.gate_quant_count,
            "issues": list(self.issues),
            "warnings": list(self.warnings),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_fasth3_checkpoint(
    path: str | Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    verify_sha256: bool = False,
) -> FastH3CheckpointInfo:
    """Inspect a checkpoint without materializing any tensor payload.

    ``verify_sha256`` is opt-in because hashing the linked 22.9GB artifact is
    intentionally never performed during WebUI bootstrap.  The normal model
    setup service already performs a full hash after a user explicitly starts
    a download.
    """

    checkpoint = Path(path).expanduser().resolve(strict=False)
    if not checkpoint.is_file():
        return FastH3CheckpointInfo(
            checkpoint,
            False,
            None,
            None,
            0,
            0,
            0,
            0,
            "missing",
            False,
            (f"FastH3 checkpoint does not exist: {checkpoint}",),
        )

    file_size = checkpoint.stat().st_size
    issues: list[str] = []
    warnings: list[str] = []
    size_verified: bool | None = None
    sha256_verified: bool | None = None
    if expected_size is not None and file_size != expected_size:
        issues.append(
            f"checkpoint size mismatch: expected {expected_size} bytes, found {file_size}"
        )
        size_verified = False
    elif expected_size is not None:
        size_verified = True
    if verify_sha256 and expected_sha256:
        actual = _sha256(checkpoint)
        sha256_verified = actual.lower() == expected_sha256.lower()
        if actual.lower() != expected_sha256.lower():
            issues.append(
                f"checkpoint SHA-256 mismatch: expected {expected_sha256}, found {actual}"
            )

    try:
        header = read_safetensors_header(checkpoint)
    except Exception as exc:
        return FastH3CheckpointInfo(
            checkpoint,
            True,
            file_size,
            None,
            0,
            0,
            0,
            0,
            "invalid",
            False,
            (*issues, f"cannot inspect safetensors header: {exc}"),
        )

    names = tuple(tensor.name for tensor in header.tensors)
    lower_names = tuple(name.lower() for name in names)
    gate_names = {
        name
        for name in lower_names
        if name.endswith(".attn.to_gate_compress.weight")
    }
    gate_layers = {
        name.removesuffix(".attn.to_gate_compress.weight") for name in gate_names
    }
    gate_scale_names = {
        name
        for name in lower_names
        if name.endswith(".attn.to_gate_compress.weight_scale")
    }
    gate_quant_names = {
        name
        for name in lower_names
        if name.endswith(".attn.to_gate_compress.comfy_quant")
    }
    comfy_quant_count = sum(name.endswith(".comfy_quant") for name in lower_names)
    int8_weight_count = sum(
        tensor.dtype.upper() == "I8" and tensor.name.lower().endswith(".weight")
        for tensor in header.tensors
    )
    has_qkv = "blocks.0.attn.qkv_proj.weight" in lower_names
    has_out_proj = "blocks.0.attn.out_proj.weight" in lower_names
    has_vsa_gate = bool(gate_layers)
    looks_comfy = has_qkv and has_out_proj and comfy_quant_count > 0
    detected = has_vsa_gate and looks_comfy and int8_weight_count > 0
    layout = "comfy_single_file_vsa" if detected else "unknown"

    if not has_vsa_gate:
        issues.append("to_gate_compress weights were not found")
    if not looks_comfy:
        issues.append("the expected ComfyUI fused qkv/out_proj layout was not found")
    if not int8_weight_count:
        issues.append("no INT8 weight tensors were found")
    if detected and checkpoint.suffix.lower() == ".safetensors":
        warnings.append(
            "this is a ComfyUI single-file VSA checkpoint, not a mayble H3 W4A8 file"
        )

    identity_verified = bool(
        detected
        and (size_verified is not False)
        and (expected_sha256 is None or sha256_verified is True)
    )

    return FastH3CheckpointInfo(
        checkpoint,
        True,
        file_size,
        header.header_size,
        len(header.tensors),
        len(gate_layers),
        comfy_quant_count,
        int8_weight_count,
        layout,
        detected,
        tuple(dict.fromkeys(issues)),
        tuple(dict.fromkeys(warnings)),
        size_verified,
        sha256_verified,
        identity_verified,
        len(gate_scale_names),
        len(gate_quant_names),
    )


def inspect_kijai_fastvideo_checkpoint(
    path: str | Path, *, verify_sha256: bool = False
) -> FastH3CheckpointInfo:
    """Inspect the exact Kijai artifact contract used by the model catalog."""

    info = inspect_fasth3_checkpoint(
        path,
        expected_size=KIJAI_FASTH3_SIZE_BYTES,
        expected_sha256=KIJAI_FASTH3_SHA256,
        verify_sha256=verify_sha256,
    )
    issues = list(info.issues)
    # The published Kijai conversion has one gate group for each of the 50 H3
    # blocks.  Keep the generic header detector permissive for diagnostics, but
    # do not call a partial/truncated file an identity match.
    if info.detected and info.gate_layer_count != 50:
        issues.append(
            f"Kijai FastH3 expects 50 VSA gate groups; found {info.gate_layer_count}"
        )
    if info.detected and info.gate_scale_count != info.gate_layer_count:
        issues.append(
            "Kijai FastH3 gate weights are missing one or more weight_scale tensors"
        )
    if info.detected and info.gate_quant_count != info.gate_layer_count:
        issues.append(
            "Kijai FastH3 gate weights are missing one or more comfy_quant tensors"
        )
    return replace(
        info,
        issues=tuple(dict.fromkeys(issues)),
        identity_verified=bool(info.identity_verified and not issues),
    )


def validate_fastvideo_model_directory(
    path: str | Path, *, strict: bool = False
) -> tuple[str, ...]:
    """Return missing-file issues for an official FastVideo model directory.

    FastVideo's component loader requires a directory, unlike HAYATE's
    single-file mayble loader.  We check only stable contract files here so
    this remains compatible with different FastVideo snapshot revisions.
    """

    model_dir = Path(path).expanduser().resolve(strict=False)
    if not model_dir.is_dir():
        return (f"FastVideo model directory does not exist: {model_dir}",)
    issues: list[str] = []
    metadata = [name for name in FASTVIDEO_METADATA_FILES if (model_dir / name).is_file()]
    if not metadata:
        issues.append(
            "FastVideo model directory is missing one of "
            + ", ".join(FASTVIDEO_METADATA_FILES)
        )
    transformer = model_dir / "transformer"
    if not (transformer / "config.json").is_file():
        issues.append("FastVideo model directory is missing transformer/config.json")
    if not any(transformer.rglob("*.safetensors")):
        issues.append("FastVideo model directory has no transformer safetensors weights")
    # Metadata/config files are JSON contracts.  A malformed hand-created file
    # should be reported before FastVideo is started, while unknown optional
    # fields remain owned by the external runtime.
    for name in (*metadata, "transformer/config.json"):
        candidate = model_dir / name
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            issues.append(f"FastVideo metadata is not valid JSON ({name}): {exc}")
            continue
        if not isinstance(payload, dict):
            issues.append(f"FastVideo metadata must be a JSON object ({name})")
    if strict and not issues:
        # MiniMaxH3BasePipeline declares these modules as mandatory for T2VA.
        # Checking the complete local snapshot contract prevents a seemingly
        # valid transformer-only directory from failing much later during
        # VideoGenerator.from_config().  The release currently pins this exact
        # modular manifest; older experimental snapshots can still be checked
        # in non-strict mode for diagnostics.
        for relative, label in FASTVIDEO_FASTH3_REQUIRED_FILES.items():
            candidate = model_dir / relative
            if not candidate.is_file():
                issues.append(f"FastVideo snapshot is missing {label}: {relative}")

        # Every JSON file that controls component construction must parse.  Do
        # not validate arbitrary fields here; those belong to the external
        # FastVideo package and may evolve without changing the filesystem
        # contract.
        json_files = {
            relative
            for relative in FASTVIDEO_FASTH3_REQUIRED_FILES
            if relative.endswith(".json") and (model_dir / relative).is_file()
        }
        for relative in sorted(json_files):
            candidate = model_dir / relative
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                issues.append(f"FastVideo snapshot JSON is invalid ({relative}): {exc}")
                continue
            if not isinstance(payload, dict):
                issues.append(f"FastVideo snapshot JSON must be an object ({relative})")

        for relative, label in (
            ("text_encoder", "text encoder"),
            ("vae", "video VAE"),
        ):
            directory = model_dir / relative
            if directory.is_dir() and not any(directory.glob("*.safetensors")):
                issues.append(f"FastVideo snapshot has no {label} safetensors weights")

        for relative, label in (
            ("transformer/diffusion_pytorch_model.safetensors.index.json", "transformer"),
            ("text_encoder/model.safetensors.index.json", "text encoder"),
            ("vae/diffusion_pytorch_model.safetensors.index.json", "video VAE"),
        ):
            index_path = model_dir / relative
            if not index_path.is_file():
                continue
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
                issues.append(f"FastVideo {label} weight index has no weight_map: {relative}")
                continue
            # A stale index is worse than a missing one: FastVideo will report
            # an opaque tensor-key error after loading several GiB.  Resolve
            # each shard path against the model root and report all missing
            # names before the process is started.
            shard_names = {str(value) for value in index["weight_map"].values()}
            for shard in sorted(shard_names):
                shard_relative = Path(shard)
                if shard_relative.is_absolute() or ".." in shard_relative.parts:
                    issues.append(
                        f"FastVideo {label} weight index contains an unsafe shard path: {shard}"
                    )
                    continue
                shard_path = model_dir / shard_relative
                if not shard_path.is_file():
                    shard_path = model_dir / Path(relative).parent / shard_relative
                if not shard or not shard_path.is_file():
                    issues.append(f"FastVideo {label} index references missing shard: {shard}")

        modular_manifest = model_dir / "modular_model_index.json"
        if modular_manifest.is_file():
            try:
                manifest = json.loads(modular_manifest.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                manifest = {}
            if isinstance(manifest, dict):
                class_name = manifest.get("_class_name")
                if class_name not in {None, "MiniMaxH3ModularPipeline"}:
                    issues.append(
                        "FastVideo snapshot has an unexpected pipeline class: "
                        + str(class_name)
                    )
    return tuple(dict.fromkeys(issues))


def probe_fastvideo_runtime(
    python: str | Path = sys.executable,
    *,
    runner=subprocess.run,
    timeout: float = 20.0,
) -> dict[str, object]:
    """Probe the optional FastVideo package without importing it in WebUI."""

    try:
        spec = parse_runtime_spec(python)
    except ValueError as exc:
        return {"python": str(python), "available": False, "reason": str(exc)}
    interpreter = spec.executable
    if not runtime_exists(spec):
        return {
            "python": spec.label,
            "available": False,
            "reason": f"Python interpreter does not exist: {interpreter}",
        }
    code = """
import importlib.metadata
import importlib.util
import json
def has_spec(name):
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False
checks = {
    "fastvideo": has_spec("fastvideo"),
    "api": has_spec("fastvideo.api"),
    "minimax_h3": has_spec("fastvideo.models.dits.minimax_h3"),
    "vsa_kernel": has_spec("fastvideo_kernel"),
    "vsa_sm100a": False,
    "fa4": has_spec("flash_attn.cute"),
}
try:
    from fastvideo_kernel import block_sparse_attn_sm100a
    checks["vsa_sm100a"] = bool(getattr(block_sparse_attn_sm100a, "_HAS_VSA_SM100A", False))
except Exception as exc:
    checks["vsa_sm100a_reason"] = str(exc)
try:
    checks["version"] = importlib.metadata.version("fastvideo")
except importlib.metadata.PackageNotFoundError:
    checks["version"] = None
checks["cuda"] = False
checks["gpu_capabilities"] = []
checks["gpu_names"] = []
checks["gpu_memory_bytes"] = []
checks["system_memory_bytes"] = None
try:
    import torch
    checks["torch"] = True
    checks["cuda"] = bool(torch.cuda.is_available())
    if checks["cuda"]:
        checks["gpu_capabilities"] = [
            ".".join(str(part) for part in torch.cuda.get_device_capability(index))
            for index in range(torch.cuda.device_count())
        ]
        checks["gpu_names"] = [
            str(torch.cuda.get_device_name(index))
            for index in range(torch.cuda.device_count())
        ]
        checks["gpu_memory_bytes"] = [
            int(torch.cuda.get_device_properties(index).total_memory)
            for index in range(torch.cuda.device_count())
        ]
except Exception as exc:
    checks["torch"] = False
    checks["torch_reason"] = str(exc)
try:
    import psutil
    checks["system_memory_bytes"] = int(psutil.virtual_memory().total)
except Exception:
    try:
        import os
        checks["system_memory_bytes"] = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:
        pass
checks["available"] = all(checks[name] for name in ("fastvideo", "api", "minimax_h3", "cuda"))
print(json.dumps(checks))
    """
    # The first WSL probe may compile Triton's tiny CUDA helper.  Allow a
    # one-time retry for that cold-start path without making native
    # missing-runtime checks slower.  Force UTF-8 so Japanese/WSL diagnostics
    # are not decoded using the host's cp932 code page.
    effective_timeout = max(timeout, 90.0) if spec.is_wsl else timeout
    command, _ = build_runtime_command(
        spec,
        ["-c", code],
        environment={"CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
    )
    attempts = 2 if spec.is_wsl else 1
    result = None
    last_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            result = runner(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=effective_timeout,
            )
            break
        except subprocess.TimeoutExpired as exc:
            last_error = exc
            if attempt + 1 < attempts:
                continue
            break
        except (OSError, subprocess.SubprocessError) as exc:
            last_error = exc
            break
    if result is None:
        return {
            "python": spec.label,
            "available": False,
            "reason": str(last_error or "FastVideo probe failed"),
        }
    raw = result.stdout.strip()
    if raw == "available":  # backwards-compatible test/older probe output
        return {
            "python": spec.label,
            "available": result.returncode == 0,
            "fastvideo": result.returncode == 0,
            "api": result.returncode == 0,
            "minimax_h3": result.returncode == 0,
            "vsa_kernel": result.returncode == 0,
            "cuda": result.returncode == 0,
            "reason": "available" if result.returncode == 0 else "fastvideo probe failed",
        }
    # Triton/FastVideo may emit an informational line before the JSON probe
    # payload (especially on the first WSL import while its CUDA helper is
    # compiled).  Parse the last JSON object line rather than treating those
    # harmless logs as a failed probe.
    details: dict[str, object] = {}
    for line in reversed(raw.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            details = candidate
            break
    if not isinstance(details, dict):
        details = {}
    available = bool(result.returncode == 0 and details.get("available"))
    if not available:
        reason = result.stderr.strip()
        if not reason:
            missing = [
                label
                for label, key in (
                    ("FastVideo", "fastvideo"),
                    ("FastVideo API", "api"),
                    ("MiniMax H3 model class", "minimax_h3"),
                    ("CUDA", "cuda"),
                )
                if details.get(key) is not True
            ]
            reason = (
                "missing or unavailable: " + ", ".join(missing)
                if missing
                else "FastVideo runtime probe failed"
            )
    else:
        reason = "available"
    return {
        "python": spec.label,
        **details,
        "available": available,
        "reason": reason,
    }


def fast_profile_issues(runtime: dict[str, object]) -> tuple[str, ...]:
    """Return blockers for the measured Blackwell ``profile=all`` route.

    The regular FastH3 route can use Triton VSA on older supported CUDA GPUs.
    The fastest published route additionally relies on the sm100a tile-64
    kernel and FA4, so this check must remain explicit instead of silently
    falling back to a different performance profile.
    """

    issues: list[str] = []
    capabilities = {
        str(value) for value in (runtime.get("gpu_capabilities") or ())
    }
    if not capabilities.intersection(BLACKWELL_FASTVIDEO_CAPABILITIES):
        issues.append(
            "FastH3最速プロファイルはBlackwell（sm100a/sm103a、compute capability 10.0/10.3）専用です"
        )
    memories = [
        int(value)
        for value in (runtime.get("gpu_memory_bytes") or ())
        if isinstance(value, (int, float, str)) and str(value).strip()
    ]
    if not any(
        str(capability) in BLACKWELL_FASTVIDEO_CAPABILITIES
        and index < len(memories)
        and memories[index] >= FAST_PROFILE_MIN_VRAM_BYTES
        for index, capability in enumerate(runtime.get("gpu_capabilities") or ())
    ):
        issues.append(
            "FastH3最速プロファイルには80 GiB以上のVRAMを持つBlackwell GPUが必要です（単一GPU実行の安全ゲート）"
        )
    if runtime.get("vsa_sm100a") is not True:
        issues.append(
            "FastH3最速プロファイルにはfastvideo-kernelのsm100a VSA拡張が必要です"
        )
    if runtime.get("fa4") is not True:
        issues.append(
            "FastH3最速プロファイルにはFlashAttention-4（flash_attn.cute）が必要です"
        )
    return tuple(dict.fromkeys(issues))


def fasth3_preflight(
    checkpoint: str | Path | None = None,
    *,
    python: str | Path = sys.executable,
    model_directory: str | Path | None = None,
    task: str = "auto",
    performance_profile: str = FASTH3_PROFILE_STRICT,
) -> dict[str, object]:
    """Combine optional Kijai diagnostics with FastVideo execution checks.

    This is intentionally an advisory/preflight API.  It never falls back to
    dense attention or the standard H3 loader when a VSA requirement is absent.
    ``ready`` refers to the separate official FastVideo directory route; the
    Kijai single-file diagnostics are returned in ``checkpoint_issues`` and do
    not make that route executable.
    """

    if performance_profile not in {FASTH3_PROFILE_STRICT, FASTH3_PROFILE_FAST}:
        raise ValueError(f"unknown FastH3 performance profile: {performance_profile}")

    checkpoint_path = (
        Path(checkpoint).expanduser().resolve(strict=False)
        if checkpoint is not None and str(checkpoint).strip()
        else None
    )
    info = (
        inspect_kijai_fastvideo_checkpoint(checkpoint_path)
        if checkpoint_path is not None
        else None
    )
    checkpoint_issues = [] if info is None else list(info.issues)
    warnings = [] if info is None else list(info.warnings)
    if info is not None and info.is_kijai_fastvideo_single_file:
        checkpoint_issues.append(
            "Kijaiの単一safetensorsはFastVideoのディレクトリローダーへ直接渡せません"
        )
        warnings.append(
            "Kijaiの単一safetensorsは外部ComfyUI/VSA用です。"
            "HAYATEのFastVideo実行には公式FastVideoディレクトリを使用します"
        )
    # ``issues`` retains the historical aggregate field for CLI/API clients;
    # ``blocking_issues`` is the authoritative readiness gate because the
    # optional Kijai single-file diagnostics are informational for the
    # separate official FastVideo directory route.
    issues: list[str] = list(checkpoint_issues)
    blocking_issues: list[str] = []
    if task not in {"auto", "t2va"}:
        blocking_issues.append("FastH3 VSA preview is currently validated for T2VA only")
    runtime = probe_fastvideo_runtime(python)
    if not runtime["available"]:
        blocking_issues.append(f"FastVideo runtime is unavailable: {runtime['reason']}")
    elif runtime.get("vsa_kernel") is not True:
        warnings.append(
            "fastvideo-kernel is not detected; the HAYATE adapter will use the Triton VSA fallback"
        )
    directory_issues: tuple[str, ...] = ()
    if model_directory is not None and str(model_directory).strip():
        model_path = Path(model_directory).expanduser().resolve(strict=False)
        directory_issues = validate_fastvideo_model_directory(
            model_path, strict=(model_path / "modular_model_index.json").is_file()
        )
        blocking_issues.extend(directory_issues)
        modular_manifest = model_path / "modular_model_index.json"
        if modular_manifest.is_file():
            try:
                manifest = json.loads(modular_manifest.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                manifest = {}
            if isinstance(manifest, dict) and manifest.get("transformer_ref"):
                warnings.append(
                    "公式manifestのtransformer_refはRef2VA用の外部参照です。"
                    "現在のT2VA経路では使用しません"
                )
    else:
        directory_issues = ("a FastVideo model directory must be configured",)
        blocking_issues.extend(directory_issues)
    fast_profile_blockers = list(fast_profile_issues(runtime))
    if performance_profile == FASTH3_PROFILE_FAST:
        blocking_issues.extend(fast_profile_blockers)
    issues.extend(blocking_issues)
    # The single-file conversion is optional evidence, not the input consumed
    # by FastVideo.  Readiness is therefore determined by the external runtime
    # and directory contract; a missing Kijai file must not block an official
    # FastVideo snapshot that is already configured.
    runtime_ready = bool(runtime.get("available"))
    memory_bytes = runtime.get("system_memory_bytes")
    if isinstance(memory_bytes, (int, float)) and memory_bytes < 16 * 1024**3:
        warnings.append(
            "FastVideo実行環境から見えるシステムRAMが"
            f"{memory_bytes / 1024**3:.1f} GiBです。35BモデルのCPUオフロードには不足する可能性があります"
        )
    directory_ready = bool(model_directory is not None and str(model_directory).strip() and not directory_issues)
    execution_adapter_available = True
    ready = bool(runtime_ready and directory_ready and execution_adapter_available and not blocking_issues)
    return {
        "ready": ready,
        "performance_profile": performance_profile,
        "fast_profile_ready": bool(
            runtime_ready
            and directory_ready
            and execution_adapter_available
            and not fast_profile_blockers
        ),
        "fast_profile_issues": fast_profile_blockers,
        "execution_adapter_available": execution_adapter_available,
        "checkpoint": info.to_dict() if info is not None else None,
        "kijai_single_file_runnable": False,
        "fastvideo_repository": FASTVIDEO_FASTH3_REPOSITORY,
        "fastvideo_revision": FASTVIDEO_FASTH3_REVISION,
        "fastvideo_source_url": FASTVIDEO_FASTH3_SOURCE_URL,
        "runtime": runtime,
        "model_directory": (
            str(Path(model_directory).expanduser().resolve(strict=False))
            if model_directory is not None and str(model_directory).strip()
            else None
        ),
        "directory_issues": list(directory_issues),
        "issues": list(dict.fromkeys(issues)),
        "blocking_issues": list(dict.fromkeys(blocking_issues)),
        "checkpoint_issues": list(dict.fromkeys(checkpoint_issues)),
        "warnings": list(dict.fromkeys(warnings)),
        "limitations": [
            "Kijai FastH3 single-file conversion is for external ComfyUI/VSA; it is not a HAYATE FastVideo input",
            "HAYATE execution uses the separate official FastVideo directory snapshot",
            "the current validated preview is T2VA; I2V/FL2VA is not enabled",
            "VSA gate and sparse-attention support are required; dense fallback is disabled",
            "the Blackwell fast profile additionally requires sm100a VSA and FlashAttention-4",
        ],
    }
