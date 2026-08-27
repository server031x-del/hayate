"""Parallel Decoding Distillation support for MiniMax-H3.

The checkpoint contract and sampling schedule follow Alibaba PAI's
MiniMax-H3-Acc-LoRAs release.  HAYATE keeps the PDD adapters separate from the
quantized backbone instead of merging them into W4A8 weights.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from hayate.loaders.tensor_source import CheckpointTensorSource
from hayate.models.safetensors_header import read_safetensors_header

logger = logging.getLogger(__name__)

EXPECTED_TARGETS = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
    "adaln_proj.linear",
)
HEAD_KEYS = {
    "proj_out.weight",
    "proj_out.bias",
    "audio_proj_out.weight",
    "audio_proj_out.bias",
}
EXPECTED_ADAPTER_MODULES = 362
# Comfy-Org/MiniMax-H3's released FL2VA pruned coordinate table.  The affine
# map is meaningful only in this exact rank-8 coordinate gauge.
COMFY_PRUNED_ADALN_TABLE_SHA256 = "ac8727cdec52137c73878d004de5bd2a0e19227e8311e29ab3b68f328310e34e"


@dataclass(frozen=True)
class PDDCheckpointConfig:
    path: Path
    num_steps: int
    block_size: int
    rank: int
    alpha: float
    targets: tuple[str, ...]
    tensor_count: int

    @property
    def nfe(self) -> int:
        return self.num_steps // self.block_size

    @classmethod
    def inspect(cls, path: str | Path) -> "PDDCheckpointConfig":
        resolved = Path(path).expanduser().resolve(strict=False)
        if not resolved.is_file():
            raise ValueError(f"PDD checkpoint does not exist: {resolved}")
        header = read_safetensors_header(resolved)
        metadata = header.metadata
        try:
            num_steps = int(metadata["pdd_num_steps"])
            block_size = int(metadata["pdd_block_size"])
            rank = int(metadata["lora_rank"])
            alpha = float(metadata["lora_alpha"])
            targets = tuple(
                value.strip()
                for value in str(metadata["lora_targets"]).split(",")
                if value.strip()
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid PDD metadata in {resolved}: {exc}") from exc
        if num_steps < 1 or block_size < 1 or num_steps % block_size:
            raise ValueError(
                f"PDD grid {num_steps} must be positive and divisible by block size {block_size}"
            )
        if rank < 1 or alpha <= 0:
            raise ValueError("PDD LoRA rank and alpha must be positive")
        if targets != EXPECTED_TARGETS:
            raise ValueError(
                "unsupported PDD target contract: " + ",".join(targets)
            )
        names = {tensor.name for tensor in header.tensors}
        missing_heads = sorted(HEAD_KEYS - names)
        if missing_heads:
            raise ValueError(f"PDD checkpoint is missing output heads: {missing_heads}")
        adapter_keys = names - HEAD_KEYS
        if not adapter_keys or any(
            not (key.endswith(".lora_down") or key.endswith(".lora_up"))
            for key in adapter_keys
        ):
            raise ValueError("PDD checkpoint contains unsupported tensor names")
        down = {key.removesuffix(".lora_down") for key in adapter_keys if key.endswith(".lora_down")}
        up = {key.removesuffix(".lora_up") for key in adapter_keys if key.endswith(".lora_up")}
        if down != up:
            raise ValueError("PDD checkpoint has unmatched lora_down/lora_up tensors")
        if len(down) != EXPECTED_ADAPTER_MODULES:
            raise ValueError(
                f"unsupported PDD adapter count: {len(down)} (expected {EXPECTED_ADAPTER_MODULES})"
            )
        adaln_count = sum(name.endswith("adaln_proj.linear") for name in down)
        if adaln_count != 50 or len(down) - adaln_count != 312:
            raise ValueError(
                "unsupported PDD target distribution: "
                f"AdaLN={adaln_count}, other={len(down) - adaln_count}"
            )
        return cls(resolved, num_steps, block_size, rank, alpha, targets, len(names))


def shifted_sigma(shift, sigma):
    return shift * sigma / (1 + (shift - 1) * sigma)


def pdd_time_grid(shift: float, num_steps: int):
    import torch

    sigma = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    return 1.0 - shifted_sigma(float(shift), sigma)


class PDDLoRALinear:
    """Factory namespace so torch is imported only in the generation process."""

    @staticmethod
    def create(base, down, up, scaling: float, offset=None):
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional

        class _PDDLoRALinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = base
                self.base.requires_grad_(False)
                self.lora_down = nn.Parameter(down, requires_grad=False)
                self.lora_up = nn.Parameter(up, requires_grad=False)
                self.scaling = float(scaling)
                if offset is not None:
                    self.register_buffer("lora_offset", offset.to(torch.float32), persistent=False)
                else:
                    self.lora_offset = None

            @property
            def weight(self):
                return self.base.weight

            @property
            def bias(self):
                return self.base.bias

            @property
            def in_features(self):
                return self.base.in_features

            @property
            def out_features(self):
                return self.base.out_features

            def forward(self, hidden_states):
                output = self.base(hidden_states)
                adapter_input = hidden_states.to(self.lora_down.dtype)
                low_rank = functional.linear(adapter_input, self.lora_down)
                update = functional.linear(low_rank.to(self.lora_up.dtype), self.lora_up)
                result = output + update.to(output.dtype) * self.scaling
                if self.lora_offset is not None:
                    result = result + self.lora_offset.to(result.dtype) * self.scaling
                return result

        return _PDDLoRALinear()


class PDDParallelHead:
    @staticmethod
    def create(source, weight, bias, num_steps: int):
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional

        class _PDDParallelHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_steps = int(num_steps)
                self.in_features = source.in_features
                self.out_features = source.out_features
                self.weight = nn.Parameter(weight, requires_grad=False)
                self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)
                self.register_buffer(
                    "active_coefficients",
                    torch.ones(1, dtype=torch.float64),
                    persistent=False,
                )
                self.active_start = 0

            def set_interval(self, start: int, coefficients) -> None:
                end = start + int(coefficients.numel())
                if start < 0 or end > self.num_steps:
                    raise ValueError(f"PDD interval [{start}, {end}) exceeds {self.num_steps}")
                self.active_start = int(start)
                self.active_coefficients = coefficients.detach().to(
                    device=self.weight.device, dtype=self.weight.dtype
                )

            def forward(self, hidden_states):
                selected_weight = self.weight[
                    self.active_start : self.active_start + self.active_coefficients.numel()
                ]
                weight = torch.einsum(
                    "n,noi->oi", self.active_coefficients, selected_weight
                )
                bias = None
                if self.bias is not None:
                    bias = torch.einsum(
                        "n,no->o",
                        self.active_coefficients,
                        self.bias[
                            self.active_start : self.active_start
                            + self.active_coefficients.numel()
                        ],
                    )
                return functional.linear(hidden_states, weight, bias)

        return _PDDParallelHead()


class PDDStepController:
    def __init__(self, transformer, *, video_shift: float, audio_shift: float, config):
        self.transformer = transformer
        self.config = config
        self.video_steps = pdd_time_grid(video_shift, config.num_steps).diff()
        self.audio_steps = pdd_time_grid(audio_shift, config.num_steps).diff()
        self.index = 0
        self.forward_calls = 0
        self.pinned_adapter_bytes = 0
        self.arm(0)

    def _coefficients(self, steps, start: int):
        selected = steps[start : start + self.config.block_size]
        return selected / selected.sum()

    def arm(self, index: int) -> None:
        start = index * self.config.block_size
        self.transformer.proj_out.set_interval(
            start, self._coefficients(self.video_steps, start)
        )
        self.transformer.audio_proj_out.set_interval(
            start, self._coefficients(self.audio_steps, start)
        )

    def __call__(self, _module, _args, output):
        self.forward_calls += 1
        self.index = (self.index + 1) % self.config.nfe
        self.arm(self.index)
        return output

    def stats(self) -> dict[str, object]:
        return {
            "checkpoint": str(self.config.path),
            "grid_steps": self.config.num_steps,
            "block_size": self.config.block_size,
            "nfe": self.config.nfe,
            "forward_calls": self.forward_calls,
            "tensor_count": self.config.tensor_count,
            "pinned_adapter_bytes": self.pinned_adapter_bytes,
        }


def _scheduler_shift(checkpoint_dir: Path, name: str) -> float:
    path = checkpoint_dir / name / "scheduler_config.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return float(payload["shift"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read PDD scheduler shift from {path}: {exc}") from exc


def _load_adaln_affine(path: str | Path):
    import torch

    resolved = Path(path).expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise ValueError(
            "PDD on an AdaLN-pruned transformer needs adaln_affine.safetensors: "
            f"{resolved}"
        )
    with CheckpointTensorSource(resolved) as source:
        names = set(source.keys())
        if names != {"adaln_basis", "adaln_mean"}:
            raise ValueError(f"invalid AdaLN affine checkpoint keys: {sorted(names)}")
        basis = source.take_tensor("adaln_basis", copy_mmap=True)
        mean = source.take_tensor("adaln_mean", copy_mmap=True)
    if tuple(basis.shape) != (8, 2688) or tuple(mean.shape) != (2688,):
        raise ValueError(
            f"invalid AdaLN affine shapes: basis={tuple(basis.shape)}, mean={tuple(mean.shape)}"
        )
    if not torch.isfinite(basis).all() or not torch.isfinite(mean).all():
        raise ValueError("AdaLN affine map contains NaN or Inf")
    return basis, mean


def validate_pruned_adaln_coordinates(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve(strict=False)
    header = read_safetensors_header(resolved)
    try:
        table = next(tensor for tensor in header.tensors if tensor.name == "adaln_t_table")
    except StopIteration as exc:
        raise ValueError(f"pruned transformer has no adaln_t_table: {resolved}") from exc
    with resolved.open("rb") as handle:
        handle.seek(8 + header.header_size + table.data_start)
        payload = handle.read(table.data_end - table.data_start)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != COMFY_PRUNED_ADALN_TABLE_SHA256:
        raise ValueError(
            "AdaLN affine coordinates do not match this pruned transformer: "
            f"table sha256={digest}"
        )
    return digest


def validate_adaln_affine(path: str | Path) -> None:
    _load_adaln_affine(path)


def project_adaln_adapter(down, up, basis, mean):
    """Map a released AdaLN LoRA onto the pruned affine coordinates."""
    import torch

    down64 = down.to(torch.float64)
    projected = down64 @ basis.to(torch.float64).T
    offset = up.to(torch.float64) @ (down64 @ mean.to(torch.float64))
    return projected.to(torch.float32).contiguous(), offset.to(torch.float32).contiguous()


def apply_pdd_checkpoint(
    transformer,
    checkpoint: str | Path,
    checkpoint_dir: str | Path,
    *,
    adaln_affine: str | Path | None = None,
):
    import torch
    import torch.nn as nn

    config = PDDCheckpointConfig.inspect(checkpoint)
    if getattr(transformer, "_hayate_pdd_controller", None) is not None:
        raise RuntimeError("PDD is already installed on this transformer")

    header = read_safetensors_header(config.path)
    shapes = {tensor.name: tensor.shape for tensor in header.tensors}
    module_names = sorted(
        key.removesuffix(".lora_down")
        for key in shapes
        if key.endswith(".lora_down")
    )
    needs_projection = any(
        module_name.endswith("adaln_proj.linear")
        and transformer.get_submodule(module_name).in_features == 8
        for module_name in module_names
    )
    basis = mean = None
    if needs_projection:
        if adaln_affine is None:
            raise ValueError(
                "released PDD AdaLN adapters are 2688-wide, but this transformer is pruned to 8; "
                "configure HAYATE_PDD_ADALN_AFFINE"
            )
        basis, mean = _load_adaln_affine(adaln_affine)
        table = getattr(transformer, "adaln_t_table", None)
        if table is None:
            raise ValueError("8-wide PDD projection target has no adaln_t_table")
        table_bytes = table.detach().cpu().contiguous().numpy().tobytes()
        table_hash = hashlib.sha256(table_bytes).hexdigest()
        if table_hash != COMFY_PRUNED_ADALN_TABLE_SHA256:
            raise ValueError(
                "AdaLN affine coordinates do not match this pruned transformer: "
                f"table sha256={table_hash}"
            )
    pin_adapters = (
        os.environ.get("HAYATE_PDD_PIN_LORA", "").strip().lower()
        in {"1", "true", "yes", "on"}
        and torch.cuda.is_available()
    )
    pinned_bytes = 0
    loaded = 0
    with CheckpointTensorSource(config.path) as source:
        for module_name in module_names:
            parent_name, _, attribute = module_name.rpartition(".")
            parent = transformer.get_submodule(parent_name) if parent_name else transformer
            base = getattr(parent, attribute)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"PDD target is not nn.Linear: {module_name} ({type(base).__name__})")
            if not any(module_name.endswith(target) for target in config.targets):
                raise ValueError(f"PDD tensor targets an undeclared module: {module_name}")
            down = source.take_tensor(module_name + ".lora_down", copy_mmap=True)
            up = source.take_tensor(module_name + ".lora_up", copy_mmap=True)
            if not torch.isfinite(down).all() or not torch.isfinite(up).all():
                raise ValueError(f"PDD adapter contains NaN or Inf: {module_name}")
            offset = None
            if base.in_features == 8 and tuple(down.shape) == (config.rank, 2688):
                if not module_name.endswith("adaln_proj.linear") or basis is None or mean is None:
                    raise ValueError(f"unsupported PDD projection target: {module_name}")
                down, offset = project_adaln_adapter(down, up, basis, mean)
            if tuple(down.shape) != (config.rank, base.in_features):
                raise ValueError(f"PDD lora_down shape mismatch at {module_name}: {tuple(down.shape)}")
            if tuple(up.shape) != (base.out_features, config.rank):
                raise ValueError(f"PDD lora_up shape mismatch at {module_name}: {tuple(up.shape)}")
            if pin_adapters:
                try:
                    down = down.pin_memory()
                    up = up.pin_memory()
                    pinned_bytes += down.numel() * down.element_size()
                    pinned_bytes += up.numel() * up.element_size()
                except RuntimeError as exc:
                    pin_adapters = False
                    logger.warning("PDD pinned adapter fallback to pageable memory: %s", exc)
            setattr(
                parent,
                attribute,
                PDDLoRALinear.create(
                    base, down, up, config.alpha / config.rank, offset=offset
                ),
            )
            loaded += 2

        for name in ("proj_out", "audio_proj_out"):
            source_head = getattr(transformer, name)
            weight = source.take_tensor(name + ".weight", copy_mmap=True)
            bias = source.take_tensor(name + ".bias", copy_mmap=True)
            if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
                raise ValueError(f"PDD output head contains NaN or Inf: {name}")
            expected_weight = (config.num_steps, source_head.out_features, source_head.in_features)
            expected_bias = (config.num_steps, source_head.out_features)
            if tuple(weight.shape) != expected_weight or tuple(bias.shape) != expected_bias:
                raise ValueError(
                    f"PDD {name} shape mismatch: weight={tuple(weight.shape)}, bias={tuple(bias.shape)}"
                )
            setattr(
                transformer,
                name,
                PDDParallelHead.create(source_head, weight, bias, config.num_steps),
            )
            loaded += 2

        if loaded != config.tensor_count or source.remaining:
            raise RuntimeError(
                f"PDD strict tensor check failed: loaded={loaded}, expected={config.tensor_count}, "
                f"remaining={source.remaining}"
            )

    root = Path(checkpoint_dir).expanduser().resolve(strict=False)
    controller = PDDStepController(
        transformer,
        video_shift=_scheduler_shift(root, "scheduler"),
        audio_shift=_scheduler_shift(root, "audio_scheduler"),
        config=config,
    )
    controller.pinned_adapter_bytes = pinned_bytes
    transformer.register_forward_hook(controller, always_call=True)
    transformer._hayate_pdd_controller = controller
    transformer.eval().requires_grad_(False)
    logger.info(
        "HAYATE PDD loaded: %s (%d tensors, grid=%d, block=%d, NFE=%d, pinned=%.3f GiB)",
        config.path,
        loaded,
        config.num_steps,
        config.block_size,
        config.nfe,
        pinned_bytes / 1024**3,
    )
    return controller


def install_pdd_override(generation_module) -> bool:
    checkpoint = os.environ.get("HAYATE_PDD_CHECKPOINT", "").strip()
    if not checkpoint:
        return False
    if os.environ.get("HAYATE_EASYCACHE", "").strip().lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError("PDD and EasyCache are mutually exclusive")

    from minimax_video import model_loader

    original = model_loader.load_transformer

    def dispatch(*args, **kwargs):
        task = kwargs.get("task", args[2] if len(args) > 2 else "t2va")
        if task == "ref2va":
            raise ValueError("FL2VA PDD checkpoint cannot be used with transformer_ref")
        transformer = original(*args, **kwargs)
        checkpoint_dir = kwargs.get("ckpt_dir", args[0] if args else None)
        controller = apply_pdd_checkpoint(
            transformer,
            checkpoint,
            checkpoint_dir,
            adaln_affine=os.environ.get("HAYATE_PDD_ADALN_AFFINE") or None,
        )
        generation_module._hayate_pdd_controller = controller
        return transformer

    model_loader.load_transformer = dispatch
    return True
