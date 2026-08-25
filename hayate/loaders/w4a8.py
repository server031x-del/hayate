from __future__ import annotations

import importlib.util
import json
from collections import Counter
from typing import Any

from hayate.errors import LoaderNotReadyError, LoaderValidationError
from hayate.loaders.base import BaseModelLoader, LoaderStatus, LoaderValidation
from hayate.models.types import QuantizationFormat


class W4A8Loader(BaseModelLoader):
    """Header/layout validator for a future direct W4A8 execution backend."""

    format_name = "W4A8 mixed"

    def validate(self, hardware: Any | None = None) -> LoaderValidation:
        report = self.inspect()
        if report.quantization.format is not QuantizationFormat.W4A8_MIXED:
            return LoaderValidation(
                LoaderStatus.UNSUPPORTED,
                "header evidence does not identify W4A8 mixed",
                warnings=("filename and registry expectation are not accepted as format proof",),
            )
        tensor_map = {tensor.name: tensor for tensor in report.tensors}
        names = [tensor.name.lower() for tensor in report.tensors]
        relative_scales = [name for name in names if name.endswith(".weight_s_rel")]
        channel_scales = [name for name in names if name.endswith(".weight_s_channel")]
        codebooks = [name for name in names if name.endswith(".weight_codebook")]
        zero_names = [name for name in names if "zero" in name or name.endswith(".qzeros")]
        raw_quant_metadata = report.metadata.get("_quantization_metadata")
        parsed_quant_metadata: dict[str, Any] = {}
        if isinstance(raw_quant_metadata, str):
            try:
                candidate = json.loads(raw_quant_metadata)
                if isinstance(candidate, dict):
                    parsed_quant_metadata = candidate
            except json.JSONDecodeError:
                parsed_quant_metadata = {}
        layers = parsed_quant_metadata.get("layers", {})
        if not isinstance(layers, dict):
            layers = {}
        layout_errors: list[str] = []
        valid_layers = 0
        for prefix, layer_metadata in layers.items():
            if not isinstance(layer_metadata, dict):
                layout_errors.append(f"{prefix}: quantization metadata must be an object")
                continue
            required = {
                "weight": tensor_map.get(prefix + ".weight"),
                "weight_s_rel": tensor_map.get(prefix + ".weight_s_rel"),
                "weight_s_channel": tensor_map.get(prefix + ".weight_s_channel"),
                "weight_codebook": tensor_map.get(prefix + ".weight_codebook"),
            }
            missing = [name for name, tensor in required.items() if tensor is None]
            if missing:
                layout_errors.append(f"{prefix}: missing {', '.join(missing)}")
                continue
            weight = required["weight"]
            relative = required["weight_s_rel"]
            channel = required["weight_s_channel"]
            codebook = required["weight_codebook"]
            assert weight is not None and relative is not None and channel is not None and codebook is not None
            group_size = layer_metadata.get("group_size")
            layer_ok = (
                layer_metadata.get("format") == "asym_w4a8_int8"
                and group_size == 16
                and layer_metadata.get("convrot") is True
                and layer_metadata.get("convrot_groupsize") == 256
                and weight.dtype == "I8"
                and len(weight.shape) == 2
                and relative.dtype.startswith("F8_")
                and len(relative.shape) == 2
                and len(channel.shape) == 1
                and weight.shape[0] == relative.shape[0] == channel.shape[0]
                and weight.shape[1] * 2 == relative.shape[1] * group_size
                and channel.dtype == "F32"
                and codebook.dtype == "F32"
                and codebook.shape == (16,)
            )
            if layer_ok:
                valid_layers += 1
            else:
                layout_errors.append(f"{prefix}: shape/dtype/metadata contract mismatch")
        dtype_counts = Counter(tensor.dtype for tensor in report.tensors)
        warnings = ["run `hayate kernel-check` on the selected CUDA device before generation"]
        gpu_names: list[str] = []
        if hardware is not None:
            gpu_names = [getattr(gpu, "name", "unknown GPU") for gpu in getattr(hardware, "gpus", ())]
        missing_modules = [
            name for name in ("torch", "comfy_kitchen", "safetensors")
            if importlib.util.find_spec(name) is None
        ]
        complete = bool(layers) and not layout_errors and valid_layers == len(layers)
        status = LoaderStatus.SUPPORTED if complete and not missing_modules else LoaderStatus.PARTIAL
        reason = (
            "W4A8 layout is complete; staged upstream binding and native CUDA execution are enabled"
            if status is LoaderStatus.SUPPORTED
            else "W4A8 header recognized but its execution contract is incomplete"
        )
        requirements = []
        if layout_errors:
            requirements.append("fix all packed weight/scale/codebook layout mismatches")
        if missing_modules:
            requirements.append("install HAYATE with the generation extra")
        return LoaderValidation(
            status,
            reason,
            requirements=tuple(requirements),
            warnings=tuple(warnings),
            details={
                "dtype_counts": dict(dtype_counts),
                "quantized_layer_count": len(layers),
                "valid_quantized_layer_count": valid_layers,
                "layout_error_count": len(layout_errors),
                "layout_error_examples": layout_errors[:10],
                "relative_group_scale_count": len(relative_scales),
                "channel_scale_count": len(channel_scales),
                "codebook_count": len(codebooks),
                "zero_point_tensor_count": len(zero_names),
                "detected_gpus": gpu_names,
                "format": "asym_w4a8_int8",
                "group_size": 16 if layers else "UNKNOWN",
                "convrot_group_size": 256 if layers else "UNKNOWN",
                "required_kernel": "comfy-kitchen 0.2.31 AsymW4A8Int8Layout",
                "missing_modules": missing_modules,
                "rtx_3060_sm86_execution": "FULL_STRICT_LOAD_PASS / NATIVE_REAL_LAYER_PASS",
            },
        )

    def load(self, device: str) -> Any:
        validation = self.validate()
        if validation.status is not LoaderStatus.SUPPORTED:
            raise LoaderNotReadyError(f"{validation.reason}. Target device: {device}")
        if device not in {"cpu", "cuda", "cuda:0"}:
            raise LoaderValidationError(f"unsupported W4A8 binding device: {device}")
        from hayate.loaders.w4a8_binding import W4A8CheckpointBinding

        self._loaded = W4A8CheckpointBinding(self.spec.path)
        return self._loaded
