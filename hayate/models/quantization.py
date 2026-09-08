from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterable

from hayate.models.types import (
    DetectionConfidence,
    QuantizationDetection,
    QuantizationFormat,
    TensorInfo,
)


def _metadata_text(metadata: dict[str, Any]) -> str:
    try:
        return json.dumps(metadata, sort_keys=True, ensure_ascii=True).lower()
    except (TypeError, ValueError):
        return str(metadata).lower()


def detect_quantization(
    metadata: dict[str, Any], tensors: Iterable[TensorInfo]
) -> QuantizationDetection:
    tensor_list = tuple(tensors)
    names = tuple(tensor.name.lower() for tensor in tensor_list)
    metadata_text = _metadata_text(metadata)
    dtypes = Counter(tensor.dtype.upper() for tensor in tensor_list)

    explicit_quant_fields = " ".join(
        str(value).lower()
        for key, value in metadata.items()
        if any(marker in str(key).lower() for marker in ("quant", "format", "layout"))
    )
    has_w4a8_meta = (
        "asym_w4a8_int8" in metadata_text
        or "w4a8" in explicit_quant_fields
        or "w4_a8" in explicit_quant_fields
    )
    if has_w4a8_meta:
        return QuantizationDetection(
            QuantizationFormat.W4A8_MIXED,
            DetectionConfidence.HIGH,
            ("header metadata explicitly identifies W4A8",),
        )

    has_nvfp4 = "nvfp4" in metadata_text or "nv_fp4" in metadata_text
    has_awq = "awq" in metadata_text
    if has_nvfp4 and has_awq:
        return QuantizationDetection(
            QuantizationFormat.NVFP4_AWQ,
            DetectionConfidence.HIGH,
            ("header metadata explicitly identifies NVFP4", "header metadata identifies AWQ"),
        )

    # ComfyUI's Qwen3-VL NVFP4/AWQ single-file export currently carries no
    # top-level metadata. Its header contract is still distinctive: packed U8
    # weights, FP8 primary scales, scalar FP32 secondary scales, per-module
    # comfy_quant descriptors, and AWQ pre-quant scales. Require many complete
    # groups so an isolated tensor cannot trigger this route.
    tensor_by_name = {tensor.name.lower(): tensor for tensor in tensor_list}
    comfy_markers = [name for name in names if name.endswith(".comfy_quant")]
    primary_fp8_scales = [
        name
        for name in names
        if name.endswith(".weight_scale")
        and tensor_by_name[name].dtype.upper().startswith("F8_")
    ]
    secondary_scales = [name for name in names if name.endswith(".weight_scale_2")]
    pre_quant_scales = [name for name in names if name.endswith(".pre_quant_scale")]
    packed_u8_weights = [
        name
        for name in names
        if name.endswith(".weight") and tensor_by_name[name].dtype.upper() == "U8"
    ]
    nvfp4_complete_groups = sum(
        1
        for weight_name in packed_u8_weights
        if weight_name[: -len(".weight")] + ".weight_scale" in tensor_by_name
        and weight_name[: -len(".weight")] + ".weight_scale_2" in tensor_by_name
        and weight_name[: -len(".weight")] + ".comfy_quant" in tensor_by_name
    )
    if (
        nvfp4_complete_groups >= 8
        and len(primary_fp8_scales) >= nvfp4_complete_groups
        and len(secondary_scales) >= nvfp4_complete_groups
        and pre_quant_scales
    ):
        return QuantizationDetection(
            QuantizationFormat.NVFP4_AWQ,
            DetectionConfidence.HIGH,
            (
                f"found {nvfp4_complete_groups} packed U8/FP8/FP32 NVFP4 tensor groups",
                f"found {len(comfy_markers)} comfy_quant descriptors",
                f"found {len(pre_quant_scales)} AWQ pre_quant_scale tensors",
            ),
        )

    quant_marker_names = comfy_markers
    scale_names = [name for name in names if name.endswith(".weight_scale")]
    int8_weights = [
        tensor.name
        for tensor in tensor_list
        if tensor.dtype.upper() == "I8" and tensor.name.lower().endswith(".weight")
    ]
    has_convrot_meta = "convrot" in metadata_text or "conv_rot" in metadata_text
    marker_coverage = (
        len(int8_weights) / len(quant_marker_names) if quant_marker_names else 0.0
    )
    if (has_convrot_meta and int8_weights) or (
        quant_marker_names and scale_names and marker_coverage >= 0.8
    ):
        evidence = []
        if has_convrot_meta:
            evidence.append("header metadata identifies ConvRot")
        if quant_marker_names:
            evidence.append("found .comfy_quant marker tensors")
        if scale_names:
            evidence.append("found .weight_scale tensors")
        evidence.append("found INT8 weight tensors")
        return QuantizationDetection(
            QuantizationFormat.INT8_CONVROT,
            DetectionConfidence.HIGH if has_convrot_meta else DetectionConfidence.MEDIUM,
            tuple(evidence),
        )

    fp8_dtypes = sorted(dtype for dtype in dtypes if dtype.startswith("F8_"))
    if fp8_dtypes:
        return QuantizationDetection(
            QuantizationFormat.FP8,
            DetectionConfidence.HIGH,
            (f"safetensors dtype includes {', '.join(fp8_dtypes)}",),
        )

    nonempty = sum(dtypes.values())
    for dtype, quant_format in (
        ("F32", QuantizationFormat.FP32),
        ("F16", QuantizationFormat.FP16),
        ("BF16", QuantizationFormat.BF16),
    ):
        if nonempty and dtypes[dtype] == nonempty:
            return QuantizationDetection(
                quant_format,
                DetectionConfidence.HIGH,
                (f"all {nonempty} tensors use safetensors dtype {dtype}",),
            )

    return QuantizationDetection(
        QuantizationFormat.UNKNOWN,
        DetectionConfidence.NONE,
        ("no documented header evidence matched a known format",),
    )
