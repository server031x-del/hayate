from __future__ import annotations

import json
from pathlib import Path

from hayate.errors import LoaderValidationError


class INT8ConvRotCheckpointBinding:
    """Bind marker-declared INT8 ConvRot Linear weights to comfy-kitchen."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve(strict=False)

    def materialize(self, prefix: str, *, device: str = "cpu", orig_dtype: str = "float32"):
        try:
            import torch
            from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout
            from safetensors import safe_open
        except ImportError as exc:
            raise LoaderValidationError(
                "INT8 ConvRot materialization requires the HAYATE cuda/generation extra"
            ) from exc
        dtype = getattr(torch, orig_dtype, None)
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise LoaderValidationError(f"unsupported INT8 ConvRot logical dtype: {orig_dtype}")
        with safe_open(self.path, framework="pt", device="cpu") as handle:
            try:
                qdata = handle.get_tensor(prefix + ".weight").clone()
                scale = handle.get_tensor(prefix + ".weight_scale").clone()
                marker_tensor = handle.get_tensor(prefix + ".comfy_quant").clone()
            except KeyError as exc:
                raise LoaderValidationError(f"incomplete INT8 ConvRot bundle for {prefix}") from exc
        try:
            marker = json.loads(bytes(marker_tensor.tolist()).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise LoaderValidationError(f"invalid INT8 ConvRot marker for {prefix}") from exc
        if marker.get("format") != "int8_tensorwise" or marker.get("convrot") is not True:
            raise LoaderValidationError(f"marker is not INT8 tensorwise ConvRot for {prefix}")
        groupsize = int(marker.get("convrot_groupsize", 0))
        if qdata.dtype != torch.int8 or qdata.ndim != 2 or groupsize != 256:
            raise LoaderValidationError(f"unsupported INT8 ConvRot tensor contract for {prefix}")
        target = torch.device(device)
        params = TensorWiseINT8Layout.Params(
            scale=scale.to(target),
            orig_dtype=dtype,
            orig_shape=tuple(qdata.shape),
            is_weight=True,
            convrot=True,
            convrot_groupsize=groupsize,
        )
        return QuantizedTensor(qdata.to(target), "TensorWiseINT8Layout", params)

    def materialize_converted(
        self,
        prefix: str,
        converter,
        converter_config: dict,
        *,
        device: str = "cpu",
        orig_dtype: str = "float32",
    ) -> dict:
        try:
            import torch
            from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout
            from safetensors import safe_open
        except ImportError as exc:
            raise LoaderValidationError(
                "INT8 ConvRot materialization requires the HAYATE cuda/generation extra"
            ) from exc
        dtype = getattr(torch, orig_dtype, None)
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise LoaderValidationError(f"unsupported INT8 ConvRot logical dtype: {orig_dtype}")
        with safe_open(self.path, framework="pt", device="cpu") as handle:
            qdata = handle.get_tensor(prefix + ".weight").clone()
            scale = handle.get_tensor(prefix + ".weight_scale").clone()
            marker_tensor = handle.get_tensor(prefix + ".comfy_quant").clone()
        marker = json.loads(bytes(marker_tensor.tolist()).decode("utf-8"))
        if (
            marker.get("format") != "int8_tensorwise"
            or marker.get("convrot") is not True
            or int(marker.get("convrot_groupsize", 0)) != 256
        ):
            raise LoaderValidationError(f"marker is not supported INT8 ConvRot for {prefix}")
        q_parts = converter(prefix + ".weight", qdata, converter_config)
        scale_parts = converter(prefix + ".weight", scale, converter_config)
        if len(q_parts) != len(scale_parts):
            raise LoaderValidationError(f"conversion count mismatch for {prefix}")
        target = torch.device(device)
        output = {}
        for (native_key, q_part), (scale_key, scale_part) in zip(q_parts, scale_parts):
            if native_key != scale_key:
                raise LoaderValidationError(f"conversion key mismatch for {prefix}")
            params = TensorWiseINT8Layout.Params(
                scale=scale_part.to(target),
                orig_dtype=dtype,
                orig_shape=tuple(q_part.shape),
                is_weight=True,
                convrot=True,
                convrot_groupsize=256,
            )
            output[native_key] = QuantizedTensor(
                q_part.to(target), "TensorWiseINT8Layout", params
            )
        return output
