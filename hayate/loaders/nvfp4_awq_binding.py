from __future__ import annotations

import json
from pathlib import Path

from hayate.errors import LoaderValidationError


class NVFP4AWQCheckpointBinding:
    """Reconstruct NVFP4 weights and expose optional ModelOpt AWQ input scales."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve(strict=False)

    def materialize(self, prefix: str, *, device: str = "cpu", orig_dtype: str = "bfloat16"):
        try:
            import torch
            from comfy_kitchen.tensor import QuantizedTensor, TensorCoreNVFP4Layout
            from safetensors import safe_open
        except ImportError as exc:
            raise LoaderValidationError(
                "NVFP4 materialization requires the HAYATE cuda/generation extra"
            ) from exc
        dtype = getattr(torch, orig_dtype, None)
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise LoaderValidationError(f"unsupported NVFP4 logical dtype: {orig_dtype}")
        with safe_open(self.path, framework="pt", device="cpu") as handle:
            try:
                qdata = handle.get_tensor(prefix + ".weight").clone()
                block_scale = handle.get_tensor(prefix + ".weight_scale").clone()
                tensor_scale = handle.get_tensor(prefix + ".weight_scale_2").clone()
                marker_tensor = handle.get_tensor(prefix + ".comfy_quant").clone()
            except KeyError as exc:
                raise LoaderValidationError(f"incomplete NVFP4 tensor bundle for {prefix}") from exc
        try:
            marker = json.loads(bytes(marker_tensor.tolist()).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise LoaderValidationError(f"invalid NVFP4 marker for {prefix}") from exc
        if marker.get("format") != "nvfp4":
            raise LoaderValidationError(f"marker is not NVFP4 for {prefix}")
        if qdata.dtype != torch.uint8 or qdata.ndim != 2:
            raise LoaderValidationError(f"unsupported NVFP4 storage for {prefix}")
        target = torch.device(device)
        params = TensorCoreNVFP4Layout.Params(
            scale=tensor_scale.to(target),
            block_scale=block_scale.to(target),
            orig_dtype=dtype,
            orig_shape=(qdata.shape[0], qdata.shape[1] * 2),
        )
        return QuantizedTensor(qdata.to(target), "TensorCoreNVFP4Layout", params)

    def pre_quant_scale(self, prefix: str, *, device: str = "cpu"):
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise LoaderValidationError("safetensors is required") from exc
        with safe_open(self.path, framework="pt", device="cpu") as handle:
            key = prefix + ".pre_quant_scale"
            if key not in handle.keys():
                return None
            return handle.get_tensor(key).clone().to(device)
