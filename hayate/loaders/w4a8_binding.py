from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hayate.errors import LoaderValidationError
from hayate.models.safetensors_header import read_safetensors_header


@dataclass(frozen=True)
class W4A8LayerContract:
    prefix: str
    group_size: int
    convrot_groupsize: int
    original_shape: tuple[int, int]


class W4A8CheckpointBinding:
    """Reconstruct comfy-kitchen tensors directly from the audited safetensors layout."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve(strict=False)
        self.header = read_safetensors_header(self.path)
        raw = self.header.metadata.get("_quantization_metadata")
        try:
            metadata = json.loads(raw) if isinstance(raw, str) else {}
        except json.JSONDecodeError as exc:
            raise LoaderValidationError("invalid W4A8 _quantization_metadata JSON") from exc
        layers = metadata.get("layers", {}) if isinstance(metadata, dict) else {}
        if not isinstance(layers, dict) or not layers:
            raise LoaderValidationError("W4A8 checkpoint has no declared quantized layers")
        self._layers: dict[str, dict[str, Any]] = layers

    @property
    def layer_prefixes(self) -> tuple[str, ...]:
        return tuple(self._layers)

    def contract(self, prefix: str) -> W4A8LayerContract:
        try:
            layer = self._layers[prefix]
            weight = next(t for t in self.header.tensors if t.name == prefix + ".weight")
        except (KeyError, StopIteration) as exc:
            raise LoaderValidationError(f"unknown or incomplete W4A8 layer: {prefix}") from exc
        if layer.get("format") != "asym_w4a8_int8":
            raise LoaderValidationError(f"unsupported W4A8 format for {prefix}")
        return W4A8LayerContract(
            prefix=prefix,
            group_size=int(layer.get("group_size", 0)),
            convrot_groupsize=int(layer.get("convrot_groupsize", 0)),
            original_shape=(weight.shape[0], weight.shape[1] * 2),
        )

    def materialize(self, prefix: str, *, device: str = "cpu", orig_dtype: str = "bfloat16"):
        """Load one packed layer and wrap it without dequantizing its weight payload."""

        try:
            import torch
            from comfy_kitchen.tensor import AsymW4A8Int8Layout, QuantizedTensor
            from safetensors import safe_open
        except ImportError as exc:
            raise LoaderValidationError(
                "W4A8 materialization requires the HAYATE cuda/generation extra"
            ) from exc
        contract = self.contract(prefix)
        dtype = getattr(torch, orig_dtype, None)
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise LoaderValidationError(f"unsupported W4A8 logical dtype: {orig_dtype}")
        with safe_open(self.path, framework="pt", device="cpu") as handle:
            try:
                qdata = handle.get_tensor(prefix + ".weight").clone()
                relative = handle.get_tensor(prefix + ".weight_s_rel").clone()
                channel = handle.get_tensor(prefix + ".weight_s_channel").clone()
                codebook = handle.get_tensor(prefix + ".weight_codebook").clone()
            except KeyError as exc:
                raise LoaderValidationError(f"incomplete W4A8 tensor bundle for {prefix}") from exc
        target = torch.device(device)
        qdata = qdata.to(target)
        relative = relative.to(target)
        channel = channel.to(target)
        codebook = codebook.to(target)
        params = AsymW4A8Int8Layout.Params(
            scale=relative,
            s_channel=channel,
            correction=None,
            codebook=codebook,
            orig_dtype=dtype,
            orig_shape=contract.original_shape,
            group_size=contract.group_size,
            convrot_groupsize=contract.convrot_groupsize,
        )
        return QuantizedTensor(qdata, "AsymW4A8Int8Layout", params)

    def materialize_converted(
        self,
        prefix: str,
        converter,
        *,
        device: str = "cpu",
        orig_dtype: str = "bfloat16",
    ) -> dict[str, Any]:
        """Reuse upstream's exact row/key conversion for a packed W4A8 layer."""

        try:
            import torch
            from comfy_kitchen.tensor import AsymW4A8Int8Layout, QuantizedTensor
            from safetensors import safe_open
        except ImportError as exc:
            raise LoaderValidationError(
                "W4A8 materialization requires the HAYATE cuda/generation extra"
            ) from exc
        contract = self.contract(prefix)
        dtype = getattr(torch, orig_dtype, None)
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise LoaderValidationError(f"unsupported W4A8 logical dtype: {orig_dtype}")
        with safe_open(self.path, framework="pt", device="cpu") as handle:
            qdata = handle.get_tensor(prefix + ".weight").clone()
            relative = handle.get_tensor(prefix + ".weight_s_rel").clone()
            channel = handle.get_tensor(prefix + ".weight_s_channel").clone()
            codebook = handle.get_tensor(prefix + ".weight_codebook").clone()

        def convert_rows(value):
            return converter(prefix + ".weight", value, qkv_head_dim=0)

        q_parts = convert_rows(qdata)
        relative_parts = convert_rows(relative)
        channel_parts = convert_rows(channel)
        if not (len(q_parts) == len(relative_parts) == len(channel_parts)):
            raise LoaderValidationError(f"upstream conversion count mismatch for {prefix}")
        converted: dict[str, Any] = {}
        target = torch.device(device)
        for (native_key, q_part), (scale_key, relative_part), (channel_key, channel_part) in zip(
            q_parts, relative_parts, channel_parts
        ):
            if native_key != scale_key or native_key != channel_key or not native_key.endswith(".weight"):
                raise LoaderValidationError(f"upstream conversion key mismatch for {prefix}")
            params = AsymW4A8Int8Layout.Params(
                scale=relative_part.to(target),
                s_channel=channel_part.to(target),
                correction=None,
                codebook=codebook.to(target),
                orig_dtype=dtype,
                orig_shape=(q_part.shape[0], q_part.shape[1] * 2),
                group_size=contract.group_size,
                convrot_groupsize=contract.convrot_groupsize,
            )
            converted[native_key] = QuantizedTensor(
                q_part.to(target), "AsymW4A8Int8Layout", params
            )
        return converted
