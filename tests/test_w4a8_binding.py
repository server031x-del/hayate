from __future__ import annotations

import json
import sys
from pathlib import Path

from hayate.loaders.w4a8_binding import W4A8CheckpointBinding

from .helpers import write_dummy_safetensors


def test_w4a8_checkpoint_bundle_reconstructs_quantized_tensor(tmp_path):
    prefix = "blocks.0.attn.out_proj"
    path = tmp_path / "w4a8.safetensors"
    quantization = {
        "layers": {
            prefix: {
                "format": "asym_w4a8_int8",
                "group_size": 16,
                "convrot": True,
                "convrot_groupsize": 256,
            }
        }
    }
    write_dummy_safetensors(
        path,
        [
            (prefix + ".weight", "I8", [4, 128]),
            (prefix + ".weight_s_rel", "F8_E4M3", [4, 16]),
            (prefix + ".weight_s_channel", "F32", [4]),
            (prefix + ".weight_codebook", "F32", [16]),
        ],
        {"_quantization_metadata": json.dumps(quantization)},
    )

    binding = W4A8CheckpointBinding(path)
    tensor = binding.materialize(prefix)

    assert binding.layer_prefixes == (prefix,)
    assert tuple(tensor.shape) == (4, 256)
    assert tuple(tensor.storage_shape) == (4, 128)
    assert tensor._layout_cls == "AsymW4A8Int8Layout"
    assert tensor._params.group_size == 16
    assert tuple(tensor._params.codebook.shape) == (16,)


def test_w4a8_binding_reuses_upstream_qkv_row_conversion(tmp_path):
    checkout = Path(r"M:\CodexHome\tmp\hayate-upstream-h3")
    if not checkout.is_dir():
        return
    engine = checkout / "minimax_engine"
    sys.path.insert(0, str(engine))
    try:
        from minimax_video.int8_quant import convert_int8_dit_tensor
    finally:
        sys.path.remove(str(engine))
    prefix = "blocks.0.attn.qkv_proj"
    path = tmp_path / "qkv.safetensors"
    metadata = {
        "layers": {
            prefix: {
                "format": "asym_w4a8_int8",
                "group_size": 16,
                "convrot": True,
                "convrot_groupsize": 256,
            }
        }
    }
    write_dummy_safetensors(
        path,
        [
            (prefix + ".weight", "I8", [12, 128]),
            (prefix + ".weight_s_rel", "F8_E4M3", [12, 16]),
            (prefix + ".weight_s_channel", "F32", [12]),
            (prefix + ".weight_codebook", "F32", [16]),
        ],
        {"_quantization_metadata": json.dumps(metadata)},
    )

    converted = W4A8CheckpointBinding(path).materialize_converted(
        prefix, convert_int8_dit_tensor
    )

    assert tuple(converted) == (
        "transformer_blocks.0.attn.to_q.weight",
        "transformer_blocks.0.attn.to_k.weight",
        "transformer_blocks.0.attn.to_v.weight",
    )
    assert all(tuple(tensor.shape) == (4, 256) for tensor in converted.values())
