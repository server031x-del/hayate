from __future__ import annotations

import json

from hayate.loaders import (
    INT8ConvRotLoader,
    LoaderFactory,
    LoaderStatus,
    NVFP4AWQLoader,
    StandardSafetensorsLoader,
    W4A8Loader,
)
from hayate.models import ModelRole, ModelSpec, QuantizationFormat
from hayate.backends.minimax_h3.nvfp4_conditioner import _is_nvfp4_awq

from .helpers import write_dummy_safetensors


def test_w4a8_routes_to_dedicated_partial_loader_when_layout_is_missing(tmp_path):
    path = write_dummy_safetensors(
        tmp_path / "dit.safetensors",
        [("blocks.0.qweight", "U8", [4]), ("blocks.0.scale", "F32", [2])],
        {"quantization": "W4A8 mixed"},
    )
    loader = LoaderFactory.create(ModelSpec("minimax_h3", ModelRole.TRANSFORMER, path))
    validation = loader.validate()
    assert isinstance(loader, W4A8Loader)
    assert validation.status is LoaderStatus.PARTIAL
    assert "AsymW4A8Int8Layout" in validation.details["required_kernel"]


def test_registry_expectation_does_not_override_unknown_header(tmp_path):
    path = write_dummy_safetensors(tmp_path / "dit.safetensors", [("w", "U8", [4])], {})
    loader = LoaderFactory.create(
        ModelSpec(
            "minimax_h3",
            ModelRole.TRANSFORMER,
            path,
            expected_quantization=QuantizationFormat.W4A8_MIXED,
        )
    )
    assert loader.validate().status is LoaderStatus.UNSUPPORTED


def test_w4a8_validates_packed_scale_and_codebook_shape_contract(tmp_path):
    prefix = "blocks.0.attn.qkv_proj"
    metadata = {
        "_quantization_metadata": json.dumps(
            {
                "layers": {
                    prefix: {
                        "format": "asym_w4a8_int8",
                        "group_size": 16,
                        "convrot": True,
                        "convrot_groupsize": 256,
                    }
                }
            }
        )
    }
    path = write_dummy_safetensors(
        tmp_path / "w4a8.safetensors",
        [
            (prefix + ".weight", "I8", [2, 128]),
            (prefix + ".weight_s_rel", "F8_E4M3", [2, 16]),
            (prefix + ".weight_s_channel", "F32", [2]),
            (prefix + ".weight_codebook", "F32", [16]),
        ],
        metadata,
    )
    validation = LoaderFactory.create(
        ModelSpec("minimax_h3", ModelRole.TRANSFORMER, path)
    ).validate()
    assert validation.status is LoaderStatus.SUPPORTED
    assert validation.details["quantized_layer_count"] == 1
    assert validation.details["valid_quantized_layer_count"] == 1
    assert validation.details["layout_error_count"] == 0


def test_target_formats_route_to_their_loader_classes(tmp_path):
    nvfp4_tensors = [
        item
        for index in range(8)
        for item in (
            (f"layer.{index}.weight", "U8", [2, 2]),
            (f"layer.{index}.weight_scale", "F8_E4M3", [2, 1]),
            (f"layer.{index}.weight_scale_2", "F32", []),
            (f"layer.{index}.comfy_quant", "U8", [8]),
        )
    ] + [("layer.0.pre_quant_scale", "BF16", [4])]
    nvfp4 = write_dummy_safetensors(tmp_path / "qwen.safetensors", nvfp4_tensors, {})
    convrot = write_dummy_safetensors(
        tmp_path / "video_vae.safetensors",
        [
            ("decoder.layer.weight", "I8", [2, 2]),
            ("decoder.layer.weight_scale", "F32", [2]),
            ("decoder.layer.comfy_quant", "U8", [8]),
        ],
        {},
    )
    fp32 = write_dummy_safetensors(
        tmp_path / "audio_vae.safetensors", [("decoder.weight", "F32", [2, 2])], {}
    )

    assert isinstance(
        LoaderFactory.create(ModelSpec("minimax_h3", ModelRole.TEXT_ENCODER, nvfp4)),
        NVFP4AWQLoader,
    )
    assert _is_nvfp4_awq(nvfp4)
    assert isinstance(
        LoaderFactory.create(ModelSpec("minimax_h3", ModelRole.VIDEO_VAE, convrot)),
        INT8ConvRotLoader,
    )
    assert isinstance(
        LoaderFactory.create(ModelSpec("minimax_h3", ModelRole.AUDIO_VAE, fp32)),
        StandardSafetensorsLoader,
    )
