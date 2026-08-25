from __future__ import annotations

import struct

import pytest

from hayate.errors import SafetensorsInspectionError
from hayate.models import ModelInspector, QuantizationFormat

from .helpers import write_dummy_safetensors


def test_tensor_header_parsing_without_payload_materialization(tmp_path):
    path = write_dummy_safetensors(
        tmp_path / "model.safetensors",
        [("layer.weight", "F32", [2, 3]), ("layer.bias", "F32", [2])],
        {"source": "test"},
    )
    report = ModelInspector().inspect(path)
    assert report.tensor_count == 2
    assert report.tensors[0].shape == (2, 3)
    assert report.tensor_storage_bytes == 32
    assert report.metadata == {"source": "test"}
    assert report.quantization.format is QuantizationFormat.FP32


def test_invalid_safetensors_is_a_clear_error(tmp_path):
    path = tmp_path / "invalid.safetensors"
    path.write_bytes(struct.pack("<Q", 200) + b"{}")
    with pytest.raises(SafetensorsInspectionError, match="truncated safetensors header"):
        ModelInspector().inspect(path)


@pytest.mark.parametrize(
    ("metadata", "tensors", "expected"),
    [
        ({"quantization": "W4A8 mixed"}, [("w", "U8", [4])], QuantizationFormat.W4A8_MIXED),
        (
            {"quantization": "NVFP4", "method": "AWQ"},
            [("w", "U8", [4])],
            QuantizationFormat.NVFP4_AWQ,
        ),
        (
            {},
            [
                *[
                    item
                    for index in range(8)
                    for item in (
                        (f"layer.{index}.weight", "U8", [2, 2]),
                        (f"layer.{index}.weight_scale", "F8_E4M3", [2, 1]),
                        (f"layer.{index}.weight_scale_2", "F32", []),
                        (f"layer.{index}.comfy_quant", "U8", [8]),
                    )
                ],
                ("layer.0.pre_quant_scale", "BF16", [4]),
            ],
            QuantizationFormat.NVFP4_AWQ,
        ),
        (
            {},
            [
                ("layer.weight", "I8", [2, 2]),
                ("layer.weight_scale", "F32", [2]),
                ("layer.comfy_quant", "I32", [1]),
            ],
            QuantizationFormat.INT8_CONVROT,
        ),
        ({}, [("w", "BF16", [2, 2])], QuantizationFormat.BF16),
    ],
)
def test_quantization_detection_uses_header_evidence(tmp_path, metadata, tensors, expected):
    path = write_dummy_safetensors(tmp_path / f"{expected.value}.safetensors", tensors, metadata)
    assert ModelInspector().inspect(path).quantization.format is expected


def test_filename_alone_does_not_claim_w4a8(tmp_path):
    path = write_dummy_safetensors(
        tmp_path / "pretend_w4a8_mixed.safetensors", [("opaque", "U8", [8])], {}
    )
    assert ModelInspector().inspect(path).quantization.format is QuantizationFormat.UNKNOWN
