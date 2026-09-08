from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hayate.errors import SafetensorsInspectionError
from hayate.models.types import TensorInfo

MAX_HEADER_BYTES = 128 * 1024 * 1024

_DTYPE_BYTES: dict[str, float] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "C64": 8,
    "U64": 8,
    "I64": 8,
    "F64": 8,
    "C128": 16,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2FNUZ": 1,
}


@dataclass(frozen=True)
class SafetensorsHeader:
    header_size: int
    metadata: dict[str, Any]
    tensors: tuple[TensorInfo, ...]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SafetensorsInspectionError(f"duplicate JSON key in safetensors header: {key}")
        result[key] = value
    return result


def _validate_tensor(name: str, descriptor: Any, payload_size: int) -> TensorInfo:
    if not isinstance(descriptor, dict):
        raise SafetensorsInspectionError(f"tensor {name!r} descriptor must be an object")
    dtype = descriptor.get("dtype")
    shape = descriptor.get("shape")
    offsets = descriptor.get("data_offsets")
    if not isinstance(dtype, str) or not dtype:
        raise SafetensorsInspectionError(f"tensor {name!r} has invalid dtype")
    if not isinstance(shape, list) or not all(isinstance(v, int) and v >= 0 for v in shape):
        raise SafetensorsInspectionError(f"tensor {name!r} has invalid shape")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(v, int) for v in offsets)
    ):
        raise SafetensorsInspectionError(f"tensor {name!r} has invalid data_offsets")
    start, end = offsets
    if start < 0 or end < start or end > payload_size:
        raise SafetensorsInspectionError(
            f"tensor {name!r} offsets [{start}, {end}] exceed payload size {payload_size}"
        )
    bytes_per_element = _DTYPE_BYTES.get(dtype.upper())
    if bytes_per_element is not None:
        elements = math.prod(shape)
        expected = int(elements * bytes_per_element)
        if expected != end - start:
            raise SafetensorsInspectionError(
                f"tensor {name!r} byte size mismatch: header={end - start}, expected={expected}"
            )
    return TensorInfo(name, tuple(shape), dtype.upper(), start, end)


def read_safetensors_header(
    path: str | Path, *, max_header_bytes: int = MAX_HEADER_BYTES
) -> SafetensorsHeader:
    file_path = Path(path)
    try:
        file_size = file_path.stat().st_size
    except OSError as exc:
        raise SafetensorsInspectionError(f"cannot stat model file {file_path}: {exc}") from exc
    if file_size < 10:
        raise SafetensorsInspectionError(f"invalid safetensors file (too small): {file_path}")
    try:
        with file_path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise SafetensorsInspectionError("truncated safetensors length prefix")
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size < 2:
                raise SafetensorsInspectionError(f"invalid safetensors header length: {header_size}")
            if header_size > max_header_bytes:
                raise SafetensorsInspectionError(
                    f"safetensors header exceeds safety limit: {header_size} > {max_header_bytes}"
                )
            if 8 + header_size > file_size:
                raise SafetensorsInspectionError(
                    f"truncated safetensors header: need {8 + header_size} bytes, file has {file_size}"
                )
            raw_header = handle.read(header_size)
    except SafetensorsInspectionError:
        raise
    except OSError as exc:
        raise SafetensorsInspectionError(f"cannot read model file {file_path}: {exc}") from exc
    try:
        decoded = raw_header.decode("utf-8")
        parsed = json.loads(decoded, object_pairs_hook=_reject_duplicate_keys)
    except SafetensorsInspectionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetensorsInspectionError(f"corrupt safetensors JSON header: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SafetensorsInspectionError("safetensors header root must be an object")

    metadata = parsed.pop("__metadata__", {})
    if not isinstance(metadata, dict):
        raise SafetensorsInspectionError("safetensors __metadata__ must be an object")
    payload_size = file_size - 8 - header_size
    tensors = tuple(
        _validate_tensor(name, descriptor, payload_size)
        for name, descriptor in parsed.items()
    )
    ordered = sorted(tensors, key=lambda tensor: (tensor.data_start, tensor.data_end, tensor.name))
    previous_end = 0
    for tensor in ordered:
        if tensor.data_start < previous_end:
            raise SafetensorsInspectionError(
                f"overlapping tensor payload detected at {tensor.name!r}"
            )
        previous_end = tensor.data_end
    return SafetensorsHeader(header_size, metadata, tensors)

