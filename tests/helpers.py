from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

_BYTES = {
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "I32": 4,
}


def write_dummy_safetensors(
    path: Path,
    tensors: list[tuple[str, str, list[int]]],
    metadata: dict[str, Any] | None = None,
) -> Path:
    header: dict[str, Any] = {}
    payload = bytearray()
    offset = 0
    for name, dtype, shape in tensors:
        size = math.prod(shape) * _BYTES[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        payload.extend(b"\0" * size)
        offset += size
    if metadata is not None:
        header["__metadata__"] = metadata
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return path
