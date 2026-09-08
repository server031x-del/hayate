from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ModelRole(str, Enum):
    TRANSFORMER = "transformer"
    TEXT_ENCODER = "text_encoder"
    VIDEO_VAE = "video_vae"
    AUDIO_VAE = "audio_vae"


class QuantizationFormat(str, Enum):
    W4A8_MIXED = "W4A8_MIXED"
    NVFP4_AWQ = "NVFP4_AWQ"
    INT8_CONVROT = "INT8_CONVROT"
    FP32 = "FP32"
    FP16 = "FP16"
    BF16 = "BF16"
    FP8 = "FP8"
    UNKNOWN = "UNKNOWN"


class DetectionConfidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


@dataclass(frozen=True)
class QuantizationDetection:
    format: QuantizationFormat
    confidence: DetectionConfidence
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format.value,
            "confidence": self.confidence.value,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    data_start: int
    data_end: int

    @property
    def storage_bytes(self) -> int:
        return self.data_end - self.data_start

    @property
    def element_count(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "storage_bytes": self.storage_bytes,
            "data_offsets": [self.data_start, self.data_end],
        }


@dataclass(frozen=True)
class ModelSpec:
    family: str
    role: ModelRole
    path: Path
    expected_quantization: QuantizationFormat | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def exists(self) -> bool:
        return self.path.is_file()

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "role": self.role.value,
            "path": str(self.path),
            "exists": self.exists,
            "expected_quantization": (
                self.expected_quantization.value if self.expected_quantization else None
            ),
            "options": self.options,
        }


@dataclass(frozen=True)
class ModelInspection:
    path: Path
    file_size: int
    header_size: int
    metadata: dict[str, Any]
    tensors: tuple[TensorInfo, ...]
    quantization: QuantizationDetection
    estimated_ram_bytes: int
    estimated_vram_bytes: int
    warnings: tuple[str, ...] = ()

    @property
    def tensor_count(self) -> int:
        return len(self.tensors)

    @property
    def tensor_storage_bytes(self) -> int:
        return sum(t.storage_bytes for t in self.tensors)

    @property
    def dtypes(self) -> tuple[str, ...]:
        return tuple(sorted({tensor.dtype for tensor in self.tensors}))

    def to_dict(self, *, include_tensors: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "path": str(self.path),
            "file_name": self.path.name,
            "file_size": self.file_size,
            "header_size": self.header_size,
            "tensor_count": self.tensor_count,
            "tensor_storage_bytes": self.tensor_storage_bytes,
            "dtypes": list(self.dtypes),
            "metadata": self.metadata,
            "quantization": self.quantization.to_dict(),
            "estimated_ram_bytes": self.estimated_ram_bytes,
            "estimated_vram_bytes": self.estimated_vram_bytes,
            "warnings": list(self.warnings),
        }
        if include_tensors:
            result["tensors"] = [tensor.to_dict() for tensor in self.tensors]
        return result

