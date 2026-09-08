from __future__ import annotations

from pathlib import Path

from hayate.models.quantization import detect_quantization
from hayate.models.safetensors_header import read_safetensors_header
from hayate.models.types import ModelInspection


class ModelInspector:
    """Inspect safetensors metadata without mapping or materializing tensor payloads."""

    def inspect(self, path: str | Path) -> ModelInspection:
        file_path = Path(path).expanduser().resolve(strict=False)
        file_size = file_path.stat().st_size
        header = read_safetensors_header(file_path)
        detection = detect_quantization(header.metadata, header.tensors)
        tensor_bytes = sum(tensor.storage_bytes for tensor in header.tensors)
        warnings: list[str] = []
        if detection.format.value == "UNKNOWN":
            warnings.append("quantization remains UNKNOWN; execution must not be guessed")
        # v0.1 estimates physical checkpoint residency. Kernel workspaces and
        # dequantized fallbacks are deliberately excluded until a loader supplies a plan.
        return ModelInspection(
            path=file_path,
            file_size=file_size,
            header_size=header.header_size,
            metadata=header.metadata,
            tensors=header.tensors,
            quantization=detection,
            estimated_ram_bytes=tensor_bytes,
            estimated_vram_bytes=tensor_bytes,
            warnings=tuple(warnings),
        )

