from __future__ import annotations

import importlib.util
from typing import Any

from hayate.errors import LoaderNotReadyError, LoaderValidationError
from hayate.loaders.base import BaseModelLoader, LoaderStatus, LoaderValidation
from hayate.models.types import QuantizationFormat


class StandardSafetensorsLoader(BaseModelLoader):
    format_name = "standard safetensors"
    _FORMATS = {
        QuantizationFormat.FP32,
        QuantizationFormat.FP16,
        QuantizationFormat.BF16,
        QuantizationFormat.FP8,
    }

    def validate(self, hardware: Any | None = None) -> LoaderValidation:
        report = self.inspect()
        if report.quantization.format not in self._FORMATS:
            return LoaderValidation(
                LoaderStatus.UNSUPPORTED,
                f"{report.quantization.format.value} is not a standard floating safetensors format",
            )
        missing = [name for name in ("torch", "safetensors.torch") if importlib.util.find_spec(name) is None]
        if missing:
            return LoaderValidation(
                LoaderStatus.PARTIAL,
                "header is supported but tensor materialization dependencies are unavailable",
                requirements=("install HAYATE with the cuda extra or provide a compatible torch",),
                details={"missing_modules": missing},
            )
        return LoaderValidation(
            LoaderStatus.SUPPORTED,
            "standard safetensors payload can be materialized explicitly",
            warnings=("loading materializes the full checkpoint and is not stage-budgeted",),
        )

    def load(self, device: str) -> Any:
        validation = self.validate()
        if validation.status is LoaderStatus.UNSUPPORTED:
            raise LoaderValidationError(validation.reason)
        if validation.status is LoaderStatus.PARTIAL:
            raise LoaderNotReadyError(validation.reason)
        from safetensors.torch import load_file

        self._loaded = load_file(str(self.spec.path), device=device)
        return self._loaded


class UnknownSafetensorsLoader(BaseModelLoader):
    format_name = "unknown"

    def validate(self, hardware: Any | None = None) -> LoaderValidation:
        report = self.inspect()
        return LoaderValidation(
            LoaderStatus.UNSUPPORTED,
            f"quantization format is {report.quantization.format.value}; execution is blocked safely",
            warnings=("add exporter documentation or explicit header metadata before enabling a loader",),
        )

    def load(self, device: str) -> Any:
        raise LoaderNotReadyError("UNKNOWN quantization cannot be loaded safely")

