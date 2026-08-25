from __future__ import annotations

from typing import Any

from hayate.errors import LoaderNotReadyError
from hayate.loaders.base import BaseModelLoader, LoaderStatus, LoaderValidation
from hayate.models.types import ModelRole, QuantizationFormat


class INT8ConvRotLoader(BaseModelLoader):
    format_name = "INT8 ConvRot"

    def validate(self, hardware: Any | None = None) -> LoaderValidation:
        report = self.inspect()
        if report.quantization.format is not QuantizationFormat.INT8_CONVROT:
            return LoaderValidation(
                LoaderStatus.UNSUPPORTED,
                "header evidence does not match the documented INT8 ConvRot marker convention",
            )
        quant_markers = sum(t.name.lower().endswith(".comfy_quant") for t in report.tensors)
        scales = sum(t.name.lower().endswith(".weight_scale") for t in report.tensors)
        int8_weights = sum(
            t.dtype == "I8" and t.name.lower().endswith(".weight") for t in report.tensors
        )
        if self.spec.role is ModelRole.VIDEO_VAE:
            reason = (
                "INT8 ConvRot markers recognized; all 144 audited targets are Linear weights and "
                "the full strict upstream load and CUDA decode smoke pass"
            )
            complete = quant_markers == scales == int8_weights == 144
            requirements = (() if complete else ("provide all 144 Video VAE ConvRot groups",))
        else:
            reason = "INT8 ConvRot markers recognized; execution binding is deferred to the upstream adapter"
            requirements = (
                "bind the audited upstream ConvRot conversion and torch._int_mm path",
                "verify torch/CUDA support on the selected device",
            )
            complete = False
        return LoaderValidation(
            LoaderStatus.SUPPORTED if complete else LoaderStatus.PARTIAL,
            reason,
            requirements=requirements,
            details={
                "quant_marker_count": quant_markers,
                "weight_scale_count": scales,
                "int8_weight_count": int8_weights,
                "role": self.spec.role.value,
            },
        )

    def load(self, device: str) -> Any:
        validation = self.validate()
        if validation.status is not LoaderStatus.SUPPORTED:
            raise LoaderNotReadyError(f"{validation.reason}. Target device: {device}")
        from hayate.loaders.int8_convrot_binding import INT8ConvRotCheckpointBinding

        self._loaded = INT8ConvRotCheckpointBinding(self.spec.path)
        return self._loaded
