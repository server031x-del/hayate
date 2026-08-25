from __future__ import annotations

from hayate.loaders.base import BaseModelLoader
from hayate.loaders.convrot import INT8ConvRotLoader
from hayate.loaders.nvfp4_awq import NVFP4AWQLoader
from hayate.loaders.standard import StandardSafetensorsLoader, UnknownSafetensorsLoader
from hayate.loaders.w4a8 import W4A8Loader
from hayate.models.inspector import ModelInspector
from hayate.models.types import ModelSpec, QuantizationFormat


class LoaderFactory:
    _LOADERS: dict[QuantizationFormat, type[BaseModelLoader]] = {
        QuantizationFormat.W4A8_MIXED: W4A8Loader,
        QuantizationFormat.NVFP4_AWQ: NVFP4AWQLoader,
        QuantizationFormat.INT8_CONVROT: INT8ConvRotLoader,
        QuantizationFormat.FP32: StandardSafetensorsLoader,
        QuantizationFormat.FP16: StandardSafetensorsLoader,
        QuantizationFormat.BF16: StandardSafetensorsLoader,
        QuantizationFormat.FP8: StandardSafetensorsLoader,
        QuantizationFormat.UNKNOWN: UnknownSafetensorsLoader,
    }

    @classmethod
    def create(cls, spec: ModelSpec, *, inspector: ModelInspector | None = None) -> BaseModelLoader:
        selected_inspector = inspector or ModelInspector()
        report = selected_inspector.inspect(spec.path)
        loader_type = cls._LOADERS[report.quantization.format]
        loader = loader_type(spec, inspector=selected_inspector)
        loader._inspection = report
        return loader

