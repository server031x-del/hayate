from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hayate.errors import HayateError
from hayate.loaders.base import BaseModelLoader, LoaderValidation
from hayate.loaders.factory import LoaderFactory
from hayate.models.registry import ModelRegistry
from hayate.models.types import ModelInspection, ModelSpec


@dataclass
class ModelRuntimeResult:
    spec: ModelSpec
    inspection: ModelInspection | None = None
    loader: BaseModelLoader | None = None
    validation: LoaderValidation | None = None
    error: str | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self, *, include_tensors: bool = False) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "inspection": (
                self.inspection.to_dict(include_tensors=include_tensors)
                if self.inspection
                else None
            ),
            "loader": type(self.loader).__name__ if self.loader else None,
            "validation": self.validation.to_dict() if self.validation else None,
            "error": self.error,
            "warnings": list(self.warnings),
        }


class HayateRuntime:
    """HAYATE extension layer; MiniMax generation remains an upstream backend."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def inspect_model(self, spec: ModelSpec, hardware: Any | None = None) -> ModelRuntimeResult:
        if not spec.exists:
            return ModelRuntimeResult(spec=spec, error=f"model file missing: {spec.path}")
        try:
            loader = LoaderFactory.create(spec)
            inspection = loader.inspect()
            validation = loader.validate(hardware)
            warnings: list[str] = []
            if (
                spec.expected_quantization is not None
                and inspection.quantization.format is not spec.expected_quantization
            ):
                warnings.append(
                    "registry expected "
                    f"{spec.expected_quantization.value}, header detected "
                    f"{inspection.quantization.format.value}"
                )
            return ModelRuntimeResult(
                spec=spec,
                inspection=inspection,
                loader=loader,
                validation=validation,
                warnings=tuple(warnings),
            )
        except (HayateError, OSError, ValueError) as exc:
            return ModelRuntimeResult(spec=spec, error=str(exc))

    def inspect_all(self, family: str, hardware: Any | None = None) -> list[ModelRuntimeResult]:
        return [self.inspect_model(spec, hardware) for spec in self.registry.iter_models(family)]

