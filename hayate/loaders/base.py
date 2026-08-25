from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hayate.models.inspector import ModelInspector
from hayate.models.types import ModelInspection, ModelSpec


class LoaderStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class LoaderValidation:
    status: LoaderStatus
    reason: str
    requirements: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "requirements": list(self.requirements),
            "warnings": list(self.warnings),
            "details": self.details,
        }


class BaseModelLoader(ABC):
    format_name = "base"

    def __init__(self, spec: ModelSpec, *, inspector: ModelInspector | None = None):
        self.spec = spec
        self.inspector = inspector or ModelInspector()
        self._inspection: ModelInspection | None = None
        self._loaded: Any = None

    def inspect(self) -> ModelInspection:
        if self._inspection is None:
            self._inspection = self.inspector.inspect(self.spec.path)
        return self._inspection

    @abstractmethod
    def validate(self, hardware: Any | None = None) -> LoaderValidation:
        """Validate structure and execution compatibility without loading weights."""

    @abstractmethod
    def load(self, device: str) -> Any:
        """Load the model payload on explicit request."""

    def unload(self) -> None:
        self._loaded = None
        gc.collect()

