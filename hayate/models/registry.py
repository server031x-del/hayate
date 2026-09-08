from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterator

import yaml

from hayate.errors import ModelRegistryError
from hayate.models.types import ModelRole, ModelSpec, QuantizationFormat

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}|%([^%]+)%")


def _expand_path(raw: str, base_dir: Path) -> Path:
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        value = os.environ.get(name)
        if value is None:
            missing.append(name)
            return match.group(0)
        return value

    expanded = _ENV_PATTERN.sub(replace, raw)
    if missing:
        raise ModelRegistryError(
            f"model path references unset environment variable(s): {', '.join(sorted(set(missing)))}"
        )
    path = Path(os.path.expanduser(expanded))
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve(strict=False)


class ModelRegistry:
    def __init__(self, config_path: str | Path, models: dict[str, dict[ModelRole, ModelSpec]]):
        self.config_path = Path(config_path).resolve(strict=False)
        self._models = models

    @classmethod
    def load(cls, config_path: str | Path) -> "ModelRegistry":
        path = Path(config_path).expanduser().resolve(strict=False)
        try:
            with path.open("r", encoding="utf-8") as handle:
                document = yaml.safe_load(handle)
        except OSError as exc:
            raise ModelRegistryError(f"cannot read model registry {path}: {exc}") from exc
        except yaml.YAMLError as exc:
            raise ModelRegistryError(f"invalid model registry YAML {path}: {exc}") from exc
        if not isinstance(document, dict) or not isinstance(document.get("models"), dict):
            raise ModelRegistryError("model registry must contain a 'models' mapping")
        models: dict[str, dict[ModelRole, ModelSpec]] = {}
        for family, role_mapping in document["models"].items():
            if not isinstance(family, str) or not isinstance(role_mapping, dict):
                raise ModelRegistryError("each model family must map role names to entries")
            parsed_roles: dict[ModelRole, ModelSpec] = {}
            for role_name, entry in role_mapping.items():
                try:
                    role = ModelRole(role_name)
                except ValueError as exc:
                    raise ModelRegistryError(
                        f"unknown model role {role_name!r} under family {family!r}"
                    ) from exc
                if isinstance(entry, str):
                    entry = {"path": entry}
                if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                    raise ModelRegistryError(f"{family}.{role.value} must define a string path")
                expected = entry.get("expected_quantization")
                try:
                    expected_format = QuantizationFormat(expected) if expected else None
                except ValueError as exc:
                    raise ModelRegistryError(
                        f"unknown expected_quantization {expected!r} for {family}.{role.value}"
                    ) from exc
                options = {
                    key: value
                    for key, value in entry.items()
                    if key not in {"path", "expected_quantization"}
                }
                parsed_roles[role] = ModelSpec(
                    family=family,
                    role=role,
                    path=_expand_path(entry["path"], path.parent),
                    expected_quantization=expected_format,
                    options=options,
                )
            models[family] = parsed_roles
        return cls(path, models)

    def families(self) -> tuple[str, ...]:
        return tuple(self._models)

    def get(self, family: str, role: ModelRole | str) -> ModelSpec:
        try:
            parsed_role = role if isinstance(role, ModelRole) else ModelRole(role)
            return self._models[family][parsed_role]
        except (KeyError, ValueError) as exc:
            raise ModelRegistryError(f"model entry not found: {family}.{role}") from exc

    def iter_models(self, family: str | None = None) -> Iterator[ModelSpec]:
        families = (family,) if family else self.families()
        for family_name in families:
            if family_name not in self._models:
                raise ModelRegistryError(f"model family not found: {family_name}")
            yield from self._models[family_name].values()

