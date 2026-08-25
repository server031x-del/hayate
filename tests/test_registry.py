from __future__ import annotations

import pytest

from hayate.errors import ModelRegistryError
from hayate.models import ModelRegistry, ModelRole


def test_registry_resolves_paths_relative_to_config(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = config_dir / "models.yaml"
    config.write_text(
        "models:\n  minimax_h3:\n    transformer:\n      path: ../models/dit.safetensors\n",
        encoding="utf-8",
    )
    spec = ModelRegistry.load(config).get("minimax_h3", ModelRole.TRANSFORMER)
    assert spec.path == (tmp_path / "models" / "dit.safetensors").resolve()


def test_registry_reports_missing_model_without_copying(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text(
        "models:\n  minimax_h3:\n    audio_vae:\n      path: missing.safetensors\n",
        encoding="utf-8",
    )
    spec = ModelRegistry.load(config).get("minimax_h3", "audio_vae")
    assert not spec.exists


def test_registry_rejects_unknown_role(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text("models:\n  minimax_h3:\n    mystery:\n      path: x\n", encoding="utf-8")
    with pytest.raises(ModelRegistryError, match="unknown model role"):
        ModelRegistry.load(config)

