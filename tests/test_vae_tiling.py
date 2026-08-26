from __future__ import annotations

import sys
from types import ModuleType

import pytest

from hayate.backends.minimax_h3.vae_tiling import (
    install_vae_attention_override,
    install_vae_tiling_override,
    vae_tile_size_from_environment,
)


def test_vae_tile_size_environment_validation(monkeypatch):
    monkeypatch.setenv("HAYATE_VAE_TILE_SIZE", "512")
    assert vae_tile_size_from_environment() == 512

    monkeypatch.setenv("HAYATE_VAE_TILE_SIZE", "64")
    with pytest.raises(ValueError, match="greater than 64"):
        vae_tile_size_from_environment()

    monkeypatch.setenv("HAYATE_VAE_TILE_SIZE", "510")
    with pytest.raises(ValueError, match="multiple of 16"):
        vae_tile_size_from_environment()


def test_vae_tiling_override_configures_each_loaded_vae(monkeypatch):
    calls: list[dict] = []

    class FakeVae:
        def enable_tiling(self, **kwargs):
            calls.append(kwargs)

    model_loader = ModuleType("minimax_video.model_loader")
    model_loader.load_vae = lambda *_args, **_kwargs: FakeVae()
    package = ModuleType("minimax_video")
    package.model_loader = model_loader
    monkeypatch.setitem(sys.modules, "minimax_video", package)
    monkeypatch.setitem(sys.modules, "minimax_video.model_loader", model_loader)

    assert install_vae_tiling_override(512) is True
    assert install_vae_tiling_override(512) is False
    with pytest.raises(RuntimeError, match="already installed"):
        install_vae_tiling_override(320)
    model_loader.load_vae("checkpoint", "cuda")
    assert calls == [
        {
            "tile_sample_min_height": 512,
            "tile_sample_min_width": 512,
        }
    ]


def test_vae_attention_override_scopes_video_and_audio_to_sdpa(monkeypatch):
    calls: list[tuple[str, str]] = []

    class FakeVae:
        def __init__(self, component: str):
            self.component = component

        def set_attention_backend(self, backend: str):
            calls.append((self.component, backend))

    model_loader = ModuleType("minimax_video.model_loader")
    model_loader.load_vae = lambda *_args, **_kwargs: FakeVae("video")
    model_loader.load_audio_vae = lambda *_args, **_kwargs: FakeVae("audio")
    package = ModuleType("minimax_video")
    package.model_loader = model_loader
    monkeypatch.setitem(sys.modules, "minimax_video", package)
    monkeypatch.setitem(sys.modules, "minimax_video.model_loader", model_loader)

    assert install_vae_attention_override("sdpa") is True
    assert install_vae_attention_override("sdpa") is False
    with pytest.raises(RuntimeError, match="already installed"):
        install_vae_attention_override("sageattn")
    model_loader.load_vae("checkpoint", "cuda")
    model_loader.load_audio_vae("checkpoint", "cuda")
    assert calls == [("video", "sdpa"), ("audio", "sdpa")]
