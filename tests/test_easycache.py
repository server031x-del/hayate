from __future__ import annotations

from types import SimpleNamespace

import torch

from hayate.backends.minimax_h3.easycache import (
    EasyCacheConfig,
    EasyCacheController,
    install_easycache_override,
    install_transformer_easycache,
)


def test_easycache_skips_smooth_steps_and_preserves_caller_inputs():
    controller = EasyCacheController(
        EasyCacheConfig(enabled=True, threshold=0.2, start=0.0, end=1.0), total_steps=6
    )
    actual_calls = 0

    def transformer(*, hidden_states, audio_hidden_states, return_dict):
        nonlocal actual_calls
        actual_calls += 1
        return hidden_states + 2.0, audio_hidden_states + 3.0

    last_video = None
    last_audio = None
    for step in range(6):
        video = torch.full((1, 4, 2), step * 0.01)
        audio = torch.full((1, 3, 2), step * 0.02)
        video_before = video.clone()
        audio_before = audio.clone()
        last_video, last_audio = controller.forward(
            transformer,
            (),
            {
                "hidden_states": video,
                "audio_hidden_states": audio,
                "return_dict": False,
            },
        )
        assert torch.equal(video, video_before)
        assert torch.equal(audio, audio_before)

    assert actual_calls < 6
    assert controller.skipped_calls > 0
    assert torch.allclose(last_video, video + 2.0)
    assert torch.allclose(last_audio, audio + 3.0)


def test_easycache_disabled_window_runs_every_step():
    controller = EasyCacheController(
        EasyCacheConfig(enabled=True, threshold=1.0, start=0.4, end=0.6), total_steps=2
    )
    calls = 0

    def transformer(*, hidden_states, audio_hidden_states, return_dict):
        nonlocal calls
        calls += 1
        return hidden_states, audio_hidden_states

    for _ in range(2):
        tensor = torch.ones((1, 2, 2))
        controller.forward(
            transformer,
            (),
            {"hidden_states": tensor, "audio_hidden_states": tensor, "return_dict": False},
        )
    assert calls == 2


def test_easycache_does_not_seed_rate_before_start_window():
    controller = EasyCacheController(
        EasyCacheConfig(enabled=True, threshold=1.0, start=0.5, end=1.0), total_steps=5
    )

    def transformer(*, hidden_states, audio_hidden_states, return_dict):
        return hidden_states + 1.0, audio_hidden_states + 1.0

    for step in range(3):
        tensor = torch.full((1, 2, 2), float(step))
        controller.forward(
            transformer,
            (),
            {"hidden_states": tensor, "audio_hidden_states": tensor, "return_dict": False},
        )

    assert controller._previous_input is not None
    assert torch.equal(controller._previous_input, torch.full((1, 2, 2), 2.0))
    assert controller._change_ratio is None


def test_easycache_caps_consecutive_audio_and_video_reuse():
    controller = EasyCacheController(
        EasyCacheConfig(
            enabled=True,
            threshold=10.0,
            start=0.0,
            end=1.0,
            max_consecutive_skips=2,
        ),
        total_steps=8,
    )
    calls = 0

    def transformer(*, hidden_states, audio_hidden_states, return_dict):
        nonlocal calls
        calls += 1
        return hidden_states + 1.0, audio_hidden_states + 1.0

    for step in range(8):
        tensor = torch.full((1, 2, 2), step * 0.01)
        controller.forward(
            transformer,
            (),
            {"hidden_states": tensor, "audio_hidden_states": tensor, "return_dict": False},
        )

    assert controller.max_observed_consecutive_skips == 2
    assert controller.full_calls >= 4


def test_easycache_recalibrates_with_full_to_full_input_delta():
    controller = EasyCacheController(
        EasyCacheConfig(
            enabled=True,
            threshold=100.0,
            start=0.0,
            end=1.0,
            max_consecutive_skips=2,
        ),
        total_steps=5,
    )

    def transformer(*, hidden_states, audio_hidden_states, return_dict):
        return hidden_states * 5.0, audio_hidden_states * 5.0

    for value in (1.0, 2.0, 3.0, 4.0, 5.0):
        tensor = torch.full((1, 2, 2), value)
        controller.forward(
            transformer,
            (),
            {"hidden_states": tensor, "audio_hidden_states": tensor, "return_dict": False},
        )

    assert controller.full_calls == 3
    assert controller.skipped_calls == 2
    assert controller._change_ratio == 5.0


def test_disabled_easycache_controller_never_skips():
    controller = EasyCacheController(
        EasyCacheConfig(enabled=False, threshold=100.0, start=0.0, end=1.0), total_steps=5
    )
    calls = 0

    def transformer(*, hidden_states, audio_hidden_states, return_dict):
        nonlocal calls
        calls += 1
        return hidden_states, audio_hidden_states

    for _ in range(5):
        tensor = torch.ones((1, 2, 2))
        controller.forward(
            transformer,
            (),
            {"hidden_states": tensor, "audio_hidden_states": tensor, "return_dict": False},
        )

    assert calls == 5
    assert controller.stats()["enabled"] is False


def test_transformer_installer_binds_instance_forward():
    class Transformer:
        def forward(self, *, hidden_states, audio_hidden_states, return_dict):
            return hidden_states + 1.0, audio_hidden_states + 2.0

    transformer = Transformer()
    controller = install_transformer_easycache(
        transformer,
        EasyCacheConfig(enabled=True, start=0.0, end=1.0),
        total_steps=2,
    )
    tensor = torch.ones((1, 2, 2))
    video, audio = transformer.forward(
        hidden_states=tensor, audio_hidden_states=tensor, return_dict=False
    )
    assert torch.equal(video, tensor + 1.0)
    assert torch.equal(audio, tensor + 2.0)
    assert transformer._hayate_easycache_controller is controller


def test_module_override_is_idempotent_and_preserves_loader_tuple():
    calls = 0

    class Transformer:
        def forward(self, *, hidden_states, audio_hidden_states, return_dict):
            return hidden_states, audio_hidden_states

    transformer = Transformer()

    def load_transformer_stage(args, task, device):
        nonlocal calls
        calls += 1
        return transformer, "loader"

    module = SimpleNamespace(load_transformer_stage=load_transformer_stage)
    config = EasyCacheConfig(enabled=True)
    install_easycache_override(module, config)
    first_wrapper = module.load_transformer_stage
    install_easycache_override(module, config)
    assert module.load_transformer_stage is first_wrapper
    result = module.load_transformer_stage(SimpleNamespace(infer_steps=20), "t2va", "cuda:0")
    assert result == (transformer, "loader")
    assert calls == 1
    assert module._hayate_easycache_controller.total_steps == 19
