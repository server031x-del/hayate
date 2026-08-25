from __future__ import annotations

import torch

from hayate.backends.minimax_h3.easycache import EasyCacheConfig, EasyCacheController


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
