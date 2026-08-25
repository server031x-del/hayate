from __future__ import annotations

import os
from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable


@dataclass(frozen=True)
class EasyCacheConfig:
    """Runtime-adaptive transformer cache settings.

    The algorithm follows the Apache-2.0 EasyCache reference design while the
    integration is specific to the external MiniMax H3 engine.
    """

    enabled: bool = False
    threshold: float = 0.2
    start: float = 0.15
    end: float = 0.95

    @classmethod
    def from_environment(cls) -> "EasyCacheConfig":
        enabled = os.environ.get("HAYATE_EASYCACHE", "").lower() in {"1", "true", "yes", "on"}
        return cls(
            enabled=enabled,
            threshold=float(os.environ.get("HAYATE_EASYCACHE_THRESHOLD", "0.2")),
            start=float(os.environ.get("HAYATE_EASYCACHE_START", "0.15")),
            end=float(os.environ.get("HAYATE_EASYCACHE_END", "0.95")),
        )

    def validate(self) -> None:
        if self.threshold < 0:
            raise ValueError("EasyCache threshold must be non-negative")
        if not 0 <= self.start < self.end <= 1:
            raise ValueError("EasyCache start/end must satisfy 0 <= start < end <= 1")


class EasyCacheController:
    """Cache MiniMax H3 video and audio transformer residuals without changing its pipeline."""

    def __init__(self, config: EasyCacheConfig, total_steps: int):
        config.validate()
        self.config = config
        self.total_steps = max(1, total_steps)
        self.calls = 0
        self.full_calls = 0
        self.skipped_calls = 0
        self.accumulated_error = 0.0
        self._previous_input = None
        self._previous_output = None
        self._video_residual = None
        self._audio_residual = None
        self._change_ratio: float | None = None

    @staticmethod
    def _mean_abs(tensor) -> float:
        return float(tensor.detach().abs().mean().item())

    def _active(self, step: int) -> bool:
        progress = step / max(1, self.total_steps - 1)
        return self.config.start <= progress <= self.config.end

    def _clear_runtime_state(self) -> None:
        self.accumulated_error = 0.0
        self._previous_input = None
        self._previous_output = None
        self._video_residual = None
        self._audio_residual = None
        self._change_ratio = None

    def _inputs(self, args: tuple[Any, ...], kwargs: dict[str, Any]):
        video = kwargs.get("hidden_states", args[0] if args else None)
        audio = kwargs.get("audio_hidden_states", args[1] if len(args) > 1 else None)
        return video, audio

    def forward(
        self,
        original_forward: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        video_input, audio_input = self._inputs(args, kwargs)
        return_dict = kwargs.get("return_dict", True)
        step = self.calls
        self.calls += 1

        # EasyCache establishes its transformation-rate baseline inside the
        # configured window. Early, high-noise steps must not seed that rate.
        if not self._active(step):
            self._clear_runtime_state()
            self.full_calls += 1
            return original_forward(*args, **kwargs)

        can_cache = (
            not return_dict
            and video_input is not None
            and audio_input is not None
            and self._previous_input is not None
            and self._previous_output is not None
            and self._video_residual is not None
            and self._audio_residual is not None
            and self._change_ratio is not None
        )
        input_change = None
        if self._previous_input is not None and video_input is not None:
            input_change = self._mean_abs(video_input - self._previous_input)

        if can_cache:
            output_norm = max(self._mean_abs(self._previous_output), 1e-12)
            predicted_change = self._change_ratio * (input_change / output_norm)
            self.accumulated_error += predicted_change
            if self.accumulated_error < self.config.threshold:
                self._previous_input = video_input.detach().clone()
                self.skipped_calls += 1
                print(
                    f"HAYATE_EASYCACHE_SKIP step={step + 1}/{self.total_steps} "
                    f"accumulated={self.accumulated_error:.6f}",
                    flush=True,
                )
                # Never mutate the caller-owned H3 tensors. In particular, the
                # audio scheduler carries a distinct noise schedule.
                return (
                    video_input + self._video_residual,
                    audio_input + self._audio_residual,
                )
            print(
                f"HAYATE_EASYCACHE_FULL step={step + 1}/{self.total_steps} "
                f"accumulated={self.accumulated_error:.6f}",
                flush=True,
            )
            self.accumulated_error = 0.0

        output = original_forward(*args, **kwargs)
        self.full_calls += 1
        if return_dict or not isinstance(output, tuple) or len(output) != 2:
            return output

        video_output, audio_output = output
        self._video_residual = (video_output - video_input).detach()
        self._audio_residual = (audio_output - audio_input).detach()
        if self._previous_output is not None and input_change is not None and input_change > 1e-12:
            output_change = self._mean_abs(video_output - self._previous_output)
            self._change_ratio = output_change / input_change
        self._previous_input = video_input.detach().clone()
        self._previous_output = video_output.detach().clone()
        return output

    def stats(self) -> dict[str, int | float]:
        return {
            "enabled": True,
            "threshold": self.config.threshold,
            "start": self.config.start,
            "end": self.config.end,
            "calls": self.calls,
            "full_calls": self.full_calls,
            "skipped_calls": self.skipped_calls,
        }


def install_transformer_easycache(transformer, config: EasyCacheConfig, total_steps: int):
    controller = EasyCacheController(config, total_steps)
    original_forward = transformer.forward

    def cached_forward(_self, *args, **kwargs):
        return controller.forward(original_forward, args, kwargs)

    transformer.forward = MethodType(cached_forward, transformer)
    transformer._hayate_easycache_controller = controller
    return controller


def install_easycache_override(module, config: EasyCacheConfig) -> None:
    if not config.enabled:
        return
    config.validate()
    if getattr(module, "_hayate_easycache_installed", False):
        return
    original = module.load_transformer_stage

    def wrapped_load_transformer_stage(args, task, device):
        result = original(args, task, device)
        transformer = result[0] if isinstance(result, tuple) else result
        total_steps = max(1, int(args.infer_steps) - 1)
        controller = install_transformer_easycache(transformer, config, total_steps)
        module._hayate_easycache_controller = controller
        print(
            "HAYATE_EASYCACHE_ENABLED "
            f"threshold={config.threshold} start={config.start} end={config.end} "
            f"steps={total_steps}",
            flush=True,
        )
        return result

    module.load_transformer_stage = wrapped_load_transformer_stage
    module._hayate_easycache_installed = True
