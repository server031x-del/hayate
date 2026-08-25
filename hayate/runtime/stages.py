from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Protocol


class Releasable(Protocol):
    def unload(self) -> None: ...


class Stage(str, Enum):
    PROMPT_ENCODING = "prompt_encoding"
    DENOISING = "denoising"
    DECODE = "decode"
    OUTPUT_ENCODE = "output_encode"


_ORDER = {
    Stage.PROMPT_ENCODING: 1,
    Stage.DENOISING: 2,
    Stage.DECODE: 3,
    Stage.OUTPUT_ENCODE: 4,
}


@dataclass(frozen=True)
class StageEvent:
    stage: Stage
    started_at: float
    ended_at: float
    released_resources: int

    @property
    def duration_seconds(self) -> float:
        return self.ended_at - self.started_at


class StageRuntime:
    """Own resources by stage while leaving MiniMax generation math upstream."""

    def __init__(self):
        self._active: Stage | None = None
        self._last_completed: Stage | None = None
        self._resources: dict[Stage, list[Releasable]] = {stage: [] for stage in Stage}
        self.events: list[StageEvent] = []

    def register(self, stage: Stage, resource: Releasable) -> None:
        if self._active is not stage:
            raise RuntimeError(f"cannot register {stage.value} resource while stage is not active")
        self._resources[stage].append(resource)

    def release(self, stage: Stage) -> int:
        resources = self._resources[stage]
        released = 0
        while resources:
            resource = resources.pop()
            resource.unload()
            released += 1
        return released

    @contextmanager
    def stage(self, stage: Stage) -> Iterator["StageRuntime"]:
        if self._active is not None:
            raise RuntimeError(f"stage {self._active.value} is already active")
        if self._last_completed is not None and _ORDER[stage] <= _ORDER[self._last_completed]:
            raise RuntimeError(
                f"invalid stage order: {stage.value} after {self._last_completed.value}"
            )
        self._active = stage
        started = time.perf_counter()
        released = 0
        try:
            yield self
        finally:
            try:
                released = self.release(stage)
            finally:
                ended = time.perf_counter()
                self.events.append(StageEvent(stage, started, ended, released))
                self._active = None
                self._last_completed = stage

