from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from hayate.errors import LoaderValidationError


class CheckpointTensorSource:
    """Own a safetensors source and drain pread tensors into their destination model."""

    def __init__(self, path: str | Path, *, backend: str | None = None):
        self.path = Path(path).expanduser().resolve(strict=False)
        selected = backend or os.environ.get("HAYATE_SAFETENSORS_BACKEND")
        self.backend = selected or ("pread" if os.name == "nt" else "mmap")
        if self.backend not in {"mmap", "pread"}:
            raise LoaderValidationError(
                "HAYATE_SAFETENSORS_BACKEND must be 'mmap' or 'pread'"
            )
        self._handle: Any | None = None
        self._context: Any | None = None
        self._tensors: dict[str, Any] | None = None
        try:
            if self.backend == "pread":
                from safetensors.torch import load_file

                self._tensors = load_file(self.path, device="cpu", backend="pread")
            else:
                from safetensors import safe_open

                self._context = safe_open(
                    self.path, framework="pt", device="cpu", backend="mmap"
                )
                self._handle = self._context.__enter__()
        except (ImportError, OSError, TypeError, ValueError) as exc:
            raise LoaderValidationError(
                f"cannot open checkpoint with {self.backend} backend: {self.path}: {exc}"
            ) from exc

    def keys(self) -> tuple[str, ...]:
        if self._tensors is not None:
            return tuple(self._tensors)
        if self._handle is None:
            raise RuntimeError("checkpoint source is closed")
        return tuple(self._handle.keys())

    def take_tensor(self, key: str, *, copy_mmap: bool = False):
        if self._tensors is not None:
            try:
                return self._tensors.pop(key)
            except KeyError as exc:
                raise KeyError(f"checkpoint tensor already consumed or missing: {key}") from exc
        if self._handle is None:
            raise RuntimeError("checkpoint source is closed")
        value = self._handle.get_tensor(key)
        return value.clone() if copy_mmap else value

    @property
    def remaining(self) -> int:
        return len(self._tensors) if self._tensors is not None else 0

    def close(self) -> None:
        if self._tensors is not None:
            self._tensors.clear()
            self._tensors = None
        if self._context is not None:
            self._context.__exit__(None, None, None)
            self._context = None
            self._handle = None

    def __enter__(self) -> "CheckpointTensorSource":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

