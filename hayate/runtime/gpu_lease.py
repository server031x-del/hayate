from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import BinaryIO, Self

import psutil


class GPULease:
    """Cross-process advisory lease for one HAYATE inference resource.

    A GPU UUID creates a dedicated lock, allowing independent adapters to run
    concurrently while still preventing CLI/WebUI collisions on the same
    physical device.  The historical ``HAYATE_GPU_LEASE_PATH`` override and
    the global fallback remain supported for CPU tests and legacy callers.
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        owner: dict | None = None,
        gpu_id: str | None = None,
        namespace: str = "scheduler",
    ):
        configured = os.environ.get("HAYATE_GPU_LEASE_PATH")
        # The legacy override is intentionally global.  It is useful for
        # sandboxes/tests and remains an explicit escape hatch for operators
        # who want to serialize scheduler processes, even when a GPU UUID is
        # available.  Runtime leases stay UUID-scoped so a scheduler lease in
        # the same process never collides with its child-runtime probe.
        selected = path or (
            Path(configured)
            if configured and (not gpu_id or namespace == "scheduler")
            else None
        )
        if selected is None and gpu_id:
            digest = hashlib.sha256(gpu_id.encode("utf-8", errors="replace")).hexdigest()[:16]
            selected = Path(tempfile.gettempdir()) / f"hayate-gpu-{namespace}-{digest}.lock"
        elif selected is None and namespace != "scheduler":
            selected = Path(tempfile.gettempdir()) / f"hayate-gpu-{namespace}.lock"
        self.path = (
            selected or Path(tempfile.gettempdir()) / "hayate-gpu0.lock"
        ).resolve(strict=False)
        self.owner_path = self.path.with_suffix(self.path.suffix + ".owner.json")
        self.gpu_id = gpu_id
        self.namespace = namespace
        self.owner = {
            **(owner or self._default_owner()),
            **({"gpu_id": gpu_id, "lease_namespace": namespace} if gpu_id else {"lease_namespace": namespace}),
        }
        self._handle: BinaryIO | None = None

    @staticmethod
    def _default_owner() -> dict:
        process = psutil.Process()
        command = "\0".join(sys.argv)
        return {
            "pid": process.pid,
            "create_time": process.create_time(),
            "command_hash": hashlib.sha256(
                command.encode("utf-8", errors="replace")
            ).hexdigest(),
            "command": " ".join(sys.argv[:4]),
        }

    def _try_lock(self, handle: BinaryIO) -> bool:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return True
            except OSError:
                return False
        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def acquire(self, timeout: float = 0.0, poll_interval: float = 0.5) -> bool:
        if self._handle is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            handle = self.path.open("a+b")
            if self.path.stat().st_size == 0:
                handle.write(b"\0")
                handle.flush()
            if self._try_lock(handle):
                self._handle = handle
                metadata = {**self.owner, "acquired_at": time.time()}
                temporary = self.owner_path.with_suffix(
                    self.owner_path.suffix + f".{os.getpid()}.tmp"
                )
                temporary.write_text(
                    json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
                )
                os.replace(temporary, self.owner_path)
                return True
            handle.close()
            if time.monotonic() >= deadline:
                return False
            time.sleep(max(0.05, poll_interval))

    def busy_owner(self) -> dict | None:
        try:
            value = json.loads(self.owner_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            self.owner_path.unlink(missing_ok=True)
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None

    def __enter__(self) -> Self:
        if not self.acquire():
            owner = self.busy_owner() or {}
            raise RuntimeError(
                f"選択したGPUはHAYATEで使用中です (pid={owner.get('pid', '?')})"
            )
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.release()
