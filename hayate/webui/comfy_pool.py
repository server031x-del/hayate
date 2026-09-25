"""WebUI-owned, loopback-only ComfyUI processes kept warm between jobs."""
from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import TextIO

import psutil

from hayate.runtime.gpu_lease import GPULease


@dataclass
class _Session:
    process: subprocess.Popen
    log: TextIO
    lease: GPULease
    runtime: Path
    base_url: str
    input_dir: Path
    output_dir: Path
    log_path: Path


class WarmComfyPool:
    """Keep one ComfyUI process and runtime lease per physical GPU."""

    def __init__(self):
        self._sessions: dict[str, _Session] = {}
        self._lock = RLock()

    @staticmethod
    def _stop(session: _Session) -> None:
        try:
            if session.process.poll() is None:
                parent = psutil.Process(session.process.pid)
                children = parent.children(recursive=True)
                for child in reversed(children):
                    child.terminate()
                parent.terminate()
                _, alive = psutil.wait_procs([*children, parent], timeout=10)
                for item in alive:
                    item.kill()
                if alive:
                    psutil.wait_procs(alive, timeout=5)
                session.process.wait(timeout=5)
        except (psutil.Error, OSError, subprocess.TimeoutExpired):
            if session.process.poll() is None:
                session.process.kill()
                session.process.wait()
        finally:
            session.log.close()
            session.lease.release()

    def has_live(self, gpu_id: str) -> bool:
        with self._lock:
            session = self._sessions.get(gpu_id)
            if session and session.process.poll() is not None:
                self._stop(self._sessions.pop(gpu_id))
                return False
            return session is not None

    def discard(self, gpu_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(gpu_id, None)
            if session is not None:
                self._stop(session)

    def close(self) -> None:
        with self._lock:
            for gpu_id in tuple(self._sessions):
                self._stop(self._sessions.pop(gpu_id))

    def ensure(self, gpu_id: str, runtime: Path, python: str, root: Path,
               environment: dict[str, str]) -> dict[str, str]:
        """Start if needed and return the private connection settings."""
        with self._lock:
            runtime = runtime.resolve()
            session = self._sessions.get(gpu_id)
            reused = session is not None and session.process.poll() is None and session.runtime == runtime
            if session is not None and not reused:
                self._stop(self._sessions.pop(gpu_id))
                session = None
            if session is None:
                suffix = hashlib.sha256(gpu_id.encode()).hexdigest()[:16]
                directory = root / "data" / "comfy-warm" / suffix
                input_dir = directory / "input"
                output_dir = directory / "output"
                input_dir.mkdir(parents=True, exist_ok=True)
                output_dir.mkdir(parents=True, exist_ok=True)
                log_path = directory / "comfy.log"
                lease = GPULease(gpu_id=gpu_id, namespace="runtime")
                if not lease.acquire():
                    raise RuntimeError("GPU runtime is already owned by another process")
                log = None
                try:
                    with socket.socket() as sock:
                        sock.bind(("127.0.0.1", 0))
                        port = sock.getsockname()[1]
                    base_url = f"http://127.0.0.1:{port}"
                    log = log_path.open("w", encoding="utf-8")
                    args = [python, "-u", str(runtime / "main.py"), "--listen", "127.0.0.1",
                            "--port", str(port), "--disable-auto-launch",
                            "--output-directory", str(output_dir), "--input-directory", str(input_dir),
                            "--extra-model-paths-config", str(runtime / "hayate-models.yaml"),
                            "--enable-dynamic-vram", "--disable-pinned-memory",
                            "--async-offload", "2", "--cache-lru", "2"]
                    process = subprocess.Popen(args, cwd=runtime, env=environment,
                                               stdout=log, stderr=subprocess.STDOUT)
                    session = _Session(process, log, lease, runtime, base_url,
                                       input_dir, output_dir, log_path)
                    deadline = time.monotonic() + 180
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError("ComfyUI startup failed: " + log_path.read_text(errors="replace")[-3000:])
                        try:
                            with urllib.request.urlopen(base_url + "/object_info", timeout=5) as response:
                                json.load(response)
                            break
                        except (OSError, ValueError):
                            if time.monotonic() >= deadline:
                                raise TimeoutError("ComfyUI startup timeout: " + log_path.read_text(errors="replace")[-3000:])
                            time.sleep(1)
                    self._sessions[gpu_id] = session
                except BaseException:
                    if session is not None:
                        self._stop(session)
                    else:
                        if log is not None:
                            log.close()
                        lease.release()
                    raise
            return {
                "HAYATE_COMFY_BASE_URL": session.base_url,
                "HAYATE_COMFY_INPUT_DIR": str(session.input_dir),
                "HAYATE_COMFY_OUTPUT_DIR": str(session.output_dir),
                "HAYATE_COMFY_SERVER_LOG": str(session.log_path),
                "HAYATE_COMFY_REUSED": "1" if reused else "0",
            }
