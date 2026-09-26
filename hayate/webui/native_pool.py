"""Persistent native H3 worker processes owned by the WebUI scheduler."""
from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import psutil

READY_PREFIX = "HAYATE_WORKER_READY "
RESULT_PREFIX = "HAYATE_WORKER_RESULT "


@dataclass
class _Session:
    process: subprocess.Popen[str]
    output: queue.Queue[str | None]
    upstream: Path
    fingerprint: str
    lock: threading.RLock = field(default_factory=threading.RLock)


class WarmNativePool:
    """Keep one model-caching native H3 worker per selected physical GPU."""

    def __init__(self, *, startup_timeout: float = 300.0):
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.RLock()
        self._startup_timeout = startup_timeout

    @staticmethod
    def _fingerprint(command: list[str], environment: dict[str, str], cwd: Path) -> str:
        relevant_environment = {
            key: value
            for key, value in environment.items()
            if key.startswith("HAYATE_")
            or key in {"CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF"}
        }
        payload = {
            "python": command[0],
            "upstream": command[command.index("--upstream") + 1],
            "cwd": str(cwd.resolve(strict=False)),
            "environment": relevant_environment,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _reader(process: subprocess.Popen[str], output: queue.Queue[str | None]) -> None:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                output.put(line)
        finally:
            output.put(None)

    @staticmethod
    def _stop(session: _Session, *, force: bool) -> None:
        process = session.process
        if process.poll() is not None:
            return
        if not force:
            try:
                if process.stdin is not None:
                    process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                    process.stdin.flush()
                process.wait(timeout=8)
                return
            except (OSError, subprocess.SubprocessError):
                force = True
        if force and process.poll() is None:
            try:
                parent = psutil.Process(process.pid)
                children = parent.children(recursive=True)
                for child in reversed(children):
                    child.terminate()
                parent.terminate()
                _, alive = psutil.wait_procs([*children, parent], timeout=5)
                for item in alive:
                    item.kill()
                if alive:
                    psutil.wait_procs(alive, timeout=5)
            except (psutil.Error, OSError):
                try:
                    process.kill()
                except OSError:
                    pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def has_live(self, gpu_id: str) -> bool:
        with self._lock:
            session = self._sessions.get(gpu_id)
            if session is not None and session.process.poll() is not None:
                self._sessions.pop(gpu_id, None)
                return False
            return session is not None

    def discard(self, gpu_id: str, *, force: bool = True) -> None:
        with self._lock:
            session = self._sessions.pop(gpu_id, None)
        if session is not None:
            self._stop(session, force=force)

    def close(self) -> None:
        with self._lock:
            gpu_ids = tuple(self._sessions)
        for gpu_id in gpu_ids:
            self.discard(gpu_id, force=False)

    def ensure(
        self,
        gpu_id: str,
        command: list[str],
        environment: dict[str, str],
        cwd: Path,
        startup_log: TextIO,
    ) -> _Session:
        """Reuse a matching worker or start one and wait for engine readiness."""
        upstream = Path(command[command.index("--upstream") + 1]).resolve(strict=False)
        fingerprint = self._fingerprint(command, environment, cwd)
        with self._lock:
            session = self._sessions.get(gpu_id)
            if session is not None and session.process.poll() is not None:
                self._sessions.pop(gpu_id, None)
                session = None
            if session is not None and session.upstream == upstream and session.fingerprint == fingerprint:
                startup_log.write(f"HAYATE_WORKER_REUSED pid={session.process.pid}\n")
                startup_log.flush()
                return session
            if session is not None:
                self._sessions.pop(gpu_id, None)
                self._stop(session, force=True)

            worker_environment = environment.copy()
            worker_environment["HAYATE_GPU_RUNTIME_LEASE"] = "1"
            worker_environment["PYTHONUNBUFFERED"] = "1"
            worker_command = [
                command[0], "-u", "-m", "hayate.backends.minimax_h3.entrypoint",
                "--upstream", str(upstream), "--persistent-worker",
            ]
            process = subprocess.Popen(
                worker_command,
                cwd=str(cwd),
                env=worker_environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                ),
            )
            output: queue.Queue[str | None] = queue.Queue()
            session = _Session(process, output, upstream, fingerprint)
            threading.Thread(
                target=self._reader,
                args=(process, output),
                name=f"hayate-native-output-{process.pid}",
                daemon=True,
            ).start()
            startup_lines: list[str] = []
            deadline = time.monotonic() + self._startup_timeout
            try:
                while time.monotonic() < deadline:
                    remaining = max(0.05, deadline - time.monotonic())
                    try:
                        line = output.get(timeout=min(1.0, remaining))
                    except queue.Empty:
                        if process.poll() is not None:
                            break
                        continue
                    if line is None:
                        break
                    if line.startswith(READY_PREFIX):
                        self._sessions[gpu_id] = session
                        startup_log.write("".join(startup_lines))
                        startup_log.write(line)
                        startup_log.flush()
                        return session
                    startup_lines.append(line)
                    startup_log.write(line)
                    startup_log.flush()
                tail = "".join(startup_lines)[-5000:]
                raise RuntimeError(
                    "native H3 worker did not become ready"
                    + (f": {tail}" if tail else "")
                )
            except BaseException:
                self._stop(session, force=True)
                raise

    def run_job(
        self,
        session: _Session,
        job_id: str,
        argv: list[str],
        log: TextIO,
        on_line,
    ) -> tuple[int, dict]:
        """Send one generation to the worker and stream its logs to the job."""
        with session.lock:
            process = session.process
            if process.poll() is not None or process.stdin is None:
                raise RuntimeError("native H3 worker exited before the generation started")
            process.stdin.write(json.dumps({"job_id": job_id, "argv": argv}) + "\n")
            process.stdin.flush()
            while True:
                try:
                    line = session.output.get(timeout=1.0)
                except queue.Empty:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"native H3 worker exited with code {process.returncode}"
                        )
                    continue
                if line is None:
                    raise RuntimeError("native H3 worker closed its output stream")
                if line.startswith(RESULT_PREFIX):
                    try:
                        result = json.loads(line[len(RESULT_PREFIX):])
                    except json.JSONDecodeError as exc:
                        raise RuntimeError("native H3 worker returned invalid status data") from exc
                    return int(result.get("returncode", 1)), result.get("runtime_metrics") or {}
                log.write(line)
                log.flush()
                on_line(line)
