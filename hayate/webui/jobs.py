from __future__ import annotations

import hashlib
import json
import os
import queue
import signal
import sqlite3
import subprocess
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

import psutil

from hayate.backends.minimax_h3.generation import (
    GenerationPlan,
    generation_artifact_paths,
    write_generation_manifest,
)
from hayate.runtime.gpu_lease import GPULease
from hayate.webui.progress import H3ProgressParser

FINAL_STATUSES = {"succeeded", "partial", "failed", "cancelled", "interrupted"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    _JSON_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"request", "plan", "runtime_metrics"}
    )
    _UPDATE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "status",
            "progress",
            "stage",
            "detail",
            "eta_seconds",
            "started_at",
            "completed_at",
            "duration_seconds",
            "runtime_metrics",
            "output_path",
            "log_path",
            "error",
            "pid",
        }
    )

    def __init__(self, path: Path):
        self.path = path.resolve(strict=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    progress REAL NOT NULL,
                    stage TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    eta_seconds INTEGER,
                    duration_seconds REAL,
                    request TEXT NOT NULL,
                    plan TEXT NOT NULL,
                    runtime_metrics TEXT,
                    output_path TEXT,
                    log_path TEXT,
                    error TEXT,
                    pid INTEGER
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS jobs_created_at ON jobs(created_at DESC)"
            )
            connection.execute(
                """
                UPDATE jobs SET status = 'interrupted', stage = '中断',
                    detail = 'WebUIの前回終了時に実行中でした', completed_at = ?, pid = NULL
                WHERE status IN ('queued', 'running', 'stopping', 'cancelling')
                """,
                (_utc_now(),),
            )

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict:
        payload = dict(row)
        for field in JobStore._JSON_FIELDS:
            value = payload.get(field)
            payload[field] = json.loads(value) if value else None
        return payload

    def create(self, job: dict) -> None:
        encoded = job.copy()
        for field in self._JSON_FIELDS:
            if field in encoded:
                encoded[field] = (
                    json.dumps(encoded[field], ensure_ascii=False)
                    if encoded[field] is not None
                    else None
                )
        columns = ", ".join(encoded)
        placeholders = ", ".join("?" for _ in encoded)
        with self._lock, self._connection() as connection:
            connection.execute(
                f"INSERT INTO jobs ({columns}) VALUES ({placeholders})",
                tuple(encoded.values()),
            )

    def update(self, job_id: str, **changes) -> None:
        if not changes:
            return
        unknown = set(changes) - self._UPDATE_FIELDS
        if unknown:
            raise ValueError(f"unsupported job fields: {', '.join(sorted(unknown))}")
        encoded = changes.copy()
        for field in self._JSON_FIELDS:
            if field in encoded:
                encoded[field] = (
                    json.dumps(encoded[field], ensure_ascii=False)
                    if encoded[field] is not None
                    else None
                )
        assignments = ", ".join(f"{field} = ?" for field in encoded)
        with self._lock, self._connection() as connection:
            connection.execute(
                f"UPDATE jobs SET {assignments} WHERE id = ?",
                (*encoded.values(), job_id),
            )

    def transition(self, job_id: str, expected_statuses: set[str], **changes) -> bool:
        if not expected_statuses or not changes:
            return False
        unknown = set(changes) - self._UPDATE_FIELDS
        if unknown:
            raise ValueError(f"unsupported job fields: {', '.join(sorted(unknown))}")
        encoded = changes.copy()
        for field in self._JSON_FIELDS:
            if field in encoded:
                encoded[field] = (
                    json.dumps(encoded[field], ensure_ascii=False)
                    if encoded[field] is not None
                    else None
                )
        assignments = ", ".join(f"{field} = ?" for field in encoded)
        statuses = tuple(sorted(expected_statuses))
        placeholders = ", ".join("?" for _ in statuses)
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                f"UPDATE jobs SET {assignments} WHERE id = ? "
                f"AND status IN ({placeholders})",
                (*encoded.values(), job_id, *statuses),
            )
            return cursor.rowcount == 1

    def get(self, job_id: str) -> dict | None:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return self._decode(row) if row else None

    def get_by_output(self, output_path: Path) -> dict | None:
        resolved = str(output_path.resolve(strict=False))
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE output_path = ? ORDER BY created_at DESC LIMIT 1",
                (resolved,),
            ).fetchone()
        return self._decode(row) if row else None

    def list(self, limit: int = 100) -> list[dict]:
        limit = max(1, min(int(limit), 500))
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._decode(row) for row in rows]

    def delete(self, job_id: str) -> bool:
        with self._lock, self._connection() as connection:
            cursor = connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            return cursor.rowcount == 1

    def import_outputs(self, output_dir: Path) -> int:
        if not output_dir.is_dir():
            return 0
        imported = 0
        for output in output_dir.glob("*.mp4"):
            resolved = output.resolve(strict=False)
            if self.get_by_output(resolved) is not None:
                continue
            digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:20]
            job_id = f"history-{digest}"
            if self.get(job_id) is not None:
                continue
            log_path, manifest_path = generation_artifact_paths(output)
            manifest: dict = {}
            if manifest_path.is_file():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    manifest = {}
            if not isinstance(manifest, dict):
                manifest = {}
            plan = manifest.get("plan") or {}
            if not isinstance(plan, dict):
                plan = {}
            request = plan.get("request") or {"prompt": "過去の生成結果"}
            if not isinstance(request, dict):
                request = {"prompt": "過去の生成結果"}
            timestamp = datetime.fromtimestamp(
                output.stat().st_mtime, tz=timezone.utc
            ).isoformat()
            returncode = manifest.get("returncode", 0)
            self.create(
                {
                    "id": job_id,
                    "status": "succeeded" if returncode == 0 else "failed",
                    "source": "history",
                    "created_at": timestamp,
                    "started_at": timestamp,
                    "completed_at": timestamp,
                    "progress": 100.0,
                    "stage": "完了",
                    "detail": "既存の出力から読み込みました",
                    "eta_seconds": 0,
                    "duration_seconds": manifest.get("duration_seconds"),
                    "request": request,
                    "plan": plan,
                    "runtime_metrics": manifest.get("runtime_metrics"),
                    "output_path": str(resolved),
                    "log_path": str(log_path.resolve(strict=False)),
                    "error": None if returncode == 0 else f"exit code {returncode}",
                    "pid": None,
                }
            )
            imported += 1
        return imported


class JobManager:
    """Single-GPU FIFO executor with persisted progress and owned-process cancellation."""

    def __init__(self, store: JobStore):
        self.store = store
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._plans: dict[str, GenerationPlan] = {}
        self._active: dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._worker, name="hayate-webui-worker", daemon=True
        )
        self._thread.start()

    def submit(self, plan: GenerationPlan, request_payload: dict) -> dict:
        if not plan.executable:
            raise ValueError("generation preflight failed: " + "; ".join(plan.issues))
        job_id = uuid.uuid4().hex
        output = plan.request.output.resolve(strict=False)
        log_path, _ = generation_artifact_paths(output)
        job = {
            "id": job_id,
            "status": "queued",
            "source": "webui",
            "created_at": _utc_now(),
            "started_at": None,
            "completed_at": None,
            "progress": 0.0,
            "stage": "待機中",
            "detail": "GPUキューに追加されました",
            "eta_seconds": None,
            "duration_seconds": None,
            "request": request_payload,
            "plan": plan.to_dict(),
            "runtime_metrics": None,
            "output_path": str(output),
            "log_path": str(log_path),
            "error": None,
            "pid": None,
        }
        self.store.create(job)
        with self._lock:
            self._plans[job_id] = plan
        self._queue.put(job_id)
        return self.store.get(job_id) or job

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            job = self.store.get(job_id)
            if job is None:
                raise KeyError(job_id)
            status = job["status"]
            if status in FINAL_STATUSES or status == "cancelling":
                return job
            process_active = status in {"running", "stopping"}
            process = self._active.get(job_id)
            if process_active and process is not None and process.poll() is not None:
                return job
            changed = self.store.transition(
                job_id,
                {status},
                status="cancelling" if process_active else "cancelled",
                stage="停止中" if process_active else "キャンセル",
                detail="生成プロセスを安全に停止しています",
                completed_at=None if process_active else _utc_now(),
            )
            if not changed:
                return self.store.get(job_id) or job
        if process is not None:
            self._terminate_owned_tree(process)
        return self.store.get(job_id) or job

    def stop_and_save(self, job_id: str) -> dict:
        with self._lock:
            job = self.store.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job["status"] in FINAL_STATUSES:
                return job
            if job["status"] == "queued":
                return self.cancel(job_id)
            if job["status"] == "stopping":
                return job
            if job["status"] != "running":
                raise ValueError(
                    f"job cannot stop-and-save from status {job['status']}"
                )
            process = self._active.get(job_id)
            if process is not None and process.poll() is not None:
                return job
            output = job.get("output_path")
            if not output:
                raise ValueError("job has no managed output path")
            marker = Path(str(output) + ".stop_decode")
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch(exist_ok=True)
            changed = self.store.transition(
                job_id,
                {"running"},
                status="stopping",
                stage="途中停止",
                detail="次のステップで停止して現在の状態を保存します",
                eta_seconds=None,
            )
            if not changed:
                marker.unlink(missing_ok=True)
        return self.store.get(job_id) or job

    @staticmethod
    def _terminate_owned_tree(process: subprocess.Popen[str]) -> None:
        if os.name == "nt":
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT)
                process.wait(timeout=3)
                return
            except (OSError, subprocess.TimeoutExpired):
                pass
        try:
            parent = psutil.Process(process.pid)
            owned = parent.children(recursive=True)
            for child in reversed(owned):
                child.terminate()
            parent.terminate()
            _, alive = psutil.wait_procs([*owned, parent], timeout=5)
            for item in alive:
                item.kill()
        except (psutil.Error, OSError):
            try:
                process.terminate()
            except OSError:
                pass

    def _worker(self) -> None:
        while not self._stop.is_set():
            job_id = self._queue.get()
            if job_id is None:
                return
            if self._stop.is_set():
                with self._lock:
                    self.store.transition(
                        job_id,
                        {"queued"},
                        status="interrupted",
                        stage="中断",
                        detail="WebUIを終了したため実行しませんでした",
                        completed_at=_utc_now(),
                    )
                    self._plans.pop(job_id, None)
                return
            job = self.store.get(job_id)
            if job is None or job["status"] != "queued":
                with self._lock:
                    self._plans.pop(job_id, None)
                continue
            with self._lock:
                plan = self._plans.get(job_id)
            if plan is None:
                self.store.update(
                    job_id,
                    status="interrupted",
                    stage="中断",
                    detail="実行計画を復元できませんでした",
                    completed_at=_utc_now(),
                )
                continue
            self._run(job_id, plan)

    def _run(self, job_id: str, plan: GenerationPlan) -> None:
        output = plan.request.output.resolve(strict=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        log_path, _ = generation_artifact_paths(output)
        environment = os.environ.copy()
        environment.update(plan.environment)
        environment["PYTHONIOENCODING"] = "utf-8"
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        started = 0.0
        parser = H3ProgressParser()
        process: subprocess.Popen[str] | None = None
        lease = GPULease(
            owner={"pid": os.getpid(), "kind": "webui-generation", "job_id": job_id}
        )
        returncode = -1
        error: str | None = None
        try:
            while not lease.acquire():
                current = self.store.get(job_id) or {}
                if current.get("status") != "queued" or self._stop.is_set():
                    with self._lock:
                        self._plans.pop(job_id, None)
                    return
                owner = lease.busy_owner() or {}
                self.store.update(
                    job_id,
                    stage="GPU待機",
                    detail=f"別のHAYATE処理の完了を待っています · PID {owner.get('pid', '?')}",
                )
                time.sleep(0.75)
            started = time.perf_counter()
            with self._lock:
                if self._stop.is_set():
                    self.store.transition(
                        job_id,
                        {"queued"},
                        status="interrupted",
                        stage="中断",
                        detail="WebUIを終了したため実行しませんでした",
                        completed_at=_utc_now(),
                    )
                    self._plans.pop(job_id, None)
                    return
                if (self.store.get(job_id) or {}).get("status") != "queued":
                    self._plans.pop(job_id, None)
                    return
                process = subprocess.Popen(
                    list(plan.command),
                    cwd=str(plan.upstream.checkout),
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=creationflags,
                )
                self._active[job_id] = process
                running = self.store.transition(
                    job_id,
                    {"queued"},
                    status="running",
                    progress=1.0,
                    stage="起動準備",
                    detail="MiniMax H3エンジンを起動しています",
                    started_at=_utc_now(),
                    pid=process.pid,
                )
                if not running:
                    self._active.pop(job_id, None)
                    self._plans.pop(job_id, None)
                    self._terminate_owned_tree(process)
                    return
            with log_path.open("w", encoding="utf-8") as log:
                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    update = parser.feed(line)
                    if update is not None:
                        changes: dict[str, object] = {
                            "progress": update.progress,
                            "stage": update.stage,
                            "detail": update.detail,
                            "eta_seconds": update.eta_seconds,
                        }
                        if update.runtime_metrics is not None:
                            changes["runtime_metrics"] = update.runtime_metrics
                        self.store.update(job_id, **changes)
            returncode = process.wait()
        except (OSError, subprocess.SubprocessError) as exc:
            error = str(exc)
        finally:
            lease.release()

        duration = time.perf_counter() - started
        runtime_metrics = parser.runtime_metrics
        Path(str(output) + ".stop_decode").unlink(missing_ok=True)
        with self._lock:
            current = self.store.get(job_id) or {}
            current_status = str(current.get("status") or "")
            cancelled = current_status == "cancelling"
            stopped_early = current_status == "stopping"
            succeeded = returncode == 0 and output.is_file() and not cancelled
            status = (
                "cancelled"
                if cancelled
                else "partial"
                if succeeded and stopped_early
                else "succeeded"
                if succeeded
                else "failed"
            )
            if not succeeded and not cancelled and error is None:
                error = f"generation exited with code {returncode}"
            if current_status not in FINAL_STATUSES:
                self.store.transition(
                    job_id,
                    {current_status},
                    status=status,
                    progress=(
                        100.0 if succeeded else float(current.get("progress") or 0.0)
                    ),
                    stage=(
                        "途中保存"
                        if succeeded and stopped_early
                        else "完了"
                        if succeeded
                        else "キャンセル"
                        if cancelled
                        else "エラー"
                    ),
                    detail=(
                        "現在の状態を動画として保存しました"
                        if succeeded and stopped_early
                        else "動画を生成しました"
                        if succeeded
                        else "生成をキャンセルしました"
                        if cancelled
                        else "ログを確認してください"
                    ),
                    eta_seconds=0,
                    completed_at=_utc_now(),
                    duration_seconds=duration,
                    runtime_metrics=runtime_metrics,
                    error=error,
                    pid=None,
                )
            self._active.pop(job_id, None)
            self._plans.pop(job_id, None)
        write_generation_manifest(
            plan,
            returncode=returncode,
            duration_seconds=duration,
            log_path=log_path,
            runtime_metrics=runtime_metrics,
            job_id=job_id,
        )

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            for job in self.store.list(500):
                if job["status"] == "queued":
                    self.store.transition(
                        job["id"],
                        {"queued"},
                        status="interrupted",
                        stage="中断",
                        detail="WebUIを終了したため実行しませんでした",
                        completed_at=_utc_now(),
                    )
                    self._plans.pop(job["id"], None)
            active = list(self._active)
        for job_id in active:
            self.cancel(job_id)
        self._queue.put(None)
        self._thread.join(timeout=8)
