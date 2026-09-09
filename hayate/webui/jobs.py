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
from hayate.runtime.gpu_devices import (
    AUTO_GPU,
    GPUDevice,
    allowed_gpu_devices,
    discover_gpu_devices,
    eligible_gpu_devices,
    normalize_gpu_selector,
    resolve_gpu_selector,
)
from hayate.runtime.gpu_lease import GPULease

BLACKWELL_FASTVIDEO_CAPABILITIES = frozenset({"10.0", "10.3"})
FAST_PROFILE_MIN_VRAM_BYTES = 80 * 1024**3
from hayate.backends.minimax_h3.runtime_command import (
    update_runtime_command_environment,
)
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
            "requested_gpu_selector",
            "assigned_gpu_uuid",
            "assigned_gpu_index",
            "assigned_gpu_name",
            "assigned_gpu_compute_capability",
            "assigned_at",
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
                    pid INTEGER,
                    requested_gpu_selector TEXT,
                    assigned_gpu_uuid TEXT,
                    assigned_gpu_index INTEGER,
                    assigned_gpu_name TEXT,
                    assigned_gpu_compute_capability TEXT,
                    assigned_at TEXT
                )
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
            }
            for name, definition in (
                ("requested_gpu_selector", "TEXT"),
                ("assigned_gpu_uuid", "TEXT"),
                ("assigned_gpu_index", "INTEGER"),
                ("assigned_gpu_name", "TEXT"),
                ("assigned_gpu_compute_capability", "TEXT"),
                ("assigned_at", "TEXT"),
            ):
                if name not in columns:
                    connection.execute(f"ALTER TABLE jobs ADD COLUMN {name} {definition}")
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
            assignment = manifest.get("gpu_assignment") or {}
            if not isinstance(assignment, dict):
                assignment = {}
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
                    "requested_gpu_selector": request.get("gpu_device", AUTO_GPU),
                    "assigned_gpu_uuid": assignment.get("uuid"),
                    "assigned_gpu_index": assignment.get("index"),
                    "assigned_gpu_name": assignment.get("name"),
                    "assigned_gpu_compute_capability": assignment.get("compute_capability"),
                    "assigned_at": timestamp if assignment else None,
                }
            )
            imported += 1
        return imported


class JobManager:
    """GPU-aware executor with persisted progress and owned-process cancellation.

    The default remains one worker because H3's CPU-offloaded weights can
    consume most of a consumer host's RAM.  Setting ``worker_count`` above
    one enables one child process per physical GPU; each child sees its chosen
    adapter as ``cuda:0`` via a UUID mask, so H3 itself remains untouched.
    """

    def __init__(
        self,
        store: JobStore,
        *,
        worker_count: int = 1,
        gpu_discovery=discover_gpu_devices,
    ):
        self.store = store
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._plans: dict[str, GenerationPlan] = {}
        self._active: dict[str, subprocess.Popen[str]] = {}
        self.worker_count = max(1, min(8, int(worker_count)))
        self._gpu_discovery = gpu_discovery
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._workers = [
            threading.Thread(
                target=self._worker,
                name=f"hayate-webui-worker-{index + 1}",
                daemon=True,
            )
            for index in range(self.worker_count)
        ]
        # Kept as a compatibility alias for integrations that inspected the
        # old single worker during v0.1.
        self._thread = self._workers[0]
        for worker in self._workers:
            worker.start()

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
            "requested_gpu_selector": plan.request.gpu_device,
            "assigned_gpu_uuid": None,
            "assigned_gpu_index": None,
            "assigned_gpu_name": None,
            "assigned_gpu_compute_capability": None,
            "assigned_at": None,
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
            if (job.get("plan") or {}).get("backend") == "comfy_fasth3":
                raise ValueError("FastH3 ComfyUIは途中保存に未対応です。キャンセルを使用してください")
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

    def _gpu_candidates(
        self,
        plan: GenerationPlan,
        environment: dict[str, str],
    ) -> tuple[list[GPUDevice | None], str | None]:
        """Return candidate physical GPUs and an optional validation error."""

        try:
            selector = normalize_gpu_selector(plan.request.gpu_device)
        except ValueError as exc:
            return [], str(exc)
        require_blackwell = (
            plan.backend == "fastvideo_vsa"
            and environment.get("HAYATE_FASTH3_REQUIRE_BLACKWELL") == "1"
        )

        def eligible_for_plan(device: GPUDevice) -> bool:
            return (
                not require_blackwell
                or (
                    str(device.compute_capability) in BLACKWELL_FASTVIDEO_CAPABILITIES
                    and device.vram_total_bytes >= FAST_PROFILE_MIN_VRAM_BYTES
                )
            )

        def blackwell_issue() -> str:
            return "FastH3最速プロファイルには80 GiB以上のVRAMを持つBlackwell（compute capability 10.0/10.3）が必要です"

        # An explicit plan has already resolved the UUID during preflight. It
        # must be honored even if the inventory changes between submit/run.
        planned_uuid = environment.get("HAYATE_GPU_UUID") or ""
        if planned_uuid:
            devices = allowed_gpu_devices(
                self._gpu_discovery(),
                visible_devices=environment.get("CUDA_VISIBLE_DEVICES"),
            )
            selected = resolve_gpu_selector(planned_uuid, devices)
            if selected is not None:
                if not selected.h3_eligible:
                    return [], f"selected GPU is not eligible for H3 W4A8: {selected.eligibility_reason}"
                if not eligible_for_plan(selected):
                    return [], blackwell_issue()
                return [selected], None
            if devices:
                return [], "選択したGPUが現在のCUDA allow-listに含まれていません"
            if require_blackwell:
                return [], blackwell_issue()
            # Keep the stable identity as a synthetic device so a transient
            # nvidia-smi failure does not silently move the job elsewhere.
            return [
                GPUDevice(
                    index=int(environment.get("HAYATE_GPU_INDEX") or -1),
                    name="選択済みGPU",
                    vram_total_bytes=0,
                    uuid=planned_uuid,
                )
            ], None

        devices = allowed_gpu_devices(
            self._gpu_discovery(),
            visible_devices=environment.get("CUDA_VISIBLE_DEVICES"),
        )
        if selector != AUTO_GPU:
            selected = resolve_gpu_selector(selector, devices)
            if selected is None:
                return [], f"selected GPU was not found or is not allowed: {selector}"
            if not selected.h3_eligible:
                return [], f"selected GPU is not eligible for H3 W4A8: {selected.eligibility_reason}"
            if not eligible_for_plan(selected):
                return [], blackwell_issue()
            return [selected], None
        # No nvidia-smi (for example a CPU-only preflight test) keeps the
        # historical global lease path and lets the child decide its device.
        eligible = [device for device in eligible_gpu_devices(devices) if eligible_for_plan(device)]
        if eligible:
            return eligible, None
        if devices:
            if require_blackwell:
                return [], blackwell_issue()
            if any(device.h3_eligible for device in devices):
                return [], "H3対応GPUのUUIDを取得できないため、自動割り当てできません。物理indexを明示してください"
            return [], "H3 W4A8を自動割り当てできるSM 8.0以上のGPUがありません"
        if require_blackwell:
            return [], blackwell_issue()
        return [None], None

    def _acquire_gpu_lease(
        self,
        job_id: str,
        plan: GenerationPlan,
        environment: dict[str, str],
    ) -> tuple[GPULease | None, GPUDevice | None, str | None]:
        if self._stop.is_set():
            return None, None, None
        candidates, issue = self._gpu_candidates(plan, environment)
        if issue:
            return None, None, issue
        has_physical_candidates = any(device is not None for device in candidates)
        while not self._stop.is_set():
            for device in candidates:
                gpu_id = device.identity if device is not None else None
                lease = GPULease(
                    gpu_id=gpu_id,
                    namespace="scheduler",
                    owner={
                        "pid": os.getpid(),
                        "kind": "webui-generation",
                        "job_id": job_id,
                        "gpu_uuid": device.uuid if device is not None else None,
                        "gpu_index": device.index if device is not None else None,
                    },
                )
                if lease.acquire():
                    if device is not None:
                        # A child from a crashed parent may still own the
                        # runtime namespace.  Probe it before committing to
                        # an expensive Python/torch launch; otherwise the new
                        # child would fail only after importing H3.
                        runtime_probe = GPULease(
                            gpu_id=gpu_id,
                            namespace="runtime",
                            owner={
                                "pid": os.getpid(),
                                "kind": "runtime-probe",
                                "job_id": job_id,
                            },
                        )
                        if not runtime_probe.acquire():
                            lease.release()
                            continue
                        runtime_probe.release()
                        # UUID masking prevents physical nvidia-smi indices
                        # from being confused with PyTorch's visible ordinals.
                        environment["CUDA_VISIBLE_DEVICES"] = device.visible_id
                        environment["HAYATE_GPU_UUID"] = device.uuid or ""
                        environment["HAYATE_GPU_INDEX"] = str(device.index)
                        environment["HAYATE_GPU_RUNTIME_LEASE"] = "1"
                    return lease, device, None
            current = self.store.get(job_id) or {}
            if current.get("status") != "queued":
                return None, None, None
            if has_physical_candidates:
                # Requeue instead of pinning a worker to a busy explicit GPU;
                # a later job may be able to use another adapter.
                return None, None, "busy"
            self.store.update(
                job_id,
                stage="GPU待機",
                detail="空いているGPUの割り当てを待っています",
            )
            time.sleep(0.75)
        return None, None, None

    def _run(self, job_id: str, plan: GenerationPlan) -> None:
        output = plan.request.output.resolve(strict=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        log_path, _ = generation_artifact_paths(output)
        environment = os.environ.copy()
        environment.update(plan.environment)
        # The OpenAI prompt-authoring credential belongs to the WebUI process
        # only.  Never inherit it into the separate MiniMax H3 generation
        # process, whose logs and third-party runtime are unrelated to AI
        # authoring.
        environment.pop("OPENAI_API_KEY", None)
        environment["PYTHONIOENCODING"] = "utf-8"
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        started = 0.0
        parser = H3ProgressParser()
        process: subprocess.Popen[str] | None = None
        lease: GPULease | None = None
        assigned_device: GPUDevice | None = None
        returncode = -1
        error: str | None = None
        try:
            lease, assigned_device, acquire_issue = self._acquire_gpu_lease(
                job_id, plan, environment
            )
            if acquire_issue:
                if acquire_issue == "busy":
                    self.store.update(
                        job_id,
                        stage="GPU待機",
                        detail="空いているGPUの割り当てを待っています",
                    )
                    if not self._stop.is_set() and (self.store.get(job_id) or {}).get("status") == "queued":
                        time.sleep(0.15)
                        self._queue.put(job_id)
                    return
                error = acquire_issue
                self.store.transition(
                    job_id,
                    {"queued"},
                    status="failed",
                    stage="エラー",
                    detail="GPUを割り当てられませんでした",
                    completed_at=_utc_now(),
                    error=error,
                )
                return
            if lease is None:
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
            if assigned_device is not None:
                self.store.update(
                    job_id,
                    detail=(
                        f"GPU {assigned_device.index} · {assigned_device.name} を割り当てました"
                    ),
                    assigned_gpu_uuid=assigned_device.uuid,
                    assigned_gpu_index=assigned_device.index,
                    assigned_gpu_name=assigned_device.name,
                    assigned_gpu_compute_capability=assigned_device.compute_capability,
                    assigned_at=_utc_now(),
                )
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
                # FastVideo's WSL bridge serializes its allow-listed
                # environment after ``/usr/bin/env`` in argv.  The scheduler
                # selects a physical GPU immediately before spawning, so
                # refresh that embedded assignment as well as Popen(env=...).
                command = list(plan.command)
                if plan.backend == "fastvideo_vsa":
                    command = update_runtime_command_environment(
                        command,
                        environment,
                        project_root=Path(__file__).resolve().parents[2],
                    )
                process = subprocess.Popen(
                    command,
                    cwd=str(
                        plan.working_directory
                        or (plan.upstream.checkout if plan.upstream is not None else Path.cwd())
                    ),
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
            if lease is not None:
                lease.release()

        duration = time.perf_counter() - started
        runtime_metrics = parser.runtime_metrics
        if assigned_device is not None:
            runtime_metrics = dict(runtime_metrics or {})
            runtime_metrics["gpu"] = assigned_device.to_dict()
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
            gpu_assignment=assigned_device.to_dict() if assigned_device is not None else None,
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
        for _ in self._workers:
            self._queue.put(None)
        for worker in self._workers:
            worker.join(timeout=8)
