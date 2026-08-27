from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import subprocess
import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

import psutil
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from hayate.ai.openai_prompt import (
    H3PromptResult,
    MiniMaxH3PromptAssistant,
    OpenAIClientConfig,
    PromptAssistantError,
    PromptRequest,
)
from hayate import __version__
from hayate.backends.minimax_h3 import ExternalH3GenerationBackend, GenerationRequest
from hayate.backends.minimax_h3.generation import generation_artifact_paths
from hayate.hardware import HardwareProfiler
from hayate.models import ModelRegistry
from hayate.profiles import get_generation_profile
from hayate.runtime.gpu_lease import GPULease
from hayate.webui.jobs import FINAL_STATUSES, JobManager, JobStore
from hayate.webui.openai_settings import (
    DEFAULT_OPENAI_MODEL,
    OpenAISettingsStore,
    validate_model,
)
from hayate.webui.settings import SettingsStore, WebUISettings

PROFILE_NAMES = (
    "quality",
    "fast",
    "fast_sage",
    "fast_sage_detail",
    "pdd",
    "pdd_sage",
    "custom",
)
ASSET_ID_RE = re.compile(r"^[a-f0-9]{32}$")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
IMAGE_SIGNATURES = {
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".webp": (b"RIFF",),
}


class SettingsPayload(BaseModel):
    config_path: str
    upstream_path: str
    checkpoint_dir: str
    output_dir: str
    python_path: str
    prompt_cache_dir: str
    pdd_checkpoint_path: str
    pdd_adaln_affine_path: str
    openai_model: str = Field(default=DEFAULT_OPENAI_MODEL, min_length=1, max_length=128)
    # ``None`` means keep the current credential.  A non-empty value replaces
    # it; clearing is an explicit, separate operation.
    openai_api_key: str | None = Field(default=None, max_length=512)
    clear_openai_api_key: bool = False

    @field_validator("openai_model")
    @classmethod
    def valid_openai_model(cls, value: str) -> str:
        return validate_model(value)


class PromptAssistantPayload(BaseModel):
    brief: str = Field(min_length=1, max_length=4000)
    task: Literal["auto", "t2va", "fl2va", "ref2va"] = "auto"
    duration_seconds: float = Field(default=5.0, ge=1.0, le=30.0)
    width: int = Field(default=512, ge=256, le=1536)
    height: int = Field(default=512, ge=256, le=1536)
    include_audio: bool = True
    language: Literal["ja", "en"] = "ja"
    current_prompt: str | None = Field(default=None, max_length=12000)

    @field_validator("width", "height")
    @classmethod
    def multiple_of_32(cls, value: int) -> int:
        if value % 32:
            raise ValueError("must be a multiple of 32")
        return value


class GenerationPayload(BaseModel):
    prompt: str = Field(min_length=1, max_length=12000)
    original_prompt: str | None = Field(default=None, max_length=12000)
    prompt_transform_applied: bool = False
    prompt_transform_template_version: str | None = Field(default=None, max_length=64)
    profile: Literal[
        "quality",
        "fast",
        "fast_sage",
        "fast_sage_detail",
        "pdd",
        "pdd_sage",
        "custom",
    ] = "fast_sage"
    task: Literal["auto", "t2va", "fl2va", "ref2va"] = "auto"
    width: int = Field(default=512, ge=256, le=1536)
    height: int = Field(default=512, ge=256, le=1536)
    duration_seconds: float = Field(default=5.0, ge=1.0, le=30.0)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    filename: str | None = Field(default=None, max_length=120)
    image_asset_id: str | None = None
    last_image_asset_id: str | None = None
    reference_asset_ids: list[str] = Field(default_factory=list, max_length=8)
    use_prompt_cache: bool = True
    steps: int = Field(default=20, ge=2, le=100)
    attention_backend: Literal["sdpa", "sageattn"] = "sageattn"
    easycache: bool = True
    pdd: bool = False
    easycache_threshold: float = Field(default=0.4, ge=0.0, le=2.0)
    easycache_start: float = Field(default=0.15, ge=0.0, le=1.0)
    easycache_end: float = Field(default=0.95, ge=0.0, le=1.0)
    easycache_max_consecutive_skips: int = Field(default=2, ge=1, le=20)
    blocks_to_swap: int = Field(default=49, ge=0, le=49)
    activation_chunk_rows: int = Field(default=32768, ge=0, le=1048576)
    vae_tile_size: int = Field(default=256, ge=16, le=2048)

    @field_validator("width", "height")
    @classmethod
    def multiple_of_32(cls, value: int) -> int:
        if value % 32:
            raise ValueError("must be a multiple of 32")
        return value

    @field_validator("image_asset_id", "last_image_asset_id")
    @classmethod
    def valid_optional_asset(cls, value: str | None) -> str | None:
        if value is not None and not ASSET_ID_RE.fullmatch(value):
            raise ValueError("invalid asset id")
        return value

    @field_validator("reference_asset_ids")
    @classmethod
    def valid_assets(cls, values: list[str]) -> list[str]:
        if any(not ASSET_ID_RE.fullmatch(value) for value in values):
            raise ValueError("invalid reference asset id")
        return values


def _nearest_h3_frame_count(duration_seconds: float) -> int:
    target = duration_seconds * 24.0
    groups = max(1, round((target - 5) / 17))
    return 17 * groups + 5


def _safe_output_name(value: str | None, job_token: str) -> str:
    stem = Path(value or "hayate").stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._") or "hayate"
    return f"{stem}-{job_token}.mp4"


def _gpu_resources() -> list[dict[str, object]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows: list[dict[str, object]] = []
    if result.returncode:
        return rows
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "used_bytes": int(float(parts[2]) * 1024 * 1024),
                    "total_bytes": int(float(parts[3]) * 1024 * 1024),
                    "utilization_percent": float(parts[4]),
                    "temperature_c": float(parts[5]),
                }
            )
        except ValueError:
            continue
    return rows


def _profile_values(payload: GenerationPayload) -> dict[str, object]:
    values: dict[str, object] = {
        "steps": payload.steps,
        "attention_backend": payload.attention_backend,
        "easycache": payload.easycache,
        "pdd": payload.pdd,
        "easycache_threshold": payload.easycache_threshold,
        "easycache_start": payload.easycache_start,
        "easycache_end": payload.easycache_end,
        "easycache_max_consecutive_skips": payload.easycache_max_consecutive_skips,
        "blocks_to_swap": payload.blocks_to_swap,
        "activation_chunk_rows": payload.activation_chunk_rows,
        "vae_tile_size": payload.vae_tile_size,
    }
    if payload.profile != "custom":
        profile = get_generation_profile(payload.profile)
        values.update(
            {key: value for key, value in profile.to_dict().items() if key in values}
        )
    return values


def create_app(
    workspace: Path | None = None,
    *,
    job_manager: JobManager | None = None,
    openai_store: OpenAISettingsStore | None = None,
    trusted_hosts: list[str] | None = None,
) -> FastAPI:
    root = (workspace or Path.cwd()).resolve(strict=False)
    data_dir = root / "data" / "webui"
    static_dir = Path(__file__).resolve().parent / "static"
    settings_store = SettingsStore(data_dir / "settings.json", root)
    openai_settings_store = openai_store or OpenAISettingsStore(
        data_dir / "openai-settings.json"
    )
    instance_lease = GPULease(
        data_dir / "webui-server.lock",
        owner={"pid": os.getpid(), "kind": "webui-server"},
    )
    if not instance_lease.acquire():
        owner = instance_lease.busy_owner() or {}
        raise RuntimeError(
            f"HAYATE WebUI is already running (pid={owner.get('pid', '?')})"
        )
    try:
        store = (
            job_manager.store
            if job_manager is not None
            else JobStore(data_dir / "hayate-webui.sqlite3")
        )
        manager = job_manager or JobManager(store)
        settings = settings_store.load()
        store.import_outputs(Path(settings.output_dir))
    except Exception:
        instance_lease.release()
        raise
    hardware_lock = threading.Lock()
    hardware_cache: dict | None = None

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            try:
                if job_manager is None:
                    manager.shutdown()
            finally:
                instance_lease.release()

    app = FastAPI(
        title="HAYATE Studio",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.workspace = root
    app.state.settings_store = settings_store
    app.state.openai_settings_store = openai_settings_store
    app.state.job_store = store
    app.state.job_manager = manager
    app.state.network_exposed = trusted_hosts == ["*"]
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=trusted_hosts
        or ["127.0.0.1", "localhost", "[::1]", "testserver"],
    )
    app.mount("/assets", StaticFiles(directory=static_dir), name="assets")

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        """Keep credential values out of FastAPI's default 422 echo."""

        errors = []
        for error in exc.errors():
            safe_error = dict(error)
            location = tuple(error.get("loc", ()))
            if "openai_api_key" in location:
                safe_error.pop("input", None)
            errors.append(safe_error)
        return JSONResponse(
            status_code=422,
            content={"detail": jsonable_encoder(errors)},
        )

    @app.middleware("http")
    async def local_security(request: Request, call_next):
        if request.url.path.startswith("/api/") and request.method in {
            "POST",
            "PUT",
            "PATCH",
            "DELETE",
        }:
            if request.headers.get("X-HAYATE-UI") != "1":
                return JSONResponse(
                    status_code=403, content={"detail": "missing HAYATE UI token"}
                )
            origin = request.headers.get("Origin")
            if (
                origin
                and urlsplit(origin).netloc.lower()
                != request.headers.get("Host", "").lower()
            ):
                return JSONResponse(
                    status_code=403, content={"detail": "cross-origin request rejected"}
                )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
            "connect-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'"
        )
        if request.url.path.startswith("/assets/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def cached_hardware() -> dict:
        nonlocal hardware_cache
        with hardware_lock:
            if hardware_cache is None:
                hardware_cache = HardwareProfiler().profile().to_dict()
            return hardware_cache

    def asset_path(asset_id: str | None) -> Path | None:
        if asset_id is None:
            return None
        if not ASSET_ID_RE.fullmatch(asset_id):
            raise HTTPException(400, "invalid asset id")
        matches = list((data_dir / "uploads").glob(f"{asset_id}.*"))
        if len(matches) != 1 or not matches[0].is_file():
            raise HTTPException(404, "uploaded image was not found")
        return matches[0].resolve(strict=False)

    def persisted_artifact_path(value: str) -> Path:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = root / candidate
        return candidate.resolve(strict=False)

    def ensure_local_secret_action(request: Request) -> None:
        """Do not expose credential mutation or paid AI calls over LAN."""

        if not app.state.network_exposed:
            return
        client_host = request.client.host if request.client else ""
        if client_host not in {"127.0.0.1", "::1", "localhost"}:
            raise HTTPException(
                403,
                "OpenAIの設定とAIプロンプト作成はローカル接続でのみ利用できます",
            )

    def with_media_availability(job: dict) -> dict:
        enriched = dict(job)
        raw_path = enriched.get("output_path") or ""
        path = persisted_artifact_path(raw_path) if raw_path else None
        enriched["media_available"] = bool(
            path is not None and path.suffix.lower() == ".mp4" and path.is_file()
        )
        return enriched

    @app.get("/")
    async def index():
        return FileResponse(
            static_dir / "index.html", headers={"Cache-Control": "no-store"}
        )

    @app.get("/api/bootstrap")
    async def bootstrap():
        current = settings_store.load()
        hardware = await asyncio.to_thread(cached_hardware)
        return {
            "version": __version__,
            "settings": current.to_dict(),
            "readiness": SettingsStore.readiness(current),
            "openai": openai_settings_store.public_dict(),
            "hardware": hardware,
            "jobs": [with_media_availability(job) for job in store.list(100)],
            "profiles": {
                name: get_generation_profile(name).to_dict()
                for name in PROFILE_NAMES
                if name != "custom"
            },
        }

    @app.get("/api/system")
    async def system_resources():
        virtual = psutil.virtual_memory()
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_used_bytes": virtual.used,
            "ram_total_bytes": virtual.total,
            "ram_percent": virtual.percent,
            "gpus": await asyncio.to_thread(_gpu_resources),
        }

    @app.get("/api/settings")
    async def get_settings():
        current = settings_store.load()
        return {
            "settings": current.to_dict(),
            "readiness": SettingsStore.readiness(current),
            "openai": openai_settings_store.public_dict(),
        }

    @app.put("/api/settings")
    async def put_settings(payload: SettingsPayload, request: Request):
        openai_current = openai_settings_store.load()
        openai_changed = payload.openai_model != openai_current.model
        if payload.openai_api_key or payload.clear_openai_api_key or openai_changed:
            ensure_local_secret_action(request)
        if payload.clear_openai_api_key and payload.openai_api_key:
            raise HTTPException(
                422, "APIキーの入力と消去は同時に指定できません"
            )
        runtime_keys = {
            "config_path",
            "upstream_path",
            "checkpoint_dir",
            "output_dir",
            "python_path",
            "prompt_cache_dir",
            "pdd_checkpoint_path",
            "pdd_adaln_affine_path",
        }
        current = settings_store.save(
            WebUISettings(**payload.model_dump(include=runtime_keys))
        )
        if payload.clear_openai_api_key:
            if not openai_settings_store.clear_api_key():
                raise HTTPException(
                    502,
                    {
                        "code": "openai_key_clear_failed",
                        "message": "OpenAI APIキーをCredential Managerから消去できませんでした。状態を確認して再試行してください。",
                    },
                )
        elif payload.openai_api_key and payload.openai_api_key.strip():
            openai_settings_store.set_api_key(payload.openai_api_key)
        openai_settings_store.update(model=payload.openai_model)
        Path(current.output_dir).mkdir(parents=True, exist_ok=True)
        Path(current.prompt_cache_dir).mkdir(parents=True, exist_ok=True)
        store.import_outputs(Path(current.output_dir))
        return {
            "settings": current.to_dict(),
            "readiness": SettingsStore.readiness(current),
            "openai": openai_settings_store.public_dict(),
        }

    @app.post("/api/prompt-assistant", response_model=H3PromptResult)
    async def prompt_assistant(payload: PromptAssistantPayload, request: Request):
        ensure_local_secret_action(request)
        openai_settings = openai_settings_store.load()
        api_key, _source = openai_settings_store.credentials()
        if not api_key:
            raise HTTPException(
                409,
                {
                    "code": "openai_key_missing",
                    "message": "設定画面でOpenAI APIキーを登録してください。",
                },
            )
        assistant = MiniMaxH3PromptAssistant(
            OpenAIClientConfig(
                model=openai_settings.model,
                api_key=api_key,
            )
        )
        request_data = PromptRequest(
            brief=payload.brief,
            task=payload.task,
            duration_seconds=payload.duration_seconds,
            width=payload.width,
            height=payload.height,
            include_audio=payload.include_audio,
            language=payload.language,
            current_prompt=payload.current_prompt or "",
        )
        try:
            return await asyncio.to_thread(assistant.generate, request_data)
        except PromptAssistantError as exc:
            status = 409 if exc.code == "openai_key_missing" else 502
            raise HTTPException(status, {"code": exc.code, "message": exc.message}) from exc

    @app.post("/api/assets")
    async def upload_asset(file: Annotated[UploadFile, File()]):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in IMAGE_SIGNATURES:
            raise HTTPException(415, "PNG, JPEG, or WebP images are supported")
        upload_dir = data_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        asset_id = hashlib.md5(os.urandom(32), usedforsecurity=False).hexdigest()
        target = upload_dir / f"{asset_id}{suffix}"
        total = 0
        prefix = b""
        try:
            with target.open("xb") as handle:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "image exceeds the 25 MiB limit")
                    if not prefix:
                        prefix = chunk[:16]
                    handle.write(chunk)
            signatures = IMAGE_SIGNATURES[suffix]
            if not any(prefix.startswith(signature) for signature in signatures):
                raise HTTPException(
                    415, "file content does not match its image extension"
                )
            if suffix == ".webp" and prefix[8:12] != b"WEBP":
                raise HTTPException(415, "invalid WebP image")
            try:
                from PIL import Image

                with Image.open(target) as image:
                    width, height = image.size
                    if width * height > 64_000_000:
                        raise HTTPException(
                            413, "image dimensions exceed 64 megapixels"
                        )
                    image.verify()
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(415, "image could not be decoded safely") from exc
        except Exception:
            target.unlink(missing_ok=True)
            raise
        finally:
            await file.close()
        return {
            "id": asset_id,
            "name": file.filename,
            "size": total,
            "url": f"/api/assets/{asset_id}",
        }

    @app.get("/api/assets/{asset_id}")
    async def get_asset(asset_id: str):
        path = asset_path(asset_id)
        assert path is not None
        return FileResponse(path)

    @app.post("/api/jobs", status_code=202)
    async def create_job(payload: GenerationPayload):
        current = settings_store.load()
        output_dir = Path(current.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        token = os.urandom(4).hex()
        output = output_dir / _safe_output_name(payload.filename, token)
        first_image = asset_path(payload.image_asset_id)
        last_image = asset_path(payload.last_image_asset_id)
        references = tuple(
            path
            for item in payload.reference_asset_ids
            if (path := asset_path(item)) is not None
        )
        seed = (
            payload.seed
            if payload.seed is not None
            else random.SystemRandom().randint(0, 2**31 - 1)
        )
        frames = _nearest_h3_frame_count(payload.duration_seconds)
        profile = _profile_values(payload)
        prompt_cache = None
        if payload.use_prompt_cache:
            cache_key = hashlib.sha256(
                json.dumps(
                    {
                        "prompt": payload.prompt,
                        "first": payload.image_asset_id,
                        "last": payload.last_image_asset_id,
                        "references": payload.reference_asset_ids,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()[:24]
            prompt_cache = Path(current.prompt_cache_dir) / f"{cache_key}.safetensors"
        try:
            backend = ExternalH3GenerationBackend(
                current.upstream_path,
                ModelRegistry.load(current.config_path),
                python=current.python_path,
            )
            request = GenerationRequest(
                prompt=payload.prompt,
                checkpoint_dir=Path(current.checkpoint_dir),
                output=output,
                task=payload.task,
                image_path=first_image,
                last_image_path=last_image,
                references=references,
                height=payload.height,
                width=payload.width,
                frames=frames,
                steps=int(profile["steps"]),
                seed=seed,
                blocks_to_swap=int(profile["blocks_to_swap"]),
                activation_chunk_rows=int(profile["activation_chunk_rows"]),
                prompt_cache=prompt_cache,
                easycache=bool(profile["easycache"]),
                easycache_threshold=float(profile["easycache_threshold"]),
                easycache_start=float(profile["easycache_start"]),
                easycache_end=float(profile["easycache_end"]),
                easycache_max_consecutive_skips=int(
                    profile["easycache_max_consecutive_skips"]
                ),
                vae_tile_size=int(profile["vae_tile_size"]),
                attention_backend=str(profile["attention_backend"]),
                pdd_checkpoint=(
                    Path(current.pdd_checkpoint_path) if bool(profile["pdd"]) else None
                ),
                pdd_adaln_affine=(
                    Path(current.pdd_adaln_affine_path) if bool(profile["pdd"]) else None
                ),
            )
            plan = await asyncio.to_thread(backend.plan, request)
            if not plan.executable:
                raise HTTPException(
                    422,
                    {
                        "message": "generation preflight failed",
                        "issues": list(plan.issues),
                    },
                )
            cli_ready, reason = await asyncio.to_thread(backend.probe_cli)
            if not cli_ready:
                raise HTTPException(
                    422, f"upstream generation CLI is unavailable: {reason}"
                )
        except HTTPException:
            raise
        except (OSError, ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc
        request_payload = payload.model_dump()
        request_payload.update(
            frames=frames,
            seed=seed,
            output=str(output.resolve(strict=False)),
            original_prompt=payload.original_prompt or payload.prompt,
            effective_prompt=payload.prompt,
            effective_profile=profile,
            warnings=list(plan.warnings),
        )
        return manager.submit(plan, request_payload)

    @app.get("/api/jobs")
    async def list_jobs(limit: int = 100):
        return {
            "jobs": [with_media_availability(job) for job in store.list(limit)]
        }

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return with_media_availability(job)

    @app.delete("/api/jobs/{job_id}")
    async def delete_job(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        if job["status"] not in FINAL_STATUSES:
            raise HTTPException(409, "active or queued jobs cannot be deleted")

        deleted: list[str] = []
        raw_output = job.get("output_path") or ""
        if raw_output:
            output = persisted_artifact_path(raw_output)
            if output.suffix.lower() != ".mp4":
                raise HTTPException(409, "refusing to delete a non-MP4 job artifact")
            log_path, manifest_path = generation_artifact_paths(output)
            targets = (output, log_path, manifest_path)
            try:
                for target in targets:
                    if target.is_file():
                        target.unlink()
                        deleted.append(target.name)
            except OSError as exc:
                raise HTTPException(
                    409, f"artifact deletion failed for {target.name}: {exc}"
                ) from exc

        if not store.delete(job_id):
            raise HTTPException(409, "job history changed before deletion completed")
        return {"deleted": True, "job_id": job_id, "artifacts_deleted": deleted}

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str):
        try:
            return await asyncio.to_thread(manager.cancel, job_id)
        except KeyError as exc:
            raise HTTPException(404, "job not found") from exc

    @app.post("/api/jobs/{job_id}/stop-and-save")
    async def stop_and_save_job(job_id: str):
        try:
            return await asyncio.to_thread(manager.stop_and_save, job_id)
        except KeyError as exc:
            raise HTTPException(404, "job not found") from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str):
        if store.get(job_id) is None:
            raise HTTPException(404, "job not found")

        async def stream():
            previous = None
            while True:
                job = store.get(job_id)
                if job is None:
                    return
                job = with_media_availability(job)
                encoded = json.dumps(job, ensure_ascii=False, separators=(",", ":"))
                if encoded != previous:
                    yield f"data: {encoded}\n\n"
                    previous = encoded
                if job["status"] in FINAL_STATUSES:
                    return
                await asyncio.sleep(0.75)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/api/jobs/{job_id}/media")
    async def job_media(job_id: str):
        job = store.get(job_id)
        if job is None or not job.get("output_path"):
            raise HTTPException(404, "job output was not found")
        path = persisted_artifact_path(job["output_path"])
        if path.suffix.lower() != ".mp4" or not path.is_file():
            raise HTTPException(404, "job output was not found")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/jobs/{job_id}/log")
    async def job_log(job_id: str, lines: int = 300):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        raw_path = job.get("log_path") or ""
        if not raw_path:
            return {"lines": [], "available": False}
        path = persisted_artifact_path(raw_path)
        if not path.name.endswith(".hayate.log") or not path.is_file():
            return {"lines": [], "available": False}
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return {"lines": content[-max(1, min(lines, 2000)) :], "available": True}

    @app.exception_handler(Exception)
    async def unexpected_error(_request, exc: Exception):
        return JSONResponse(
            status_code=500, content={"detail": f"HAYATE WebUI error: {exc}"}
        )

    return app


def run_webui(
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
    open_browser: bool = False,
    allow_network: bool = False,
    workspace: Path | None = None,
) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"} and not allow_network:
        raise ValueError(
            "network binding requires --allow-network; local-only is the safe default"
        )
    import uvicorn

    app = create_app(
        workspace,
        trusted_hosts=(
            ["*"]
            if allow_network
            else [host, "127.0.0.1", "localhost", "[::1]"]
        ),
    )
    if open_browser:
        timer = threading.Timer(1.2, webbrowser.open, args=(f"http://{host}:{port}",))
        timer.daemon = True
        timer.start()
    uvicorn.run(app, host=host, port=port, log_level="info")
