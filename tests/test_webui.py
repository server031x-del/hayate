from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hayate.backends.minimax_h3.generation import GenerationPlan, GenerationRequest
from hayate.backends.minimax_h3.upstream import AUDITED_COMMIT, UpstreamValidation
from hayate.webui.jobs import FINAL_STATUSES, JobManager, JobStore
from hayate.webui.progress import H3ProgressParser
from hayate.webui.server import (
    GenerationPayload,
    _nearest_h3_frame_count,
    _profile_values,
    create_app,
)


def test_h3_duration_snaps_to_supported_frame_geometry():
    assert _nearest_h3_frame_count(5) == 124
    assert _nearest_h3_frame_count(10) == 243
    assert (_nearest_h3_frame_count(15) - 5) % 17 == 0


def test_web_profiles_use_shared_validated_values():
    quality = _profile_values(GenerationPayload(prompt="test", profile="quality"))
    sage = _profile_values(GenerationPayload(prompt="test", profile="fast_sage"))
    detail = _profile_values(
        GenerationPayload(prompt="test", profile="fast_sage_detail")
    )
    assert quality["steps"] == 50
    assert quality["attention_backend"] == "sdpa"
    assert quality["easycache"] is False
    assert sage["steps"] == 20
    assert sage["attention_backend"] == "sageattn"
    assert sage["vae_tile_size"] == 256
    assert detail["steps"] == 20
    assert detail["attention_backend"] == "sageattn"
    assert detail["easycache_threshold"] == 0.4
    assert detail["easycache_end"] == 0.85
    assert detail["easycache_max_consecutive_skips"] == 2
    assert detail["vae_tile_size"] == 256


def test_prompt_transform_metadata_is_explicit_and_optional():
    plain = GenerationPayload(prompt="plain")
    assert plain.original_prompt is None
    assert plain.prompt_transform_applied is False
    assert plain.prompt_transform_template_version is None

    transformed = GenerationPayload(
        prompt="integrated_multimodal_description: [Shot 1] dance",
        original_prompt="dance",
        prompt_transform_applied=True,
        prompt_transform_template_version="h3-base-fields-v1",
    )
    assert transformed.original_prompt == "dance"
    assert transformed.prompt_transform_applied is True


def test_structured_progress_event_is_primary_contract():
    parser = H3ProgressParser()
    event = {
        "schema": 1,
        "type": "progress",
        "phase": "denoise",
        "progress": 65.5,
        "stage": "動画生成",
        "detail": "デノイズ 10/19",
        "eta_seconds": 42,
    }
    update = parser.feed("HAYATE_EVENT " + json.dumps(event, ensure_ascii=False))
    assert update is not None
    assert update.progress == 65.5
    assert update.stage == "動画生成"
    assert update.eta_seconds == 42


def test_structured_progress_survives_tqdm_carriage_return_and_trailing_text():
    parser = H3ProgressParser()
    event = {
        "schema": 1,
        "type": "progress",
        "progress": 71.0,
        "stage": "動画生成",
        "detail": "デノイズ 11/19",
    }
    line = (
        "denoise: 57%\rHAYATE_EVENT "
        + json.dumps(event, ensure_ascii=True)
        + "next"
    )
    update = parser.feed(line)
    assert update is not None
    assert update.progress == 71.0
    assert update.detail == "デノイズ 11/19"


def test_webui_static_shell_and_mutation_security(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "HAYATE Studio" in response.text
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]

        settings = client.get("/api/settings").json()["settings"]
        assert client.put("/api/settings", json=settings).status_code == 403
        saved = client.put(
            "/api/settings",
            json=settings,
            headers={"X-HAYATE-UI": "1"},
        )
        assert saved.status_code == 200


def test_webui_rejects_a_second_server_for_the_same_workspace(tmp_path):
    app = create_app(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            create_app(tmp_path)
    finally:
        with TestClient(app):
            pass


def test_persisted_history_remains_playable_after_output_directory_changes(tmp_path):
    previous_output = tmp_path / "previous-output"
    previous_output.mkdir()
    video = previous_output / "older.mp4"
    video.write_bytes(b"older-video")
    unrelated_log = tmp_path / "private.hayate.log"
    unrelated_log.write_text("must not be exposed", encoding="utf-8")
    video.with_suffix(".mp4.hayate.json").write_text(
        json.dumps({"log_path": str(unrelated_log)}), encoding="utf-8"
    )
    malformed_video = previous_output / "malformed.mp4"
    malformed_video.write_bytes(b"still-importable")
    malformed_video.with_suffix(".mp4.hayate.json").write_text(
        "[]", encoding="utf-8"
    )
    app = create_app(tmp_path)
    app.state.job_store.import_outputs(previous_output)
    job = app.state.job_store.get_by_output(video)
    assert job is not None
    assert app.state.job_store.get_by_output(malformed_video) is not None
    with TestClient(app) as client:
        assert client.get(f"/api/jobs/{job['id']}").json()["media_available"] is True
        response = client.get(f"/api/jobs/{job['id']}/media")
        assert response.status_code == 200
        assert response.content == b"older-video"
        log_response = client.get(f"/api/jobs/{job['id']}/log")
        assert log_response.status_code == 200
        assert log_response.json() == {"lines": [], "available": False}

        video.unlink()
        assert client.get(f"/api/jobs/{job['id']}").json()["media_available"] is False
        listed = client.get("/api/jobs").json()["jobs"]
        assert next(item for item in listed if item["id"] == job["id"])[
            "media_available"
        ] is False


def test_library_delete_removes_only_final_job_artifacts_and_history(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    video = output_dir / "delete-me.mp4"
    log = output_dir / "delete-me.mp4.hayate.log"
    manifest = output_dir / "delete-me.mp4.hayate.json"
    unrelated = output_dir / "keep-me.txt"
    video.write_bytes(b"video")
    log.write_text("log", encoding="utf-8")
    manifest.write_text("{}", encoding="utf-8")
    unrelated.write_text("keep", encoding="utf-8")

    app = create_app(tmp_path)
    app.state.job_store.import_outputs(output_dir)
    job = app.state.job_store.get_by_output(video)
    assert job is not None
    with TestClient(app) as client:
        path = f"/api/jobs/{job['id']}"
        assert client.delete(path).status_code == 403

        app.state.job_store.update(job["id"], status="running")
        blocked = client.delete(path, headers={"X-HAYATE-UI": "1"})
        assert blocked.status_code == 409
        assert video.is_file()

        app.state.job_store.update(job["id"], status="succeeded")
        response = client.delete(path, headers={"X-HAYATE-UI": "1"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["deleted"] is True
        assert set(payload["artifacts_deleted"]) == {
            video.name,
            log.name,
            manifest.name,
        }
        assert app.state.job_store.get(job["id"]) is None
        assert not video.exists()
        assert not log.exists()
        assert not manifest.exists()
        assert unrelated.read_text(encoding="utf-8") == "keep"


def _fake_plan(tmp_path: Path, output: Path) -> GenerationPlan:
    code = (
        "import json,os,time; from pathlib import Path; "
        "assert os.environ['PYTHONIOENCODING'].lower() == 'utf-8'; "
        "print('HAYATE_EVENT '+json.dumps({'schema':1,'type':'progress','progress':64,"
        "'stage':'動画生成','detail':'test'}),flush=True); "
        "time.sleep(0.1); "
        f"Path({str(output)!r}).write_bytes(b'fake-mp4'); "
        "print('HAYATE_RUNTIME_METRICS '+json.dumps({'cuda_peak_allocated_bytes':123}),flush=True)"
    )
    validation = UpstreamValidation(
        tmp_path,
        True,
        AUDITED_COMMIT,
        AUDITED_COMMIT,
        (),
        (),
    )
    request = GenerationRequest("test prompt", tmp_path, output)
    return GenerationPlan(request, (sys.executable, "-c", code), {}, validation, (), ())


def test_job_manager_runs_structured_job_and_persists_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("HAYATE_GPU_LEASE_PATH", str(tmp_path / "gpu.lock"))
    store = JobStore(tmp_path / "jobs.sqlite3")
    manager = JobManager(store)
    output = tmp_path / "output.mp4"
    try:
        job = manager.submit(
            _fake_plan(tmp_path, output), {"prompt": "test", "profile": "fast"}
        )
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            current = store.get(job["id"])
            if current and current["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.05)
        assert current is not None
        assert current["status"] == "succeeded"
        assert current["progress"] == 100.0
        assert current["runtime_metrics"]["cuda_peak_allocated_bytes"] == 123
        manifest = output.with_suffix(".mp4.hayate.json")
        deadline = time.monotonic() + 2
        while not manifest.is_file() and time.monotonic() < deadline:
            time.sleep(0.02)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 1
        assert payload["job_id"] == job["id"]
    finally:
        manager.shutdown()


def test_stop_then_cancel_finishes_as_cancelled(tmp_path, monkeypatch):
    monkeypatch.setenv("HAYATE_GPU_LEASE_PATH", str(tmp_path / "gpu.lock"))
    output = tmp_path / "cancelled.mp4"
    code = (
        "import json,time; "
        "print('HAYATE_EVENT '+json.dumps({'schema':1,'type':'progress',"
        "'progress':40,'stage':'動画生成','detail':'デノイズ 1/20'}),flush=True); "
        "time.sleep(30)"
    )
    validation = UpstreamValidation(
        tmp_path,
        True,
        AUDITED_COMMIT,
        AUDITED_COMMIT,
        (),
        (),
    )
    plan = GenerationPlan(
        GenerationRequest("cancel test", tmp_path, output),
        (sys.executable, "-c", code),
        {},
        validation,
        (),
        (),
    )
    store = JobStore(tmp_path / "cancel-jobs.sqlite3")
    manager = JobManager(store)
    try:
        job = manager.submit(plan, {"prompt": "cancel test"})
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            current = store.get(job["id"])
            if current and current["status"] == "running":
                break
            time.sleep(0.02)
        assert current is not None and current["status"] == "running"
        assert manager.stop_and_save(job["id"])["status"] == "stopping"
        assert manager.cancel(job["id"])["status"] in {"cancelling", "cancelled"}
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            current = store.get(job["id"])
            if current and current["status"] in FINAL_STATUSES:
                break
            time.sleep(0.05)
        assert current is not None
        assert current["status"] == "cancelled"
        assert not Path(str(output) + ".stop_decode").exists()
    finally:
        manager.shutdown()


def test_shutdown_before_popen_never_launches_child(tmp_path, monkeypatch):
    monkeypatch.setenv("HAYATE_GPU_LEASE_PATH", str(tmp_path / "gpu.lock"))
    output = tmp_path / "must-not-launch.mp4"
    plan = _fake_plan(tmp_path, output)
    store = JobStore(tmp_path / "shutdown-jobs.sqlite3")
    job_id = "shutdown-race"
    store.create(
        {
            "id": job_id,
            "status": "queued",
            "source": "webui",
            "created_at": "2026-08-25T00:00:00+00:00",
            "progress": 0.0,
            "stage": "待機中",
            "detail": "GPUキューに追加されました",
            "request": {"prompt": "must not launch"},
            "plan": plan.to_dict(),
            "output_path": str(output),
            "log_path": str(output) + ".hayate.log",
        }
    )
    manager = JobManager(store)
    manager._stop.set()

    def reject_popen(*_args, **_kwargs):
        raise AssertionError("Popen must not run after shutdown")

    monkeypatch.setattr(subprocess, "Popen", reject_popen)
    try:
        manager._run(job_id, plan)
        current = store.get(job_id)
        assert current is not None
        assert current["status"] == "interrupted"
        assert not output.exists()
    finally:
        manager.shutdown()
