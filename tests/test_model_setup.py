from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hayate.webui.model_setup import (
    ModelArtifact,
    ModelAsset,
    ModelSetupError,
    ModelSetupService,
)
from hayate.webui.server import create_app


PAYLOAD = b"audited-model-payload"


def _asset() -> ModelAsset:
    return ModelAsset(
        id="test_asset",
        role="transformer",
        label="Test model",
        repo_id="verified/repository",
        revision="0123456789abcdef",
        license="Test license",
        license_url="https://huggingface.co/verified/repository",
        artifacts=(
            ModelArtifact(
                remote_path="model.safetensors",
                relative_path="test/model.safetensors",
                size_bytes=len(PAYLOAD),
                sha256=hashlib.sha256(PAYLOAD).hexdigest(),
            ),
        ),
    )


def _downloader(_asset: ModelAsset, artifact: ModelArtifact, temporary: Path) -> Path:
    target = temporary / artifact.remote_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(PAYLOAD)
    return target


def _wait(service: ModelSetupService, download_id: str) -> dict:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        job = service.downloads()[download_id]
        if job["status"] in {"completed", "failed", "interrupted"}:
            return job
        time.sleep(0.01)
    raise AssertionError("model download did not finish")


def test_model_setup_prepares_standard_directories_and_verifies_download(tmp_path):
    asset = _asset()
    service = ModelSetupService(tmp_path, downloader=_downloader, assets=(asset,))

    prepared = service.prepare()
    assert prepared["directories"]["models"]["exists"] is True
    assert prepared["directories"]["checkpoint"]["exists"] is True
    assert prepared["assets"][0]["status"] == "missing"

    job = service.start_download(asset.id, license_accepted=True)
    completed = _wait(service, job["id"])
    assert completed["status"] == "completed"
    assert completed["progress"] == 100.0
    assert (tmp_path / "models" / "test" / "model.safetensors").read_bytes() == PAYLOAD

    detected = service.status()["assets"][0]
    assert detected["exists"] is True
    assert detected["verified"] is True
    assert detected["status"] == "verified"


def test_model_setup_refuses_to_overwrite_unexpected_existing_file(tmp_path):
    asset = _asset()
    target = tmp_path / "models" / "test" / "model.safetensors"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"do-not-overwrite")
    called = False

    def downloader(*_args):
        nonlocal called
        called = True
        raise AssertionError("download must not start")

    service = ModelSetupService(tmp_path, downloader=downloader, assets=(asset,))
    job = service.start_download(asset.id, license_accepted=True)
    failed = _wait(service, job["id"])

    assert failed["status"] == "failed"
    assert "上書きしません" in failed["message"]
    assert target.read_bytes() == b"do-not-overwrite"
    assert called is False


def test_model_setup_blocks_duplicate_download_for_same_asset(tmp_path):
    asset = _asset()
    entered = threading.Event()
    release = threading.Event()

    def slow_downloader(_asset, artifact, temporary):
        entered.set()
        assert release.wait(2)
        return _downloader(_asset, artifact, temporary)

    service = ModelSetupService(tmp_path, downloader=slow_downloader, assets=(asset,))
    first = service.start_download(asset.id, license_accepted=True)
    assert entered.wait(2)
    with pytest.raises(ModelSetupError, match="既にダウンロード中"):
        service.start_download(asset.id, license_accepted=True)
    release.set()
    assert _wait(service, first["id"])["status"] == "completed"


def test_model_setup_reports_partial_download_progress(tmp_path):
    payload = b"partial-progress-payload" * (128 * 1024)
    artifact = ModelArtifact(
        remote_path="model.safetensors",
        relative_path="test/model.safetensors",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    asset = ModelAsset(
        id="progress_asset",
        role="transformer",
        label="Progress model",
        repo_id="verified/repository",
        revision="0123456789abcdef",
        license="Test license",
        license_url="https://huggingface.co/verified/repository",
        artifacts=(artifact,),
    )
    entered = threading.Event()
    release = threading.Event()
    first_chunk = len(payload) // 2

    def slow_downloader(_asset, current_artifact, temporary):
        target = temporary / current_artifact.remote_path
        target.parent.mkdir(parents=True, exist_ok=True)
        incomplete = temporary / ".cache" / "huggingface" / "download" / "hashed.etag.incomplete"
        incomplete.parent.mkdir(parents=True, exist_ok=True)
        with incomplete.open("wb") as handle:
            handle.write(payload[:first_chunk])
            handle.flush()
            entered.set()
            assert release.wait(4)
            handle.write(payload[first_chunk:])
            handle.flush()
        incomplete.replace(target)
        return target

    service = ModelSetupService(tmp_path, downloader=slow_downloader, assets=(asset,))
    job = service.start_download(asset.id, license_accepted=True)
    assert entered.wait(2)
    observed = None
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            current = service.downloads()[job["id"]]
            if current["bytes_downloaded"] >= first_chunk:
                observed = current
                break
            time.sleep(0.05)
        assert observed is not None
        assert observed["progress"] > 0
        assert observed["download_speed_bytes_per_sec"] > 0
        assert "残り" in observed["message"]
    finally:
        release.set()
    assert _wait(service, job["id"])["status"] == "completed"


def test_model_setup_recovers_only_recorded_stale_temporary_directory(tmp_path):
    models = tmp_path / "models"
    stale = models / ".hayate-download-recorded"
    unrelated = models / ".hayate-download-untracked"
    stale.mkdir(parents=True)
    unrelated.mkdir()
    state_path = tmp_path / "data" / "webui" / "model-setup.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "downloads": {
                    "old": {
                        "id": "old",
                        "asset_id": "test_asset",
                        "status": "downloading",
                        "temp_dir": str(stale),
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    service = ModelSetupService(tmp_path, state_path=state_path, assets=(_asset(),))

    assert not stale.exists()
    assert unrelated.is_dir()
    assert service.downloads()["old"]["status"] == "interrupted"


def test_model_setup_ignores_malformed_state_sections(tmp_path):
    state_path = tmp_path / "data" / "webui" / "model-setup.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "downloads": ["not-a-record"],
                "verified": {"bad": "not-a-record", "ok": {"revision": "old"}},
                "license_acceptance": None,
            }
        ),
        encoding="utf-8",
    )

    service = ModelSetupService(tmp_path, state_path=state_path, assets=(_asset(),))

    assert service.downloads() == {}
    assert service.status()["assets"][0]["status"] == "missing"


def test_model_setup_api_uses_allowlist_and_requires_license_acceptance(tmp_path):
    asset = _asset()
    service = ModelSetupService(tmp_path, downloader=_downloader, assets=(asset,))

    with TestClient(create_app(tmp_path, model_setup_service=service)) as client:
        listing = client.get("/api/models/setup")
        assert listing.status_code == 200
        assert listing.json()["assets"][0]["id"] == asset.id

        assert client.post("/api/models/setup/prepare").status_code == 403
        prepared = client.post(
            "/api/models/setup/prepare", headers={"X-HAYATE-UI": "1"}
        )
        assert prepared.status_code == 200

        rejected = client.post(
            "/api/models/setup/download",
            json={"asset_id": asset.id, "license_accepted": False},
            headers={"X-HAYATE-UI": "1"},
        )
        assert rejected.status_code == 422

        unknown = client.post(
            "/api/models/setup/download",
            json={"asset_id": "unknown_asset", "license_accepted": True},
            headers={"X-HAYATE-UI": "1"},
        )
        assert unknown.status_code == 404

        accepted = client.post(
            "/api/models/setup/download",
            json={"asset_id": asset.id, "license_accepted": True},
            headers={"X-HAYATE-UI": "1"},
        )
        assert accepted.status_code == 202
        completed = _wait(service, accepted.json()["id"])
        assert completed["status"] == "completed"
        downloads = client.get("/api/models/setup/downloads").json()["downloads"]
        assert downloads[accepted.json()["id"]]["status"] == "completed"


def test_model_setup_can_explicitly_apply_standard_model_paths(tmp_path):
    asset = _asset()
    service = ModelSetupService(tmp_path, downloader=_downloader, assets=(asset,))
    headers = {"X-HAYATE-UI": "1"}
    with TestClient(create_app(tmp_path, model_setup_service=service)) as client:
        settings = client.get("/api/settings").json()["settings"]
        custom = {**settings, "config_path": str(tmp_path / "custom-models.yaml")}
        assert client.put("/api/settings", json=custom, headers=headers).status_code == 200
        applied = client.post("/api/models/setup/apply-standard", headers=headers)
        assert applied.status_code == 200
        body = applied.json()
        assert body["settings"]["config_path"] == str(tmp_path / "configs" / "models.yaml")
        assert body["settings"]["checkpoint_dir"] == str(tmp_path / "models" / "minimax-h3-snapshot")
        assert body["settings"]["pdd_checkpoint_path"] == str(
            tmp_path / "models" / "lora" / "MiniMax-H3-FL2VA-Acc-8Step.safetensors"
        )
