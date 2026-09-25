from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hayate.backends.minimax_h3.generation import (
    ExternalH3GenerationBackend,
    GenerationRequest,
)
from hayate.backends.minimax_h3.nvfp4_conditioner import conditioner_int_mm_enabled
from hayate.loaders import LoaderFactory
from hayate.loaders.base import LoaderStatus
from hayate.models import ModelRegistry
from hayate.models.types import ModelRole, ModelSpec
from hayate.profiles import LARGE_GPU_MIN_VRAM_GIB, get_generation_profile
from hayate.runtime.gpu_devices import GPUDevice
from hayate.webui.model_setup import MODEL_ASSETS
from hayate.webui.server import _large_gpu_issue, create_app

from .helpers import write_dummy_safetensors
from .test_generation import _upstream_fixture

GIB = 1024**3


def _int8_convrot(path: Path, groups: int) -> Path:
    tensors = []
    for index in range(groups):
        tensors += [
            (f"blocks.{index}.fc.weight", "I8", [2, 2]),
            (f"blocks.{index}.fc.weight_scale", "F32", [2]),
            (f"blocks.{index}.fc.comfy_quant", "U8", [8]),
        ]
    return write_dummy_safetensors(path, tensors, {})


def _checkpoint(root: Path) -> Path:
    for relative in (
        "transformer/config.json",
        "vae/config.json",
        "audio_vae/config.json",
        "text_encoder/config.json",
        "scheduler/scheduler_config.json",
        "audio_scheduler/scheduler_config.json",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    (root / "tokenizer").mkdir()
    (root / "processor").mkdir()
    return root


def _plan(tmp_path: Path, **request_fields):
    upstream = tmp_path / "h3"
    _upstream_fixture(upstream)
    model = write_dummy_safetensors(tmp_path / "model.safetensors", [("weight", "F32", [2])])
    config = tmp_path / "models.yaml"
    config.write_text(
        "models:\n  minimax_h3:\n"
        + "".join(
            f"    {role}:\n      path: '{model.as_posix()}'\n"
            for role in ("transformer", "text_encoder", "video_vae", "audio_vae")
        ),
        encoding="utf-8",
    )
    backend = ExternalH3GenerationBackend(
        upstream, ModelRegistry.load(config), python=Path(sys.executable)
    )
    request = GenerationRequest(
        "test", _checkpoint(tmp_path / "checkpoint"), tmp_path / "out.mp4", **request_fields
    )
    return backend.plan(request)


def test_a100_profiles_keep_every_stage_resident():
    for name in ("a100_detail", "a100_quality", "a100_pdd"):
        profile = get_generation_profile(name)
        assert profile.blocks_to_swap == 0
        assert profile.text_encoder_gpu_layers == -1
        assert profile.text_encoder_stream is False
        assert profile.min_vram_gib == LARGE_GPU_MIN_VRAM_GIB
        assert profile.vae_tile_size == 256  # validated tile; 512 ghosted
    detail = get_generation_profile("a100_detail")
    # Same validated cache schedule as fast_sage_detail; only placement changes.
    reference = get_generation_profile("fast_sage_detail")
    for key in ("steps", "attention_backend", "easycache_threshold", "easycache_start", "easycache_end"):
        assert getattr(detail, key) == getattr(reference, key)
    assert detail.int8_fast is True
    assert get_generation_profile("a100_quality").int8_fast is False
    assert get_generation_profile("a100_quality").steps == 50
    pdd = get_generation_profile("a100_pdd")
    assert pdd.pdd and pdd.steps == 9 and pdd.attention_backend == "sdpa"
    # Consumer profiles are unchanged.
    consumer = get_generation_profile("fast_sage_detail")
    assert consumer.text_encoder_gpu_layers == 0 and consumer.text_encoder_stream
    assert consumer.min_vram_gib == 0


def test_command_forwards_resident_conditioner_and_int8_fast(tmp_path):
    command = list(
        _plan(
            tmp_path,
            blocks_to_swap=0,
            text_encoder_gpu_layers=-1,
            text_encoder_stream=False,
            int8_fast=True,
        ).command
    )
    assert command[command.index("--blocks_to_swap") + 1] == "0"
    assert command[command.index("--text_encoder_gpu_layers") + 1] == "-1"
    assert "--text_encoder_stream" not in command
    assert "--int8_fast" in command


def test_default_command_keeps_consumer_streaming(tmp_path):
    command = list(_plan(tmp_path).command)
    assert command[command.index("--text_encoder_gpu_layers") + 1] == "0"
    assert "--text_encoder_stream" in command
    assert "--int8_fast" not in command


def test_invalid_conditioner_layer_count_is_rejected(tmp_path):
    plan = _plan(tmp_path, text_encoder_gpu_layers=-2)
    assert any("text_encoder_gpu_layers" in issue for issue in plan.issues)


@pytest.mark.parametrize(
    ("role", "groups", "supported"),
    [
        (ModelRole.TRANSFORMER, 200, True),  # pruned FL2VA
        (ModelRole.TRANSFORMER, 250, True),  # full FL2VA
        (ModelRole.TRANSFORMER, 199, False),
        (ModelRole.TEXT_ENCODER, 350, True),
        (ModelRole.TEXT_ENCODER, 200, False),
    ],
)
def test_int8_convrot_accepts_only_audited_layouts(tmp_path, role, groups, supported):
    path = _int8_convrot(tmp_path / "model.safetensors", groups)
    validation = LoaderFactory.create(ModelSpec("minimax_h3", role, path)).validate()
    expected = LoaderStatus.SUPPORTED if supported else LoaderStatus.PARTIAL
    assert validation.status is expected


def test_conditioner_int_mm_is_opt_in(monkeypatch):
    monkeypatch.delenv("HAYATE_TEXT_ENCODER_INT_MM", raising=False)
    assert conditioner_int_mm_enabled() is False
    monkeypatch.setenv("HAYATE_TEXT_ENCODER_INT_MM", "1")
    assert conditioner_int_mm_enabled() is True


def _gpu(index: int, vram_gib: int, name: str = "NVIDIA A100-SXM4-80GB") -> GPUDevice:
    return GPUDevice(index, name, vram_gib * GIB, compute_capability="8.0", uuid=f"GPU-{index:08d}")


def test_large_gpu_gate():
    a100 = _gpu(0, 80)
    rtx = _gpu(1, 12, "NVIDIA GeForce RTX 3060")
    need = LARGE_GPU_MIN_VRAM_GIB
    assert _large_gpu_issue("auto", need, [a100]) is None
    assert _large_gpu_issue("auto", need, [_gpu(0, 40, "NVIDIA A100-PCIE-40GB")]) is None
    assert "見つかりません" in _large_gpu_issue("auto", need, [rtx])
    assert "見つかりません" in _large_gpu_issue("auto", need, [])
    # Mixed hosts must pin the job; Auto could otherwise land on the small card.
    assert "指定" in _large_gpu_issue("auto", need, [a100, rtx])
    assert _large_gpu_issue(a100.uuid, need, [a100, rtx]) is None
    assert "RTX 3060" in _large_gpu_issue(rtx.uuid, need, [a100, rtx])


def test_a100_catalog_entries_are_pinned_and_pdd_compatible_layout():
    by_id = {asset.id: asset for asset in MODEL_ASSETS}
    dit = by_id["transformer_int8_pruned"]
    assert dit.role == "transformer" and dit.execution_supported
    assert dit.artifacts[0].size_bytes == 20_970_379_616
    assert by_id["text_encoder_int8_convrot"].artifacts[0].size_bytes == 27_141_342_152
    assert by_id["text_encoder_bf16"].artifacts[0].size_bytes == 51_506_295_256
    for asset_id in ("transformer_int8_pruned", "text_encoder_int8_convrot", "text_encoder_bf16"):
        assert by_id[asset_id].revision == "4cc1d817b6184899b41293954329f576cb5ae86b"


def test_a100_registry_resolves_to_catalog_paths():
    root = Path(__file__).resolve().parents[1]
    registry = ModelRegistry.load(root / "configs" / "models.a100.yaml")
    by_id = {asset.id: asset for asset in MODEL_ASSETS}
    expected = {
        ModelRole.TRANSFORMER: "transformer_int8_pruned",
        ModelRole.TEXT_ENCODER: "text_encoder_int8_convrot",
        ModelRole.VIDEO_VAE: "video_vae_int8_convrot",
        ModelRole.AUDIO_VAE: "audio_vae_fp32",
    }
    for role, asset_id in expected.items():
        path = registry.get("minimax_h3", role).path
        assert path == (root / "models" / by_id[asset_id].artifacts[0].relative_path).resolve()


def test_apply_a100_paths_and_report_active_configuration(tmp_path):
    headers = {"X-HAYATE-UI": "1"}
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/models/setup").json()["active_configuration"] == "standard"
        result = client.post(
            "/api/models/setup/apply-standard", json={"configuration": "a100"}, headers=headers
        )
        assert result.status_code == 200
        assert Path(result.json()["settings"]["config_path"]).name == "models.a100.yaml"
        assert (tmp_path / "configs" / "models.a100.yaml").is_file()
        assert client.get("/api/models/setup").json()["active_configuration"] == "a100"
        back = client.post("/api/models/setup/apply-standard", json={}, headers=headers)
        assert Path(back.json()["settings"]["config_path"]).name == "models.yaml"


def test_a100_profile_is_refused_without_a_large_gpu(tmp_path, monkeypatch):
    from hayate.webui import server

    monkeypatch.setattr(server, "discover_gpu_devices", lambda: [_gpu(0, 12, "NVIDIA GeForce RTX 3060")])
    with TestClient(create_app(tmp_path)) as client:
        response = client.post(
            "/api/jobs",
            json={"prompt": "a test", "profile": "a100_detail"},
            headers={"X-HAYATE-UI": "1"},
        )
    assert response.status_code == 422
    assert "GiB" in response.json()["detail"]["issues"][0]
