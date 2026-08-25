from __future__ import annotations

import subprocess
from pathlib import Path

from hayate.backends.minimax_h3.generation import ExternalH3GenerationBackend, GenerationRequest
from hayate.backends.minimax_h3.upstream import REQUIRED_COMPONENTS
from hayate.models import ModelRegistry

from .helpers import write_dummy_safetensors


def _upstream_fixture(root: Path):
    for relative in REQUIRED_COMPONENTS.values():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")


def test_generation_plan_maps_four_single_file_overrides_and_memory_flags(tmp_path):
    upstream = tmp_path / "h3"
    _upstream_fixture(upstream)
    checkpoint = tmp_path / "checkpoint"
    for relative in (
        "transformer/config.json",
        "vae/config.json",
        "audio_vae/config.json",
        "text_encoder/config.json",
        "scheduler/scheduler_config.json",
        "audio_scheduler/scheduler_config.json",
    ):
        path = checkpoint / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    (checkpoint / "tokenizer").mkdir()
    (checkpoint / "processor").mkdir()
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
        upstream,
        ModelRegistry.load(config),
        python=Path(__import__("sys").executable),
    )
    plan = backend.plan(
        GenerationRequest(
            "test",
            checkpoint,
            tmp_path / "out.mp4",
            height=512,
            width=768,
            easycache=True,
        )
    )

    command = list(plan.command)
    assert not plan.executable  # fixture checkout is not the pinned audited git commit
    for flag in ("--dit", "--text_encoder", "--vae", "--audio_vae"):
        assert flag in command
    assert command[command.index("--blocks_to_swap") + 1] == "49"
    assert "--text_encoder_stream" in command
    assert "--vae_tiling" in command
    assert command[command.index("--attn_mode") + 1] == "sdpa"
    assert command[command.index("--infer_steps") + 1] == "50"
    assert plan.environment["HAYATE_EASYCACHE"] == "1"
    assert plan.environment["HAYATE_EASYCACHE_THRESHOLD"] == "0.2"
    assert plan.environment["HAYATE_EASYCACHE_MAX_CONSECUTIVE_SKIPS"] == "2"
    assert plan.environment["HAYATE_VAE_TILE_SIZE"] == "256"


def test_generation_request_rejects_unknown_attention_backend(tmp_path):
    issues = ExternalH3GenerationBackend._validate_request(
        GenerationRequest(
            "test",
            tmp_path / "checkpoint",
            tmp_path / "out.mp4",
            attention_backend="unknown",
        )
    )
    assert "unsupported attention backend: unknown" in issues


def test_generation_request_rejects_non_progressing_vae_tile_geometry(tmp_path):
    issues = ExternalH3GenerationBackend._validate_request(
        GenerationRequest(
            "test",
            tmp_path / "checkpoint",
            tmp_path / "out.mp4",
            vae_tile_size=64,
        )
    )
    assert "vae_tile_size must be a multiple of 16 greater than 64" in issues


def test_optional_python_module_probe_reports_import_failure(tmp_path):
    backend = object.__new__(ExternalH3GenerationBackend)
    backend.python = tmp_path / "python.exe"
    backend._runner = lambda *_args, **_kwargs: subprocess.CompletedProcess(
        [], 1, "", "ModuleNotFoundError: no module named sageattention"
    )
    available, reason = backend._probe_python_module("sageattention")
    assert available is False
    assert reason == "ModuleNotFoundError: no module named sageattention"
