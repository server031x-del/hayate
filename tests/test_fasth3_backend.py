from __future__ import annotations

from types import SimpleNamespace

from hayate.backends.minimax_h3.fasth3_backend import FastH3GenerationBackend
from hayate.backends.minimax_h3.generation import GenerationRequest

from .helpers import write_dummy_safetensors


def _model_dir(tmp_path):
    model = tmp_path / "fastvideo"
    (model / "transformer").mkdir(parents=True)
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    (model / "transformer" / "config.json").write_text("{}", encoding="utf-8")
    write_dummy_safetensors(model / "transformer" / "model.safetensors", [("w", "F32", [1])])
    return model


def _runner(command, **kwargs):
    if "importlib.util" in command:
        return SimpleNamespace(returncode=0, stdout="available\n", stderr="")
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def test_fasth3_backend_builds_separate_vsa_command(tmp_path, monkeypatch):
    model = _model_dir(tmp_path)
    monkeypatch.setattr(
        "hayate.backends.minimax_h3.fasth3_backend.probe_fastvideo_runtime",
        lambda *_args, **_kwargs: {"available": True, "reason": "available"},
    )
    request = GenerationRequest(
        prompt="a cinematic car commercial",
        checkpoint_dir=model,
        output=tmp_path / "out.mp4",
        task="t2va",
        height=512,
        width=512,
        frames=124,
        steps=5,
    )

    plan = FastH3GenerationBackend(model, runner=_runner).plan(request)

    assert plan.executable is True
    assert plan.backend == "fastvideo_vsa"
    assert plan.upstream is None
    assert "fasth3_entrypoint" in " ".join(plan.command)
    assert "--vsa-kernel triton" in " ".join(plan.command)


def test_fasth3_backend_fast_profile_requires_and_selects_blackwell_recipe(tmp_path, monkeypatch):
    model = _model_dir(tmp_path)
    monkeypatch.setattr(
        "hayate.backends.minimax_h3.fasth3_backend.probe_fastvideo_runtime",
        lambda *_args, **_kwargs: {
            "available": True,
            "reason": "available",
            "gpu_capabilities": ["10.0"],
            "gpu_memory_bytes": [80 * 1024**3],
            "vsa_sm100a": True,
            "fa4": True,
        },
    )
    request = GenerationRequest(
        prompt="a cinematic car commercial",
        checkpoint_dir=model,
        output=tmp_path / "out.mp4",
        task="t2va",
        height=512,
        width=512,
        frames=124,
        steps=5,
    )

    plan = FastH3GenerationBackend(
        model,
        performance_profile="fast",
        runner=_runner,
    ).plan(request)

    command = " ".join(plan.command)
    assert plan.executable is True
    assert "--vsa-kernel sm100a" in command
    assert "--profile all" in command
    assert "--fa4" in command
    assert "--inference-torch-compile" in command
    assert "--parallel-vae" in command
    assert "--no-low-memory" in command


def test_fasth3_backend_fast_profile_fails_closed_on_non_blackwell(tmp_path, monkeypatch):
    model = _model_dir(tmp_path)
    monkeypatch.setattr(
        "hayate.backends.minimax_h3.fasth3_backend.probe_fastvideo_runtime",
        lambda *_args, **_kwargs: {
            "available": True,
            "reason": "available",
            "gpu_capabilities": ["8.0"],
            "vsa_sm100a": False,
            "fa4": False,
        },
    )
    request = GenerationRequest(
        prompt="test",
        checkpoint_dir=model,
        output=tmp_path / "out.mp4",
        task="t2va",
        steps=5,
    )

    plan = FastH3GenerationBackend(
        model,
        performance_profile="fast",
        runner=_runner,
    ).plan(request)

    assert plan.executable is False
    assert any("Blackwell" in issue for issue in plan.issues)


def test_fasth3_backend_rejects_image_to_video(tmp_path, monkeypatch):
    model = _model_dir(tmp_path)
    image = tmp_path / "start.png"
    image.write_bytes(b"image")
    monkeypatch.setattr(
        "hayate.backends.minimax_h3.fasth3_backend.probe_fastvideo_runtime",
        lambda *_args, **_kwargs: {"available": True, "reason": "available"},
    )
    request = GenerationRequest(
        prompt="test",
        checkpoint_dir=model,
        output=tmp_path / "out.mp4",
        image_path=image,
        steps=5,
    )

    plan = FastH3GenerationBackend(model, runner=_runner).plan(request)

    assert plan.executable is False
    assert any("I2V" in issue for issue in plan.issues)


def test_fasth3_cli_execution_writes_manifest_and_releases_lease(tmp_path, monkeypatch):
    model = _model_dir(tmp_path)
    monkeypatch.setattr(
        "hayate.backends.minimax_h3.fasth3_backend.probe_fastvideo_runtime",
        lambda *_args, **_kwargs: {"available": True, "reason": "available"},
    )
    request = GenerationRequest(
        prompt="test",
        checkpoint_dir=model,
        output=tmp_path / "out.mp4",
        task="t2va",
        steps=5,
    )
    backend = FastH3GenerationBackend(model, runner=_runner)
    plan = backend.plan(request)
    result = backend.execute(plan)

    assert result.returncode == 0
    assert result.output == tmp_path / "out.mp4"
    assert (tmp_path / "out.mp4.hayate.log").is_file()
    manifest = (tmp_path / "out.mp4.hayate.json").read_text(encoding="utf-8")
    assert '"backend": "fastvideo_vsa"' in manifest
