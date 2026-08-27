from __future__ import annotations

from pathlib import Path

import pytest

from hayate.cli.main import build_parser, main, run_generate

from .helpers import write_dummy_safetensors


def test_webui_defaults_to_lan_bind_with_loopback_opt_out():
    args = build_parser().parse_args(["webui"])
    assert args.host == "0.0.0.0"
    assert args.allow_network is True
    assert args.local_only is False

    local = build_parser().parse_args(["webui", "--local-only"])
    assert local.local_only is True


def test_cli_startup_and_inspect(tmp_path, capsys):
    model = write_dummy_safetensors(tmp_path / "audio.safetensors", [("weight", "F32", [2])], {})
    config = tmp_path / "models.yaml"
    config.write_text(
        f"models:\n  minimax_h3:\n    audio_vae:\n      path: '{model.as_posix()}'\n",
        encoding="utf-8",
    )
    exit_code = main(
        ["inspect", "--config", str(config), "--json", "--no-save-benchmark"]
    )
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "audio_vae" in output
    assert "FP32" in output


def test_rtx3060_fast_profile_resolves_validated_generation_settings(monkeypatch, tmp_path):
    args = build_parser().parse_args(
        [
            "generate",
            "--prompt",
            "test",
            "--ckpt-dir",
            str(tmp_path / "checkpoint"),
            "--output",
            str(tmp_path / "out.mp4"),
            "--attention-backend",
            "sageattn",
            "--rtx3060-fast",
            "--dry-run",
        ]
    )

    class StopAfterProfile(RuntimeError):
        pass

    def stop_backend(*_args, **_kwargs):
        assert args.steps == 20
        assert args.easycache is True
        assert args.easycache_threshold == 0.4
        assert args.easycache_max_consecutive_skips == 2
        assert args.blocks_to_swap == 49
        assert args.activation_chunk_rows == 32768
        assert args.attention_backend == "sdpa"
        assert args.vae_tile_size == 256
        raise StopAfterProfile

    monkeypatch.setattr("hayate.cli.main.ExternalH3GenerationBackend", stop_backend)
    try:
        run_generate(args, None)
    except StopAfterProfile:
        pass
    else:
        raise AssertionError("profile was not applied before backend construction")


def test_rtx3060_fast_sage_profile_selects_sageattention(monkeypatch, tmp_path):
    args = build_parser().parse_args(
        [
            "generate",
            "--prompt",
            "test",
            "--ckpt-dir",
            str(tmp_path / "checkpoint"),
            "--output",
            str(tmp_path / "out.mp4"),
            "--vae-tile-size",
            "512",
            "--rtx3060-fast-sage",
            "--dry-run",
        ]
    )

    class StopAfterProfile(RuntimeError):
        pass

    def stop_backend(*_args, **_kwargs):
        assert args.steps == 20
        assert args.easycache is True
        assert args.attention_backend == "sageattn"
        assert args.vae_tile_size == 256
        raise StopAfterProfile

    monkeypatch.setattr("hayate.cli.main.ExternalH3GenerationBackend", stop_backend)
    try:
        run_generate(args, None)
    except StopAfterProfile:
        pass
    else:
        raise AssertionError("SageAttention profile was not applied")


def test_rtx3060_pdd_profile_uses_sdpa(monkeypatch, tmp_path):
    args = build_parser().parse_args(
        [
            "generate",
            "--prompt",
            "test",
            "--ckpt-dir",
            str(tmp_path / "checkpoint"),
            "--output",
            str(tmp_path / "out.mp4"),
            "--rtx3060-pdd",
            "--dry-run",
        ]
    )

    class StopAfterProfile(RuntimeError):
        pass

    def stop_backend(*_args, **_kwargs):
        assert args.steps == 9
        assert args.easycache is False
        assert args.attention_backend == "sdpa"
        assert args.pdd_checkpoint == Path(
            "models/lora/MiniMax-H3-FL2VA-Acc-8Step.safetensors"
        )
        raise StopAfterProfile

    monkeypatch.setattr("hayate.cli.main.ExternalH3GenerationBackend", stop_backend)
    try:
        run_generate(args, None)
    except StopAfterProfile:
        pass
    else:
        raise AssertionError("PDD SDPA profile was not applied")


def test_rtx3060_fast_sage_detail_profile_protects_final_steps(monkeypatch, tmp_path):
    args = build_parser().parse_args(
        [
            "generate",
            "--prompt",
            "test",
            "--ckpt-dir",
            str(tmp_path / "checkpoint"),
            "--output",
            str(tmp_path / "out.mp4"),
            "--rtx3060-fast-sage-detail",
            "--dry-run",
        ]
    )

    class StopAfterProfile(RuntimeError):
        pass

    def stop_backend(*_args, **_kwargs):
        assert args.steps == 20
        assert args.attention_backend == "sageattn"
        assert args.easycache is True
        assert args.easycache_threshold == 0.4
        assert args.easycache_start == 0.15
        assert args.easycache_end == 0.85
        assert args.easycache_max_consecutive_skips == 2
        assert args.vae_tile_size == 256
        raise StopAfterProfile

    monkeypatch.setattr("hayate.cli.main.ExternalH3GenerationBackend", stop_backend)
    with pytest.raises(StopAfterProfile):
        run_generate(args, None)
