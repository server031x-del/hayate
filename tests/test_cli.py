from __future__ import annotations

from hayate.cli.main import build_parser, main, run_generate

from .helpers import write_dummy_safetensors


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
        raise StopAfterProfile

    monkeypatch.setattr("hayate.cli.main.ExternalH3GenerationBackend", stop_backend)
    try:
        run_generate(args, None)
    except StopAfterProfile:
        pass
    else:
        raise AssertionError("profile was not applied before backend construction")
