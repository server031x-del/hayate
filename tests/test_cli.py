from __future__ import annotations

from hayate.cli.main import main

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

