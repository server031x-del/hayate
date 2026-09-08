from __future__ import annotations

import json
from types import SimpleNamespace

from hayate.backends.minimax_h3.fasth3 import (
    FASTVIDEO_FASTH3_REPOSITORY,
    FASTVIDEO_FASTH3_REVISION,
    FASTVIDEO_FASTH3_REQUIRED_FILES,
    KIJAI_FASTH3_FILENAME,
    fasth3_preflight,
    inspect_fasth3_checkpoint,
    probe_fastvideo_runtime,
    validate_fastvideo_model_directory,
    fast_profile_issues,
)

from .helpers import write_dummy_safetensors


def _fasth3_fixture(tmp_path):
    return write_dummy_safetensors(
        tmp_path / KIJAI_FASTH3_FILENAME,
        [
            ("blocks.0.attn.qkv_proj.weight", "I8", [4]),
            ("blocks.0.attn.out_proj.weight", "I8", [4]),
            ("blocks.0.attn.to_gate_compress.weight", "I8", [4]),
            ("blocks.0.attn.to_gate_compress.weight_scale", "F32", [1]),
            ("blocks.0.attn.to_gate_compress.comfy_quant", "U8", [1]),
        ],
    )


def test_fasth3_header_detects_vsa_gate_and_comfy_layout(tmp_path):
    path = _fasth3_fixture(tmp_path)

    info = inspect_fasth3_checkpoint(path)

    assert info.detected is True
    assert info.layout == "comfy_single_file_vsa"
    assert info.gate_layer_count == 1
    assert info.comfy_quant_count == 1
    assert info.int8_weight_count == 3
    assert info.is_kijai_fastvideo_single_file is True


def test_fasth3_preflight_never_treats_single_file_as_fastvideo_directory(tmp_path, monkeypatch):
    path = _fasth3_fixture(tmp_path)
    python = tmp_path / "python.exe"
    python.write_bytes(b"python")
    monkeypatch.setattr(
        "hayate.backends.minimax_h3.fasth3.probe_fastvideo_runtime",
        lambda *_args, **_kwargs: {"python": str(python), "available": True, "reason": "available"},
    )

    result = fasth3_preflight(
        path,
        python=python,
        model_directory=tmp_path / "fastvideo",
        task="t2va",
    )

    assert result["ready"] is False
    assert any("単一safetensors" in issue for issue in result["issues"])


def test_probe_fastvideo_runtime_uses_the_selected_interpreter(tmp_path):
    python = tmp_path / "python.exe"
    python.write_bytes(b"python")
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="missing\n", stderr="")

    result = probe_fastvideo_runtime(python, runner=runner)

    assert result["available"] is False
    assert calls[0][0][0] == str(python.resolve())


def test_probe_fastvideo_runtime_reports_component_and_cuda_contract(tmp_path):
    python = tmp_path / "python.exe"
    python.write_bytes(b"python")

    def runner(command, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout='{"fastvideo":true,"api":true,"minimax_h3":true,"cuda":false,"available":false}\n',
            stderr="",
        )

    result = probe_fastvideo_runtime(python, runner=runner)

    assert result["available"] is False
    assert result["cuda"] is False
    assert "CUDA" in result["reason"]


def test_fastvideo_directory_accepts_official_modular_metadata(tmp_path):
    model = tmp_path / "fastvideo"
    transformer = model / "transformer"
    transformer.mkdir(parents=True)
    (model / "modular_model_index.json").write_text("{}", encoding="utf-8")
    (transformer / "config.json").write_text("{}", encoding="utf-8")
    write_dummy_safetensors(transformer / "diffusion_pytorch_model.safetensors", [("w", "F32", [1])])

    assert validate_fastvideo_model_directory(model) == ()


def test_fastvideo_defaults_pin_current_v1_snapshot():
    assert FASTVIDEO_FASTH3_REPOSITORY == "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"
    assert FASTVIDEO_FASTH3_REVISION == "5ea076f35b84da4c3c82217112fa733d8eea2ae1"


def test_fast_profile_issues_are_empty_for_blackwell_kernel_and_fa4():
    assert fast_profile_issues(
        {
            "gpu_capabilities": ["10.0"],
            "gpu_memory_bytes": [80 * 2**30],
            "vsa_sm100a": True,
            "fa4": True,
        }
    ) == ()


def test_fast_profile_issues_explain_missing_blackwell_dependencies():
    issues = fast_profile_issues(
        {"gpu_capabilities": ["8.6"], "vsa_sm100a": False, "fa4": False}
    )
    assert len(issues) == 4
    assert any("Blackwell" in issue for issue in issues)
    assert any("sm100a" in issue for issue in issues)
    assert any("FlashAttention" in issue for issue in issues)
    assert any("80 GiB" in issue for issue in issues)


def test_fasth3_preflight_ready_depends_on_directory_runtime_not_kijai_file(tmp_path, monkeypatch):
    model = tmp_path / "fastvideo"
    transformer = model / "transformer"
    transformer.mkdir(parents=True)
    (model / "fastvideo_inference.json").write_text("{}", encoding="utf-8")
    (transformer / "config.json").write_text("{}", encoding="utf-8")
    write_dummy_safetensors(transformer / "diffusion_pytorch_model.safetensors", [("w", "F32", [1])])
    python = tmp_path / "python.exe"
    python.write_bytes(b"python")
    monkeypatch.setattr(
        "hayate.backends.minimax_h3.fasth3.probe_fastvideo_runtime",
        lambda *_args, **_kwargs: {"available": True, "reason": "available"},
    )

    result = fasth3_preflight(None, python=python, model_directory=model, task="t2va")

    assert result["ready"] is True
    assert result["kijai_single_file_runnable"] is False


def test_fastvideo_strict_contract_rejects_stale_weight_index(tmp_path):
    model = tmp_path / "fastvideo"
    for relative in FASTVIDEO_FASTH3_REQUIRED_FILES:
        target = model / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith(".safetensors"):
            write_dummy_safetensors(target, [("weight", "F32", [1])])
        elif relative.endswith(".index.json"):
            target.write_text(json.dumps({"weight_map": {"weight": "missing.safetensors"}}), encoding="utf-8")
        elif relative == "modular_model_index.json":
            target.write_text(json.dumps({"_class_name": "MiniMaxH3ModularPipeline"}), encoding="utf-8")
        else:
            target.write_text("{}", encoding="utf-8")
    # The validator also checks that indexed component directories contain at
    # least one shard, so add a harmless local shard for each component.
    for directory in (model / "transformer", model / "text_encoder", model / "vae"):
        write_dummy_safetensors(directory / "present.safetensors", [("weight", "F32", [1])])

    issues = validate_fastvideo_model_directory(model, strict=True)

    assert any("index references missing shard" in issue for issue in issues)
