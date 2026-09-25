"""Read-only readiness check for the direct H3 engine used by the WebUI."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from hayate.backends.minimax_h3 import ExternalH3GenerationBackend, GenerationRequest
from hayate.models import ModelRegistry


STANDARD_ASSETS = frozenset({
    "transformer_w4a8", "text_encoder_nvfp4_awq", "video_vae_int8_convrot",
    "audio_vae_fp32", "checkpoint_support",
})
A100_ASSETS = frozenset({
    "transformer_int8_pruned", "video_vae_int8_convrot", "audio_vae_fp32",
    "checkpoint_support",
})
GENERATION_MODULES = (
    "accelerate", "av", "diffusers", "easydict", "einops", "imageio_ffmpeg",
    "numpy", "omegaconf", "cv2", "PIL", "pydantic", "tiktoken", "torch",
    "torchaudio", "torchvision", "transformers", "comfy_kitchen", "safetensors",
)


def _missing_modules(python: Path) -> list[str]:
    script = (
        "import importlib.util,json; "
        f"names={GENERATION_MODULES!r}; "
        "print(json.dumps([name for name in names if importlib.util.find_spec(name) is None]))"
    )
    result = subprocess.run(
        [str(python), "-c", script], capture_output=True, text=True, check=False, timeout=15,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "Python probe failed")
    return json.loads(result.stdout.strip())


def native_readiness(settings, assets: list[dict], configuration: str, root: Path) -> dict:
    """Inspect the configured registry without loading weights or allocating a GPU."""
    ready_ids = {asset["id"] for asset in assets if asset.get("status") == "verified"}
    required = A100_ASSETS if configuration == "a100" else STANDARD_ASSETS
    if configuration == "a100":
        models_ready = required <= ready_ids and bool(
            {"text_encoder_int8_convrot", "text_encoder_bf16"} & ready_ids
        )
    else:
        models_ready = required <= ready_ids
    if configuration not in {"standard", "a100"}:
        return {"ready": False, "sage_ready": False, "issues": ["カスタムモデル定義は生成前チェックで確認してください"]}
    if not models_ready:
        return {"ready": False, "sage_ready": False, "issues": ["選択中の構成のモデル取得・検証が未完了です"]}

    try:
        python = Path(settings.python_path)
        if not python.is_file():
            raise RuntimeError(f"Pythonが見つかりません: {python}")
        missing = _missing_modules(python)
        if missing:
            return {
                "ready": False, "sage_ready": False,
                "issues": ["生成用Pythonに不足: " + ", ".join(missing)],
            }
        backend = ExternalH3GenerationBackend(
            settings.upstream_path, ModelRegistry.load(settings.config_path), python=python,
        )
        plan = backend.plan(GenerationRequest(
            prompt="readiness", checkpoint_dir=Path(settings.checkpoint_dir),
            output=root / "outputs" / "readiness.mp4", steps=50,
        ))
        issues = list(plan.issues)
        if plan.upstream is not None and not plan.upstream.valid:
            issues.append("監査済みの上流H3が未配置か、コミットが一致しません")
        if issues:
            return {"ready": False, "sage_ready": False, "issues": issues}
        sage_ready, sage_reason = backend._probe_python_module("sageattention")
        return {
            "ready": True, "sage_ready": sage_ready, "issues": [],
            "sage_issue": "" if sage_ready else f"SageAttentionが利用できません: {sage_reason}",
        }
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        return {"ready": False, "sage_ready": False, "issues": [str(exc)]}
