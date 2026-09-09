"""Explicit ComfyUI FastH3 route; no inference math is implemented here."""
from __future__ import annotations

import json
import os
from pathlib import Path

from hayate.backends.minimax_h3.generation import GenerationPlan, GenerationRequest

MODEL_IDS = ("transformer_fastvideo_vsa_4step", "text_encoder_nvfp4_awq",
             "video_vae_int8_convrot", "audio_vae_fp32")
MODEL = "minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors"
ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO = "minimax_h3_audio_vae_fp32.safetensors"


def build_graph(prompt: str, seed: int, width: int, height: int, frames: int,
                keep: float = 10, tile_batch: int = 2) -> dict:
    if keep not in (5, 7.5, 10) or tile_batch not in (1, 2):
        raise ValueError("Unsupported VSA keep / tile batch")
    if not prompt.strip() or width % 32 or height % 32 or min(width, height) < 256 or frames % 17 != 5:
        raise ValueError("Invalid FastH3 dimensions, frames or prompt")
    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}
    return {
        "1": node("UNETLoader", unet_name=MODEL, weight_dtype="default"),
        "2": node("MiniMaxH3SigmaShift", model=["1", 0], shift_video=12., shift_audio=3.),
        "17": node("MiniMaxChunkFeedForward", model=["2", 0], chunks=2, seq_threshold=4096),
        "3": node("SolAttnMiniMax", model=["17", 0], selection="VSA (FastVideo)",
                  **{"selection.vsa_keep_percent": keep}, start_percent=0., end_percent=1.,
                  min_tokens=0, sink_conditioning="exact_kv_and_rows", verbose=True),
        "4": node("CLIPLoader", clip_name=ENCODER, type="minimax", device="default"),
        "5": node("VAELoader", vae_name=VIDEO),
        "6": node("VAELoader", vae_name=AUDIO),
        "7": node("MiniMaxH3ImageToVideo", clip=["4", 0], vae=["5", 0],
                  prompt=prompt, width=width, height=height, length=frames),
        "8": node("RandomNoise", noise_seed=seed),
        "9": node("BasicGuider", model=["3", 0], conditioning=["7", 0]),
        "10": node("KSamplerSelect", sampler_name="euler"),
        "11": node("ManualSigmas", sigmas="0.9999166, 0.9728326, 0.9230769, 0.8, 0.0"),
        "12": node("SamplerCustomAdvanced", noise=["8", 0], guider=["9", 0],
                   sampler=["10", 0], sigmas=["11", 0], latent_image=["7", 1]),
        "13": node("MiniMaxH3FastVAEDecode", samples=["12", 0], vae=["5", 0], tile_batch_size=tile_batch),
        "14": node("VAEDecodeAudio", samples=["12", 0], vae=["6", 0]),
        "15": node("CreateVideo", images=["13", 0], audio=["14", 0], fps=24., bit_depth=8, color_space="sRGB"),
        "16": node("SaveVideo", video=["15", 0], filename_prefix="hayate", format="mp4", codec="h264"),
    }


def runtime_paths(root: Path):
    runtime = root / "upstream" / "comfy-fasth3"
    python = runtime / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return runtime, python


def readiness(root: Path, assets: list[dict]) -> dict:
    runtime, python = runtime_paths(root)
    installed = all(p.is_file() for p in (runtime / "main.py", python, runtime / "hayate-ready.json"))
    by_id = {a["id"]: a for a in assets}
    missing = [key for key in MODEL_IDS if by_id.get(key, {}).get("status") != "verified"]
    return {"ready": installed and not missing, "runtime_ready": installed,
            "missing_models": missing, "runtime_path": str(runtime),
            "message": "FastH3モデルと実行環境を準備済み（GPU実行は生成時に確認）" if installed and not missing
            else "ColabのFastH3環境セルを実行してください" if not installed
            else "FastH3構成のモデル取得・検証を完了してください"}


class ComfyFastH3Backend:
    def __init__(self, root: Path, assets: list[dict], keep=10., tile_batch=2):
        self.root, self.assets = root, assets
        self.keep, self.tile_batch = keep, tile_batch

    def plan(self, request: GenerationRequest) -> GenerationPlan:
        status = readiness(self.root, self.assets)
        issues = [] if status["ready"] else [status["message"]]
        if request.task not in ("auto", "t2va") or request.image_path or request.last_image_path or request.references:
            issues.append("FastH3はテキストから動画＋音声のみ対応です。開始画像を外してください")
        runtime, python = runtime_paths(self.root)
        graph = build_graph(request.prompt, request.seed, request.width or 512,
                            request.height or 512, request.frames, self.keep, self.tile_batch)
        command = (str(python), "-m", "hayate.backends.minimax_h3.comfy_worker",
                   "--runtime", str(runtime), "--output", str(request.output),
                   "--graph", json.dumps(graph, ensure_ascii=True))
        return GenerationPlan(request, command, {"PYTHONPATH": str(self.root)}, None,
                              tuple(issues), ("FastH3 T2VA実験経路。画質・速度はGPU上で要比較",),
                              working_directory=self.root, backend="comfy_fasth3")

    def probe_cli(self):
        return runtime_paths(self.root)[1].is_file(), "ComfyUI runtime"
