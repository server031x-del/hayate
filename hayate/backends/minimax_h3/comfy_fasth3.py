"""Explicit ComfyUI FastH3 route; no inference math is implemented here."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from hayate.backends.minimax_h3.generation import GenerationPlan, GenerationRequest

MODEL_IDS = ("transformer_fastvideo_vsa_4step", "text_encoder_nvfp4_awq",
             "video_vae_int8_convrot", "audio_vae_fp32")
FL2VA_MODEL_IDS = ("transformer_w4a8", "text_encoder_nvfp4_awq",
                   "video_vae_int8_convrot", "audio_vae_fp32")
MODEL = "minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors"
FL2VA_MODEL = "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors"
ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO = "minimax_h3_audio_vae_fp32.safetensors"


def build_graph(prompt: str, seed: int, width: int, height: int, frames: int,
                keep: float = 10, tile_batch: int = 2, first_image=False, last_image=False,
                mode: str = "fasth3", steps: int = 50) -> dict:
    if keep not in (5, 7.5, 10) or tile_batch not in (1, 2):
        raise ValueError("Unsupported VSA keep / tile batch")
    if not prompt.strip() or width % 32 or height % 32 or min(width, height) < 256 or frames % 17 != 5:
        raise ValueError("Invalid FastH3 dimensions, frames or prompt")
    if mode not in ("fasth3", "fl2va") or (mode == "fasth3" and (first_image or last_image)):
        raise ValueError("FastH3 supports text only; use FL2VA for images")
    if mode == "fl2va" and (not first_image or not 2 <= steps <= 100):
        raise ValueError("FL2VA requires a first image and valid sampling steps")
    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}
    graph = {
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
    for enabled, nid, field, name in ((first_image, "18", "first_frame", "first.png"), (last_image, "19", "last_frame", "last.png")):
        if enabled:
            graph[nid] = node("LoadImage", image=name)
            graph["7"]["inputs"][field] = [nid, 0]
    if mode == "fl2va":
        # The FastVideo distilled sigmas and VSA patch belong to its T2VA
        # checkpoint. FL2VA uses the matching base DiT and a normal schedule.
        graph["1"]["inputs"]["unet_name"] = FL2VA_MODEL
        del graph["2"], graph["3"]
        graph["17"]["inputs"]["model"] = ["1", 0]
        graph["9"]["inputs"]["model"] = ["17", 0]
        graph["10"]["inputs"]["sampler_name"] = "res_multistep"
        graph["11"] = node("BasicScheduler", model=["17", 0], scheduler="simple",
                            steps=steps, denoise=1.0)
    return graph



def runtime_paths(root: Path):
    runtime = root / "upstream" / "comfy-fasth3"
    python = runtime / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return runtime, python


def readiness(root: Path, assets: list[dict], mode: str = "fasth3") -> dict:
    runtime, python = runtime_paths(root)
    installed = all(p.is_file() for p in (runtime / "main.py", python, runtime / "hayate-ready.json"))
    by_id = {a["id"]: a for a in assets}
    if mode not in ("fasth3", "fl2va"):
        raise ValueError("Unknown ComfyUI model mode")
    missing = [key for key in (FL2VA_MODEL_IDS if mode == "fl2va" else MODEL_IDS)
               if by_id.get(key, {}).get("status") != "verified"]
    label = "FL2VA" if mode == "fl2va" else "FastH3"
    return {"ready": installed and not missing, "runtime_ready": installed,
            "missing_models": missing, "runtime_path": str(runtime),
            "message": f"{label}モデルと実行環境を準備済み（GPU実行は生成時に確認）" if installed and not missing
            else "ColabのFastH3環境セルを実行してください" if not installed
            else f"{label}構成のモデル取得・検証を完了してください"}


class ComfyFastH3Backend:
    def __init__(self, root: Path, assets: list[dict], keep=10., tile_batch=2, mode="fasth3"):
        self.root, self.assets = root, assets
        self.keep, self.tile_batch = keep, tile_batch
        self.mode = mode

    def plan(self, request: GenerationRequest) -> GenerationPlan:
        status = readiness(self.root, self.assets, self.mode)
        issues = [] if status["ready"] else [status["message"]]
        if request.references:
            issues.append("参照画像タスクは未対応です。開始画像を使用してください")
        if self.mode == "fasth3" and (request.image_path or request.last_image_path or request.task not in ("auto", "t2va")):
            issues.append("FastH3 4-Stepはテキスト専用です。画像を使う場合は『画像優先 FL2VA』を選択してください")
        if self.mode == "fl2va" and (not request.image_path or request.task not in ("auto", "fl2va")):
            issues.append("画像優先 FL2VAには開始画像とI2Vタスクが必要です")
        if request.last_image_path and not request.image_path:
            issues.append("開始画像も指定してください")
        for image in (request.image_path, request.last_image_path):
            if image and not Path(image).is_file():
                issues.append("画像が見つかりません")
        if self.mode == "fl2va":
            for label, path in (("開始", request.image_path), ("終了", request.last_image_path)):
                if not path or not Path(path).is_file():
                    continue
                try:
                    with Image.open(path) as image:
                        ratio = image.width / image.height
                except (UnidentifiedImageError, OSError, ZeroDivisionError):
                    issues.append(f"{label}画像を読み取れません")
                    continue
                if abs(math.log(((request.width or 512) / (request.height or 512)) / ratio)) > 0.04:
                    issues.append(f"{label}画像と出力の縦横比が異なります。『画像に合わせる』で解像度を調整してください")
        runtime, python = runtime_paths(self.root)
        graph = build_graph(request.prompt, request.seed, request.width or 512,
                            request.height or 512, request.frames, self.keep, self.tile_batch,
                            self.mode == "fl2va", bool(request.last_image_path) if self.mode == "fl2va" else False,
                            self.mode, request.steps)
        command = (str(python), "-m", "hayate.backends.minimax_h3.comfy_worker",
                   "--runtime", str(runtime), "--output", str(request.output),
                   "--graph", json.dumps(graph, ensure_ascii=True))
        for flag, image in (("--first-image", request.image_path), ("--last-image", request.last_image_path)):
            if image:
                command += (flag, str(image))
        return GenerationPlan(request, command, {"PYTHONPATH": str(self.root)}, None,
                              tuple(issues), (("FL2VA開始画像経路。人物の全編固定は保証されません" if self.mode == "fl2va"
                                               else "FastH3 T2VA実験経路。画質・速度はGPU上で要比較"),),
                              working_directory=self.root, backend="comfy_fasth3")

    def probe_cli(self):
        return runtime_paths(self.root)[1].is_file(), "ComfyUI runtime"
