"""Queue an I2V smoke/full test through ComfyUI's public API.

The graph mirrors the patched HAYATE subgraph, but is flat API JSON so it can
be validated without relying on frontend-only subgraph serialization.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8189
DEFAULT_OUTPUT_ROOT = Path(r"M:\Project\HAYATE\outputs")
DEFAULT_INPUT = "i2v_car_reference.jpg"
DEFAULT_RESULT_PATH = Path(r"M:\Project\HAYATE\data\comfyui-user\fasth3_vsa_i2v_test_result.json")
MODEL = "minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"


def _http_json(base_url: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(base_url + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI API {exc.code} {path}: {detail}") from exc
    return json.loads(raw.decode("utf-8")) if raw else {}


def build_prompt(*, input_image: str, width: int, height: int, length: int,
                 seed: int, prompt_text: str) -> dict[str, Any]:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": input_image}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL, "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "6": {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {"model": ["2", 0], "shift_video": 12.0, "shift_audio": 3.0},
        },
        "7": {
            "class_type": "H3VSA",
            "inputs": {
                "model": ["6", 0],
                "gate_file": "<model-embedded-gates>",
                "topk_ratio": 0.10,
                "min_tokens": 4096,
            },
        },
        "8": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["3", 0],
                "vae": ["4", 0],
                "first_frame": ["1", 0],
                "prompt": prompt_text,
                "width": width,
                "height": height,
                "length": length,
            },
        },
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "11": {
            "class_type": "BasicScheduler",
            "inputs": {"model": ["7", 0], "scheduler": "simple", "steps": 4, "denoise": 1.0},
        },
        "12": {"class_type": "BasicGuider", "inputs": {"model": ["7", 0], "conditioning": ["8", 0]}},
        "13": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["9", 0],
                "guider": ["12", 0],
                "sampler": ["10", 0],
                "sigmas": ["11", 0],
                "latent_image": ["8", 1],
            },
        },
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["4", 0]}},
        "15": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["13", 0], "vae": ["5", 0]}},
        "16": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["14", 0], "audio": ["15", 0], "fps": 24, "bit_depth": 8, "color_space": "sRGB"},
        },
        "17": {
            "class_type": "SaveVideo",
            "inputs": {"video": ["16", 0], "filename_prefix": "video/hayate_fasth3_vsa_i2v", "format": "mp4", "codec": "h264"},
        },
    }


def _media_paths(entry: dict[str, Any], output_root: Path) -> list[Path]:
    result: list[Path] = []
    root = output_root.resolve()
    for node_output in entry.get("outputs", {}).values():
        for key in ("videos", "gifs", "images"):
            for media in node_output.get(key, []):
                filename = media.get("filename")
                if not filename:
                    continue
                candidate = (root / media.get("subfolder", "") / filename).resolve()
                if root not in candidate.parents:
                    continue
                result.append(candidate)
    return result


def run(args: argparse.Namespace) -> int:
    base = f"http://{args.host}:{args.port}"
    graph = build_prompt(
        input_image=args.input_image,
        width=args.width,
        height=args.height,
        length=args.length,
        seed=args.seed,
        prompt_text=args.prompt,
    )
    # Explicit node inventory check catches a missing custom node before a
    # large checkpoint load is started.
    info = _http_json(base, "/object_info")
    for node_name in ("H3VSA", "MiniMaxH3SigmaShift", "MiniMaxH3ImageToVideo"):
        if node_name not in info:
            raise RuntimeError(f"required node missing from /object_info: {node_name}")
    queued = _http_json(base, "/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())})
    if queued.get("node_errors"):
        raise RuntimeError(f"ComfyUI rejected nodes: {json.dumps(queued['node_errors'], ensure_ascii=False)}")
    prompt_id = queued.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI did not return prompt_id: {queued}")
    print(f"prompt_id={prompt_id}", flush=True)
    print(f"FastH3 VSA: model={MODEL}, VSA=ON, topk_ratio=0.10, min_tokens=4096, sampler=euler, scheduler=simple, steps=4, shift=12/3", flush=True)
    started = time.monotonic()
    last = ""
    while True:
        entry = _http_json(base, f"/history/{prompt_id}").get(prompt_id)
        if entry:
            status = entry.get("status", {})
            status_text = status.get("status_str", "running")
            if status_text != last:
                print(f"status={status_text} elapsed={time.monotonic() - started:.1f}s", flush=True)
                last = status_text
            if status.get("completed") or entry.get("outputs"):
                break
            if status_text in {"error", "failed"} or any(
                isinstance(msg, list) and msg and msg[0] == "execution_error"
                for msg in status.get("messages", [])
            ):
                raise RuntimeError(json.dumps(entry, ensure_ascii=False, indent=2))
        if time.monotonic() - started > args.timeout:
            raise TimeoutError(f"generation exceeded timeout ({args.timeout}s), prompt_id={prompt_id}")
        time.sleep(args.poll_seconds)
    media = _media_paths(entry, Path(args.output_root))
    result = {
        "prompt_id": prompt_id,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "settings": {"width": args.width, "height": args.height, "length": args.length, "seed": args.seed, "steps": 4, "sampler": "euler", "scheduler": "simple", "topk_ratio": 0.10, "shift_video": 12.0, "shift_audio": 3.0},
        "model": MODEL,
        "history": entry,
        "media_paths": [str(path) for path in media if path.exists()],
    }
    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"elapsed_seconds": result["elapsed_seconds"], "media_paths": result["media_paths"]}, ensure_ascii=False), flush=True)
    if not result["media_paths"]:
        raise RuntimeError("ComfyUI completed but no saved video was reported")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--input-image", default=DEFAULT_INPUT)
    p.add_argument("--width", type=int, default=608)
    p.add_argument("--height", type=int, default=352)
    p.add_argument("--length", type=int, default=124)
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("--poll-seconds", type=float, default=2.0)
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--result-path", default=str(DEFAULT_RESULT_PATH))
    p.add_argument("--prompt", default=(
        "A premium 10-second car commercial: a sleek electric sports car accelerates smoothly "
        "through a rain-slick neon city at dusk. Preserve the supplied first-frame car identity, "
        "body proportions, paint color and lighting direction. Low tracking camera, realistic "
        "live-action motion, crisp reflections, subtle wheel rotation and tire spray, restrained "
        "cinematic grade. Audio: quiet motor hum, tire spray, city ambience and refined music. "
        "No text, subtitles, logos or watermark."
    ))
    return p


if __name__ == "__main__":
    try:
        raise SystemExit(run(parser().parse_args()))
    except (RuntimeError, TimeoutError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
