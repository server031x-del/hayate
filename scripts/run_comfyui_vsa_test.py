"""Run a small MiniMax H3 VSA generation through the local ComfyUI API.

The graph deliberately uses the Kijai VSA single-file checkpoint directly and
the existing HAYATE model registry.  It does not copy weights into ComfyUI.
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
DEFAULT_RESULT_PATH = Path(r"M:\Project\HAYATE\data\comfyui-user\vsa_test_result.json")
VSA_MODEL = "minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors"
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
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI API {exc.code} {path}: {detail}") from exc
    return json.loads(raw.decode("utf-8")) if raw else {}


def build_prompt(*, width: int, height: int, length: int, seed: int, prompt_text: str) -> dict[str, Any]:
    """Create the API-format graph without relying on a UI subgraph export."""
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": VSA_MODEL, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VIDEO_VAE},
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": AUDIO_VAE},
        },
        "5": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["2", 0],
                "vae": ["3", 0],
                "prompt": prompt_text,
                "width": width,
                "height": height,
                "length": length,
            },
        },
        "6": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        },
        "7": {
            "class_type": "BasicScheduler",
            "inputs": {"model": ["1", 0], "scheduler": "simple", "steps": 4, "denoise": 1.0},
        },
        "8": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "res_multistep"},
        },
        "9": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["1", 0], "conditioning": ["5", 0]},
        },
        "10": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["6", 0],
                "guider": ["9", 0],
                "sampler": ["8", 0],
                "sigmas": ["7", 0],
                "latent_image": ["5", 1],
            },
        },
        "11": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["10", 0], "vae": ["3", 0]},
        },
        "12": {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["10", 0], "vae": ["4", 0]},
        },
        "13": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["11", 0],
                "audio": ["12", 0],
                "fps": 24,
                "bit_depth": 8,
                "color_space": "sRGB",
            },
        },
        "14": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["13", 0],
                "filename_prefix": "video/hayate_vsa_test",
                "format": "mp4",
                "codec": "h264",
            },
        },
    }


def _media_paths(history_entry: dict[str, Any], output_root: Path) -> list[Path]:
    paths: list[Path] = []
    for node_output in history_entry.get("outputs", {}).values():
        for key in ("videos", "gifs", "images"):
            for media in node_output.get(key, []):
                filename = media.get("filename")
                if not filename:
                    continue
                subfolder = media.get("subfolder", "")
                # ComfyUI uses type=output for files under its output directory.
                # Reject traversal before turning the API response into a path.
                candidate = (output_root / subfolder / filename).resolve()
                if output_root.resolve() not in candidate.parents:
                    continue
                paths.append(candidate)
    return paths


def run(args: argparse.Namespace) -> int:
    base_url = f"http://{args.host}:{args.port}"
    prompt_text = args.prompt
    graph = build_prompt(
        width=args.width,
        height=args.height,
        length=args.length,
        seed=args.seed,
        prompt_text=prompt_text,
    )
    client_id = str(uuid.uuid4())
    queued = _http_json(base_url, "/prompt", {"prompt": graph, "client_id": client_id})
    prompt_id = queued.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI did not return a prompt_id: {json.dumps(queued, ensure_ascii=False)}")
    if queued.get("node_errors"):
        raise RuntimeError(f"ComfyUI rejected nodes: {json.dumps(queued['node_errors'], ensure_ascii=False)}")

    print(f"prompt_id={prompt_id}", flush=True)
    print(f"settings={args.width}x{args.height}, {args.length} frames, 4 steps, seed={args.seed}", flush=True)
    started = time.monotonic()
    last_status = ""
    while True:
        history = _http_json(base_url, f"/history/{prompt_id}")
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            status_text = str(status.get("status_str", ""))
            if status_text != last_status:
                print(f"status={status_text or 'running'} elapsed={time.monotonic() - started:.1f}s", flush=True)
                last_status = status_text
            if status.get("completed") or entry.get("outputs"):
                break
            if status.get("status_str") in {"error", "failed"} or status.get("messages", []) and any(
                message[0] == "execution_error" for message in status["messages"] if isinstance(message, list)
            ):
                raise RuntimeError(json.dumps(entry, ensure_ascii=False, indent=2))
        if time.monotonic() - started > args.timeout:
            raise TimeoutError(f"generation exceeded timeout ({args.timeout}s), prompt_id={prompt_id}")
        time.sleep(args.poll_seconds)

    output_root = Path(args.output_root).resolve()
    media = _media_paths(entry, output_root)
    result = {
        "prompt_id": prompt_id,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "settings": {"width": args.width, "height": args.height, "length": args.length, "seed": args.seed},
        "model": VSA_MODEL,
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
    p.add_argument("--width", type=int, default=608)
    p.add_argument("--height", type=int, default=352)
    p.add_argument("--length", type=int, default=124, help="frame count; 124 is about 5 seconds at 24 fps")
    p.add_argument("--seed", type=int, default=20260831)
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("--poll-seconds", type=float, default=2.0)
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--result-path", default=str(DEFAULT_RESULT_PATH))
    p.add_argument(
        "--prompt",
        default=(
            "A polished cinematic car commercial: a sleek electric sports car drives through a "
            "rain-slick neon city at dusk, premium realistic live-action photography, smooth "
            "tracking shots, reflections on the bodywork, natural motion and dramatic but "
            "restrained lighting. Audio: subtle motor hum, tire spray, city ambience and a "
            "refined cinematic music bed. No text, subtitles, logos or watermark."
        ),
    )
    return p


if __name__ == "__main__":
    try:
        raise SystemExit(run(parser().parse_args()))
    except (RuntimeError, TimeoutError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
