from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import psutil

from hayate.backends.minimax_h3.nvfp4_conditioner import install_nvfp4_conditioner_override
from hayate.backends.minimax_h3.upstream import H3UpstreamAdapter
from hayate.backends.minimax_h3.vae_tiling import validate_vae_tile_size
from hayate.backends.minimax_h3.w4a8_upstream import install_w4a8_override
from hayate.models import ModelRegistry
from hayate.models.types import ModelRole


def _install_upstream(checkout: Path):
    adapter = H3UpstreamAdapter(checkout)
    adapter.require_valid(require_audited_commit=True)
    for path in (adapter.checkout, adapter.checkout / "minimax_engine"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    install_w4a8_override(None)
    install_nvfp4_conditioner_override()
    return adapter


def _tensor_summary(model) -> dict:
    from comfy_kitchen.tensor import QuantizedTensor

    tensors = list(model.parameters()) + list(model.buffers())
    return {
        "parameter_and_buffer_count": len(tensors),
        "quantized_tensor_count": sum(isinstance(tensor, QuantizedTensor) for tensor in tensors),
        "meta_tensor_count": sum(tensor.is_meta for tensor in tensors),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--component",
        choices=("transformer", "text_encoder", "video_vae", "audio_vae"),
        required=True,
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--decode-smoke", action="store_true")
    parser.add_argument("--decode-latent-frames", type=int, default=2)
    parser.add_argument("--decode-latent-height", type=int, default=8)
    parser.add_argument("--decode-latent-width", type=int, default=8)
    parser.add_argument("--vae-tile-size", type=int, default=256)
    parser.add_argument("--vae-no-tiling", action="store_true")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    args = parser.parse_args(argv)
    _install_upstream(args.upstream)
    registry = ModelRegistry.load(args.config)
    process = psutil.Process()
    started = time.perf_counter()

    import torch

    if args.component == "transformer":
        from minimax_video.model_loader import load_transformer

        model = load_transformer(
            str(args.ckpt_dir),
            device=torch.device("cpu"),
            task="fl2va",
            dit_dtype=torch.bfloat16,
            dit_path=str(registry.get("minimax_h3", ModelRole.TRANSFORMER).path),
        )
        summary = _tensor_summary(model)
    elif args.component == "text_encoder":
        from minimax_video.conditioner import MiniMaxH3Conditioner

        model = MiniMaxH3Conditioner(
            str(args.ckpt_dir),
            device="cpu",
            dtype=torch.bfloat16,
            gpu_layers=0,
            stream_device=None,
            text_encoder_path=str(registry.get("minimax_h3", ModelRole.TEXT_ENCODER).path),
        )
        summary = {
            "text_model": _tensor_summary(model.text_model),
            "vision_tower": _tensor_summary(model.vision_tower),
        }
    elif args.component == "video_vae":
        from minimax_video.model_loader import load_vae

        model = load_vae(
            str(args.ckpt_dir),
            device=torch.device("cpu"),
            vae_dtype=torch.float32,
            vae_path=str(registry.get("minimax_h3", ModelRole.VIDEO_VAE).path),
        )
        summary = _tensor_summary(model)
        if args.decode_smoke:
            if min(
                args.decode_latent_frames,
                args.decode_latent_height,
                args.decode_latent_width,
            ) < 1:
                raise ValueError("decode latent dimensions must be positive")
            validate_vae_tile_size(args.vae_tile_size)
            torch.manual_seed(20260825)
            torch.cuda.reset_peak_memory_stats()
            torch.backends.cudnn.benchmark = args.cudnn_benchmark
            model = model.to("cuda:0")
            if args.vae_no_tiling:
                model.disable_tiling()
            else:
                model.enable_tiling(
                    tile_sample_min_height=args.vae_tile_size,
                    tile_sample_min_width=args.vae_tile_size,
                )
            latent = torch.randn(
                (
                    1,
                    24,
                    args.decode_latent_frames,
                    args.decode_latent_height,
                    args.decode_latent_width,
                ),
                device="cuda:0",
                dtype=torch.float32,
            )
            decode_started = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                decoded = model.decode(latent).sample
            torch.cuda.synchronize()
            summary["decode_smoke"] = {
                "seed": 20260825,
                "tiling": not args.vae_no_tiling,
                "tile_size": None if args.vae_no_tiling else args.vae_tile_size,
                "cudnn_benchmark": args.cudnn_benchmark,
                "input_shape": list(latent.shape),
                "output_shape": list(decoded.shape),
                "duration_seconds": time.perf_counter() - decode_started,
                "finite": bool(torch.isfinite(decoded).all().item()),
                "mean": float(decoded.float().mean().item()),
                "std": float(decoded.float().std().item()),
                "min": float(decoded.float().min().item()),
                "max": float(decoded.float().max().item()),
                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            }
    else:
        from minimax_video.model_loader import load_audio_vae

        model = load_audio_vae(
            str(args.ckpt_dir),
            device=torch.device("cpu"),
            dtype=torch.float32,
            audio_vae_path=str(registry.get("minimax_h3", ModelRole.AUDIO_VAE).path),
        )
        summary = _tensor_summary(model)

    duration = time.perf_counter() - started
    payload = {
        "component": args.component,
        "ok": True,
        "duration_seconds": duration,
        "process_rss_bytes": process.memory_info().rss,
        "summary": summary,
        "torch": torch.__version__,
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    del model
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
