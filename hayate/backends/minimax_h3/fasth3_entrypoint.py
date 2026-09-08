"""Small HAYATE launcher for the optional FastVideo FastH3 API.

The module has no hard import of FastVideo at package import time.  This keeps
the normal mayble H3/WebUI path lightweight and lets preflight report a clear
missing-runtime error instead of failing during WebUI startup.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path

from hayate.backends.minimax_h3.events import emit_event
from hayate.runtime.gpu_lease import GPULease


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HAYATE FastH3/VSA launcher")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--vsa-sparsity", type=float, default=0.9)
    parser.add_argument("--vsa-tile-size", type=int, choices=(64, 256), default=64)
    parser.add_argument("--vsa-kernel", choices=("triton", "sm100a"), default="triton")
    parser.add_argument("--profile", choices=("all", "strict"), default="strict")
    parser.add_argument("--fa4", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--replicated-dit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-cpu-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-vae", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parallel-vae", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--low-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable the supported single-GPU DiT layerwise offload profile",
    )
    parser.add_argument(
        "--dit-offload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override DiT CPU offload (only effective with FSDP)",
    )
    parser.add_argument(
        "--dit-layerwise-offload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override single-GPU layerwise DiT offload",
    )
    parser.add_argument(
        "--text-encoder-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--vae-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--inference-torch-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def _configure_environment(args: argparse.Namespace) -> None:
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN_H3"
    os.environ["FASTVIDEO_VSA_SM100A"] = "1" if args.vsa_kernel == "sm100a" else "0"
    os.environ["FASTVIDEO_VSA_CUTEDSL"] = "0"
    os.environ["FASTVIDEO_FA4"] = "1" if args.fa4 else "0"
    os.environ["FASTVIDEO_NVFP4_FA4"] = "0"
    os.environ["FASTVIDEO_MINIMAX_H3_FA4_PACKED_VARLEN"] = "0"
    # The student is trained on four jump points.  Five sampler points include
    # the terminal zero and therefore execute exactly four DiT forwards.
    os.environ["FASTVIDEO_DMD_DENOISING_STEPS"] = "999,749,500,250"
    os.environ["FASTVIDEO_MINIMAX_H3_FUSIONS"] = "all" if args.profile == "all" else "0"
    os.environ["FASTVIDEO_INFERENCE_TORCH_COMPILE"] = "1" if args.inference_torch_compile else "0"
    os.environ["FASTVIDEO_VAE_PARALLEL_DECODE"] = "1" if args.parallel_vae else "0"
    os.environ["FASTVIDEO_VAE_PARALLEL_ENCODE"] = "0"
    os.environ["FASTVIDEO_VAE_PARALLEL_DECODE_STRATEGY"] = "gather"
    os.environ["FASTVIDEO_ULYSSES_A2A"] = "off"
    os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"


def _api_types():
    api = importlib.import_module("fastvideo.api")
    return api


def _build_config(args: argparse.Namespace):
    api = _api_types()
    experimental = {
        "attention_backend": "VIDEO_SPARSE_ATTN_H3",
        "inference_torch_compile": args.inference_torch_compile,
        "vae_parallel_decode": args.parallel_vae,
        "vae_parallel_decode_strategy": "gather",
        "VSA_sparsity": args.vsa_sparsity,
        "VSA_tile_size": args.vsa_tile_size,
    }
    if args.low_memory:
        dit_offload = False if args.dit_offload is None else args.dit_offload
        dit_layerwise_offload = (
            True if args.dit_layerwise_offload is None else args.dit_layerwise_offload
        )
    else:
        dit_offload = False if args.dit_offload is None else args.dit_offload
        dit_layerwise_offload = False if args.dit_layerwise_offload is None else args.dit_layerwise_offload
    return api.GeneratorConfig(
        model_path=str(args.model_path),
        pipeline=api.PipelineSelection(
            components=api.ComponentConfig(),
            experimental=experimental,
        ),
        engine=api.EngineConfig(
            num_gpus=args.num_gpus,
            use_fsdp_inference=args.num_gpus > 1 and not args.replicated_dit,
            parallelism=api.ParallelismConfig(tp_size=1, sp_size=args.num_gpus),
            offload=api.OffloadConfig(
                dit=dit_offload,
                dit_layerwise=dit_layerwise_offload,
                text_encoder=args.text_encoder_offload,
                vae=args.vae_offload,
                pin_cpu_memory=args.pin_cpu_memory,
            ),
            compile=api.CompileConfig(
                enabled=False,
                mode=None,
                vae_enabled=args.compile_vae,
            ),
        ),
    )


def _build_request(args: argparse.Namespace):
    api = _api_types()
    return api.GenerationRequest(
        prompt=args.prompt,
        negative_prompt="",
        sampling=api.SamplingConfig(
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            fps=24,
            num_inference_steps=args.steps,
            guidance_scale=1.0,
            batch_cfg=False,
            seed=args.seed,
        ),
        output=api.OutputConfig(
            output_path=str(args.output),
            save_video=True,
            return_frames=False,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps != 5:
        raise SystemExit("FastH3 uses 5 sigma points (4 DiT forwards)")
    if not args.model_path.is_dir():
        raise SystemExit(f"FastVideo model directory does not exist: {args.model_path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _configure_environment(args)
    runtime_lease: GPULease | None = None
    if os.environ.get("HAYATE_GPU_RUNTIME_LEASE") == "1":
        gpu_id = os.environ.get("HAYATE_GPU_UUID") or None
        if gpu_id is None and os.environ.get("HAYATE_GPU_INDEX"):
            gpu_id = f"index:{os.environ['HAYATE_GPU_INDEX']}"
        runtime_lease = GPULease(
            gpu_id=gpu_id,
            namespace="runtime",
            owner={
                "pid": os.getpid(),
                "kind": "fasth3-generation-runtime",
                "gpu_uuid": os.environ.get("HAYATE_GPU_UUID") or None,
                "gpu_index": os.environ.get("HAYATE_GPU_INDEX"),
            },
        )
        if not runtime_lease.acquire():
            owner = runtime_lease.busy_owner() or {}
            raise SystemExit(
                f"selected GPU runtime lease is busy (pid={owner.get('pid', '?')})"
            )
    emit_event(
        "process",
        phase="start",
        progress=1.0,
        stage="起動準備",
        detail="FastH3/VSAランタイムを起動しています",
    )
    generator = None
    try:
        from fastvideo import VideoGenerator
    except Exception as exc:
        emit_event(
            "error",
            phase="error",
            progress=0.0,
            stage="起動失敗",
            detail="FastVideoがインストールされていません",
        )
        if runtime_lease is not None:
            runtime_lease.release()
        raise SystemExit(
            "FastVideo could not be imported in the configured Python; install a compatible runtime"
        ) from exc
    try:
        emit_event(
            "progress",
            phase="load",
            progress=12.0,
            stage="FastVideo読込",
            detail="VSA対応モデルを準備しています",
        )
        # Keep construction inside the lease-protected try/finally.  FastVideo
        # can fail while loading a model (missing kernel/config/OOM); the
        # physical GPU must be released in those cases as well.
        generator = VideoGenerator.from_config(_build_config(args))
        started = time.perf_counter()
        emit_event(
            "progress",
            phase="denoise",
            progress=36.0,
            stage="動画生成",
            detail="FastH3 4-forward denoiseを開始します",
        )
        result = generator.generate(_build_request(args))
        actual = getattr(result, "video_path", None)
        emit_event(
            "progress",
            phase="save",
            progress=98.0,
            stage="書き出し",
            detail=f"映像と音声を保存しました: {actual or args.output}",
        )
        emit_event(
            "metrics",
            runtime_metrics={
                "backend": "fastvideo_vsa",
                "generation_seconds": time.perf_counter() - started,
                "video_path": str(actual or args.output),
                "steps": args.steps,
                "vsa_sparsity": args.vsa_sparsity,
                "vsa_tile_size": args.vsa_tile_size,
            },
        )
        print(json.dumps({"output": str(actual or args.output)}, ensure_ascii=False), flush=True)
    finally:
        try:
            if generator is not None:
                shutdown = getattr(generator, "shutdown", None)
                if callable(shutdown):
                    shutdown()
        finally:
            if runtime_lease is not None:
                runtime_lease.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
