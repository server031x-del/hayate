from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from hayate.backends.minimax_h3.easycache import EasyCacheConfig, install_easycache_override
from hayate.backends.minimax_h3.events import emit_event, install_structured_events
from hayate.backends.minimax_h3.upstream import H3UpstreamAdapter
from hayate.backends.minimax_h3.nvfp4_conditioner import install_nvfp4_conditioner_override
from hayate.backends.minimax_h3.prompt_cache import install_prompt_cache_override
from hayate.backends.minimax_h3.pdd import install_pdd_override
from hayate.backends.minimax_h3.vae_tiling import (
    install_vae_attention_override,
    install_vae_tiling_override,
)
from hayate.backends.minimax_h3.w4a8_upstream import install_w4a8_override
from hayate.runtime.gpu_lease import GPULease


def _collect_runtime_metrics(module, psutil, torch) -> dict:
    metrics = {}
    try:
        memory = psutil.Process().memory_info()
        metrics["process_rss_bytes"] = int(memory.rss)
        metrics["process_peak_rss_bytes"] = int(getattr(memory, "peak_wset", memory.rss))
        metrics["process_private_bytes"] = int(getattr(memory, "private", memory.vms))
        metrics["process_peak_private_bytes"] = int(
            getattr(memory, "peak_pagefile", getattr(memory, "private", memory.vms))
        )
    except Exception as exc:
        metrics["process_metrics_error"] = f"{type(exc).__name__}: {exc}"
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            metrics["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            metrics["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        except Exception as exc:
            # Metrics are diagnostic. Preserve the generation exception when
            # CUDA is already in a failed state instead of masking it here.
            metrics["cuda_metrics_error"] = f"{type(exc).__name__}: {exc}"
    controller = getattr(module, "_hayate_easycache_controller", None)
    if controller is not None:
        metrics["easycache"] = controller.stats()
    pdd_controller = getattr(module, "_hayate_pdd_controller", None)
    if pdd_controller is not None:
        metrics["pdd"] = pdd_controller.stats()
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("upstream_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    # Acquire the child-owned runtime lease before importing upstream/H3 or
    # PyTorch.  This closes the startup window in which an orphaned child
    # could otherwise hold CUDA while a second process imports the engine.
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
                "kind": "generation-runtime",
                "gpu_uuid": os.environ.get("HAYATE_GPU_UUID") or None,
                "gpu_index": os.environ.get("HAYATE_GPU_INDEX"),
            },
        )
        if not runtime_lease.acquire():
            owner = runtime_lease.busy_owner() or {}
            raise RuntimeError(
                f"選択したGPUの実行ロックを取得できません (pid={owner.get('pid', '?')})"
            )

    adapter = H3UpstreamAdapter(args.upstream)
    validation = adapter.require_valid(require_audited_commit=True)
    engine_dir = adapter.checkout / "minimax_engine"
    for path in (adapter.checkout, engine_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    script = adapter.component_path("generation_cli")
    spec = importlib.util.spec_from_file_location("hayate_external_minimax_generate_video", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import upstream generation CLI: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    install_structured_events(module)
    install_easycache_override(module, EasyCacheConfig.from_environment())
    install_prompt_cache_override(module, upstream_commit=validation.commit or validation.audited_commit)
    install_w4a8_override(module)
    install_pdd_override(module)
    install_nvfp4_conditioner_override()
    install_vae_tiling_override()
    install_vae_attention_override("sdpa")
    forwarded = args.upstream_args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    sys.argv = [str(script), *forwarded]
    import psutil
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    emit_event("process", phase="start", progress=1.0, stage="起動準備", detail="MiniMax H3エンジンを起動しました")
    try:
        module.main()
    finally:
        metrics = _collect_runtime_metrics(module, psutil, torch)
        emit_event("metrics", runtime_metrics=metrics)
        print("HAYATE_RUNTIME_METRICS " + json.dumps(metrics, sort_keys=True), flush=True)
        if runtime_lease is not None:
            runtime_lease.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
