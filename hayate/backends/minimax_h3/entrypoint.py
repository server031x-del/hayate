from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from hayate.backends.minimax_h3.easycache import EasyCacheConfig, install_easycache_override
from hayate.backends.minimax_h3.upstream import H3UpstreamAdapter
from hayate.backends.minimax_h3.nvfp4_conditioner import install_nvfp4_conditioner_override
from hayate.backends.minimax_h3.prompt_cache import install_prompt_cache_override
from hayate.backends.minimax_h3.w4a8_upstream import install_w4a8_override


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("upstream_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
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
    install_easycache_override(module, EasyCacheConfig.from_environment())
    install_prompt_cache_override(module, upstream_commit=validation.commit or validation.audited_commit)
    install_w4a8_override(module)
    install_nvfp4_conditioner_override()
    forwarded = args.upstream_args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    sys.argv = [str(script), *forwarded]
    import psutil
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        module.main()
    finally:
        metrics = {}
        memory = psutil.Process().memory_info()
        metrics["process_rss_bytes"] = int(memory.rss)
        metrics["process_peak_rss_bytes"] = int(getattr(memory, "peak_wset", memory.rss))
        metrics["process_private_bytes"] = int(getattr(memory, "private", memory.vms))
        metrics["process_peak_private_bytes"] = int(
            getattr(memory, "peak_pagefile", getattr(memory, "private", memory.vms))
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            metrics["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            metrics["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        controller = getattr(module, "_hayate_easycache_controller", None)
        if controller is not None:
            metrics["easycache"] = controller.stats()
        print("HAYATE_RUNTIME_METRICS " + json.dumps(metrics, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
