"""Warm, per-process cache for the native MiniMax-H3 transformer."""
from __future__ import annotations

import gc
import importlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _model_key(args: Any, task: str, device: Any) -> str | None:
    """Return a conservative cache key for configurations safe to reuse."""
    if (
        int(getattr(args, "blocks_to_swap", 0) or 0) != 0
        or bool(getattr(args, "progressive_load", False))
        or bool(getattr(args, "lora_weight", None))
        or getattr(args, "offload_engine", "blockswap") != "blockswap"
        or bool(getattr(args, "compile", False))
        or getattr(args, "attn_mode", "sdpa") == "sol"
    ):
        return None
    fields = {
        name: getattr(args, name, None)
        for name in (
            "ckpt_dir",
            "dit",
            "dit_dtype",
            "fp8",
            "fp8_scaled",
            "fp8_fast",
            "fp8_exclude_adaln",
            "int8_fast",
            "classic_block_swap",
            "attn_mode",
            "act_chunk_rows",
        )
    }
    fields["task"] = task
    fields["device"] = str(device)
    return json.dumps(fields, sort_keys=True, default=str, separators=(",", ":"))


def install_native_transformer_cache(module: Any) -> None:
    """Retain the A100 DiT across jobs while respecting H3's staged VRAM use.

    The transformer is moved to CPU before conditioner/VAE stages and brought
    back to the GPU between jobs. This avoids keeping the H3 DiT beside the
    streamed 32B conditioner during inference, while avoiding a checkpoint
    reload for the next job.
    """
    if getattr(module, "_hayate_native_cache_installed", False):
        return

    original_load = module.load_transformer_stage
    original_run_one = module.run_one
    original_clean = module.clean_memory_on_device
    cache: dict[str, Any] = {"key": None, "transformer": None, "device": None}

    def move_to(target: Any) -> None:
        transformer = cache["transformer"]
        if transformer is None:
            return
        target = module.torch.device(target)
        current = cache["device"]
        if current is not None and module.torch.device(current) == target:
            return
        transformer.to(target)
        if target.type == "cuda":
            module.torch.cuda.synchronize(target)
        cache["device"] = target

    def clear(reason: str) -> None:
        transformer = cache["transformer"]
        cache["transformer"] = None
        cache["key"] = None
        cache["device"] = None
        if transformer is not None:
            try:
                transformer.to("cpu")
            except Exception as exc:  # noqa: BLE001 - discard must survive tensor conversion errors
                # Dropping the only cache reference is still safer than keeping
                # a partially moved transformer alive after a failed transfer.
                logger.warning("Could not move cached transformer to CPU before eviction: %s", exc)
            del transformer
            gc.collect()
            if module.torch.cuda.is_available():
                module.torch.cuda.empty_cache()
        print(f"HAYATE_DIT_CACHE evicted reason={reason}", flush=True)

    def cached_load(args, task, device):
        key = _model_key(args, task, device)
        if key is None:
            clear("unsupported-model-settings")
            return original_load(args, task, device)
        if cache["transformer"] is not None and cache["key"] == key:
            minimax_attention = importlib.import_module("minimax_video.attention")
            minimax_transformer = importlib.import_module("minimax_video.transformer")
            minimax_attention.set_attention_backend(args.attn_mode)
            minimax_transformer.set_act_chunk_rows(args.act_chunk_rows)
            move_to(device)
            print("HAYATE_DIT_CACHE reused=1", flush=True)
            return cache["transformer"], None

        if cache["transformer"] is not None:
            clear("model-settings-changed")
        result = original_load(args, task, device)
        transformer, loader = result if isinstance(result, tuple) else (result, None)
        # Progressive loading and offload engines have live hooks/worker threads
        # whose lifecycle belongs to the current generation only.
        if loader is None:
            cache["key"] = key
            cache["transformer"] = transformer
            cache["device"] = module.torch.device(device)
            print("HAYATE_DIT_CACHE loaded=1", flush=True)
        return result

    def staged_clean(device) -> None:
        transformer = cache["transformer"]
        if transformer is not None and module.torch.device(device).type == "cuda":
            controller = getattr(module, "_hayate_easycache_controller", None)
            clear_cache = getattr(controller, "_clear_runtime_state", None)
            if clear_cache is not None:
                clear_cache()
            try:
                move_to("cpu")
                print("HAYATE_DIT_CACHE staged=cpu", flush=True)
            except Exception as exc:  # noqa: BLE001 - alternate quantized tensor types raise different errors
                clear(f"cpu-offload-failed:{type(exc).__name__}")
        original_clean(device)

    def cached_run_one(args, task, device, *positional, **keyword):
        # The next request starts with conditioner and VAE work. Keep the GPU
        # free for those stages, then let load_transformer_stage restore the
        # cached DiT just before denoising.
        if cache["transformer"] is not None:
            try:
                move_to("cpu")
            except Exception as exc:  # noqa: BLE001 - fall back to a cold model after conversion errors
                clear(f"pre-job-offload-failed:{type(exc).__name__}")
        try:
            result = original_run_one(args, task, device, *positional, **keyword)
        except BaseException:
            # A failed or cancelled denoise may have left hooks, CUDA work, or
            # partially updated parameters behind. The next job starts clean.
            clear("generation-failed")
            raise
        if cache["transformer"] is not None:
            try:
                move_to(device)
                print("HAYATE_DIT_CACHE resident=1", flush=True)
            except Exception as exc:  # noqa: BLE001 - residency is best-effort; the job already succeeded
                clear(f"idle-residency-failed:{type(exc).__name__}")
        return result

    module.load_transformer_stage = cached_load
    module.clean_memory_on_device = staged_clean
    module.run_one = cached_run_one
    module._hayate_native_transformer_cache = cache
    module._hayate_native_cache_installed = True
