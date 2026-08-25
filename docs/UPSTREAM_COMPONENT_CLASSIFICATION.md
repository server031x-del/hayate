# maybleMyers/h3 component classification

This classification was completed before HAYATE implementation. It is based on
`maybleMyers/h3` commit `94220c1fdf14d6d9d40be06fb99f55b27c0d9024`
(2026-08-22) and focuses on `minimax_engine/`.

HAYATE does not replace that engine. The intended dependency direction is:

```text
maybleMyers/h3 MiniMax-H3 engine
  -> HAYATE adapter and extensions
  -> consumer GPU policy, loading, profiling, and benchmarks
```

## Classification

| Upstream component | Upstream location | Class | HAYATE decision |
|---|---|---:|---|
| MiniMax pipeline and packing contracts | `minimax_video/pipeline.py`, `packing.py`, `packing_ref2va.py` | **REUSE** | Preserve setup, packing, denoise, decode contracts. Do not recreate the generation math. Use through an adapter once redistribution/import licensing is resolved. |
| Scheduler | `minimax_video/scheduler.py` | **REUSE** | Use the upstream `MiniMaxH3Scheduler` and its video/audio instances unchanged. File carries Apache-2.0 notice. |
| H3 transformer definition | `minimax_video/transformer.py` | **REUSE** | Preserve model structure, tensor names, forward contract, and AdaLN curve behavior. File carries Apache-2.0 notice. |
| Video VAE | `minimax_video/vae_video.py` | **REUSE** | Preserve encode/decode, normalization, tiling, and float32-weight/fp16-autocast recipe. File carries Apache-2.0 notice. |
| Audio VAE | `minimax_video/vae_audio.py` | **REUSE** | Preserve the released audio latent and 32 kHz decode contract. File carries Apache-2.0 notice. |
| Attention dispatch | `minimax_video/attention.py`, `sol_attn/` | **MODIFY / EXTEND** | Keep upstream dispatch semantics. Add measured consumer-GPU choices only behind capability checks and benchmark gates. Preserve Sol-Attn notices if redistributed. |
| Block swap | `transformer.py`, `modules/custom_offloading_utils.py` | **MODIFY / EXTEND** | Keep block-loop hooks and ordering. Add HAYATE budgets, telemetry, RTX 3060 policy, and later NVMe tier without changing transformer math. |
| Progressive block loading | `minimax_video/progressive_load.py` | **MODIFY / EXTEND** | Reuse gate/ordering design. Add explicit stage ownership, RAM budgets, cancellation, and benchmark events. |
| CPU/GPU offload | `custom_offloading_utils.py`, `diffusers-mm` integration | **MODIFY / EXTEND** | Wrap behind HAYATE memory policy. Do not assume two GPUs form one memory pool. v0.1 selects RTX 3060 only for execution. |
| Existing checkpoint/model loader | `minimax_video/model_loader.py` | **REPLACE / EXTEND** | Retain upstream component construction and single-file conversion contracts, but route files through HAYATE inspection/validation and add direct W4A8, NVFP4/AWQ, and ConvRot loaders. |
| Existing INT8 infrastructure | `minimax_video/int8_quant.py` | **MODIFY / EXTEND** | Reuse documented ConvRot marker/key conventions and fast INT8 path where applicable. The audited Video VAE export quantizes 144 2D Linear weights; its convolutions remain FP32. |
| Qwen3-VL conditioner | `conditioner.py`, `qwen3vl_text.py`, `qwen3vl_vision.py` | **MODIFY / EXTEND** | Preserve layer-50 conditioning and media-token rules. Extend the loader for NVFP4/AWQ and consumer-RAM streaming. |
| Text encoder streaming | `conditioner.py` | **MODIFY / EXTEND** | Preserve ordered one-layer streaming/double-buffer concept. Add budget selection, metrics, and safe fallback. |
| Prompt cache | `minimax_generate_video.py` | **MODIFY / EXTEND** | Preserve cache-key inputs and CPU-resident embeddings. HAYATE now adds schema/version/upstream/model/input fingerprints and atomic writes through a runtime override. |
| Single-file overrides | `model_loader.py`, `conditioner.py`, `int8_quant.py` | **MODIFY / EXTEND** | Make single-file models first-class registry entries and validate headers before model allocation. |
| Generation CLI | `minimax_generate_video.py` | **REUSE / WRAP** | HAYATE launches the pinned external entry point, installs only quantized-loader overrides, and forwards generation arguments. Pipeline/scheduler/packing/denoise/decode remain upstream-owned. |
| Job queue and worker | `wan_job_queue.py`, `wan_worker.py` | **REUSE / WRAP** | Reuse persistent sequential-job semantics when generation is enabled. No GUI or worker is required in v0.1. |
| Hardware profiler | none | **ADD** | CPU, RAM, CUDA driver, GPU, compute capability, PyTorch, and runtime detection without importing models. |
| Safetensors inspector | limited private header reader in `int8_quant.py` | **ADD** | Bounded, metadata-only parser with corruption checks, tensor inventory, physical byte estimates, and evidence-based quantization detection. |
| Model registry | none | **ADD** | YAML paths, relative/env expansion, and direct references to existing ComfyUI model folders without copying files or importing ComfyUI. |
| Stage runtime | staged functions in generation CLI | **ADD** | General stage lifecycle and resource-release API, while preserving the upstream A/B/C/D ordering. |
| Consumer GPU memory manager | no standalone public API | **ADD** | RAM/VRAM snapshots, peaks, budgets, and future GPU/RAM/NVMe tiers. |
| Benchmark framework | ad-hoc timers/logs | **ADD** | Structured stage timings, memory peaks, JSON output, and before/after comparison fields. |
| RTX 3060 / 32 GB policy | none | **ADD** | Main-GPU selection, conservative compatibility checks, and stage residency policy. GTX 1660 SUPER is detected but not scheduled in v0.1. |
| Future NVMe streaming | none | **ADD (interface only)** | Reserve a storage-tier interface; no NVMe streaming implementation in v0.1. |
| Gradio UI, interpolation, upscale, SeedVR | `h3.py`, `GIMM-VFI/`, `modules/SeedVR/` | **REPLACE / EXCLUDE** | Outside HAYATE v0.1 and not a runtime dependency. GIMM-VFI is non-commercial-only unless separately licensed. |

## Upstream contracts HAYATE must preserve

- Components are loaded in stages so the Qwen3-VL conditioner, 33B transformer,
  and VAEs do not coexist on the GPU.
- MiniMax-H3 reads the unnormalized Qwen3-VL hidden state after decoder layer 50.
- The FL2VA and Ref2VA transformer partitions remain distinct.
- Scheduler instances for video and audio remain distinct.
- Packed row order, timestep pinning, random draw order, keyframe seed/rounding,
  and video normalization are generation-quality contracts, not optimization targets.
- Block swapping/offload may change residency and transfer scheduling, not tensor
  meanings or block execution order.
- Every optimization must record before/after time, VRAM, RAM, and quality impact.

## Current boundary

HAYATE implements inspection, validation, routing, profiling, stage ownership,
memory snapshots, kernel checks, and a pinned external generation wrapper. It
deliberately does not copy or recreate the generation pipeline. Direct W4A8,
NVFP4/AWQ, and INT8 ConvRot bindings reuse the upstream model/key contracts and
`comfy-kitchen` tensor layouts. All four complete files now strict-load, and a
fixed-seed 50-step video/audio generation is verified on RTX 3060 / 32 GB RAM.
Longer clips, larger resolutions, LoRA-on-packed weights, and cross-backend
quality-equivalence measurements remain benchmark-gated.
