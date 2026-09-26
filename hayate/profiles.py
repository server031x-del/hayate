from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GenerationProfile:
    name: str
    label: str
    steps: int
    attention_backend: str
    easycache: bool
    easycache_threshold: float
    easycache_start: float
    easycache_end: float
    easycache_max_consecutive_skips: int
    blocks_to_swap: int = 49
    activation_chunk_rows: int = 32768
    vae_tile_size: int = 256
    approximate: bool = False
    pdd: bool = False
    # Consumer profiles stream the 32B conditioner layer by layer from host
    # RAM.  Large-memory profiles keep it resident (-1 = every decoder layer).
    text_encoder_gpu_layers: int = 0
    text_encoder_stream: bool = True
    # Route INT8 ConvRot DiT Linears through torch._int_mm (INT8 tensor cores).
    # The conditioner is kept on the weight-only dequantized path regardless.
    int8_fast: bool = False
    # Profiles above the consumer-GPU operating point declare the adapter
    # memory they were designed for; the WebUI refuses smaller GPUs.
    min_vram_gib: int = 0
    # Keep the native H3 transformer resident between WebUI jobs.  This is
    # limited to the A100 detail profile because it reserves substantial VRAM.
    keep_model_warm: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# Profiles for a single 40/80 GB datacenter adapter (A100 class, SM80).  DiT
# stays resident with no block swap; the 32B conditioner streams on demand so
# the same profile fits the 40 GB card after the real CUDA/non-PyTorch overhead
# is included.  The stream can be disabled manually on an 80 GB card.
LARGE_GPU_MIN_VRAM_GIB = 38

GENERATION_PROFILES = {
    "quality": GenerationProfile(
        "quality", "Quality SDPA", 50, "sdpa", False, 0.2, 0.15, 0.95, 2
    ),
    "fast": GenerationProfile(
        "fast", "Fast SDPA", 20, "sdpa", True, 0.4, 0.15, 0.95, 2, approximate=True
    ),
    "fast_sage": GenerationProfile(
        "fast_sage",
        "Fast Sage",
        20,
        "sageattn",
        True,
        0.4,
        0.15,
        0.95,
        2,
        approximate=True,
    ),
    "fast_sage_detail": GenerationProfile(
        "fast_sage_detail",
        "Fast Sage Detail",
        20,
        "sageattn",
        True,
        0.4,
        0.15,
        0.85,
        2,
        approximate=True,
    ),
    "pdd_sage": GenerationProfile(
        "pdd_sage",
        "PDD Acc 8-Step + Sage",
        9,
        "sageattn",
        False,
        0.4,
        0.15,
        0.95,
        2,
        approximate=True,
        pdd=True,
    ),
    "pdd": GenerationProfile(
        "pdd",
        "PDD Acc 8-Step",
        9,
        "sdpa",
        False,
        0.4,
        0.15,
        0.95,
        2,
        approximate=True,
        pdd=True,
    ),
    "comfy_fasth3": GenerationProfile(
        "comfy_fasth3", "FastH3 INT8 · ComfyUI", 5, "sdpa", False,
        0.0, 0.0, 1.0, 1, approximate=True,
    ),
    "comfy_fl2va": GenerationProfile(
        "comfy_fl2va", "画像優先 FL2VA · ComfyUI", 50, "sdpa", False,
        0.0, 0.0, 1.0, 1,
    ),
    "fasth3": GenerationProfile(
        "fasth3",
        "FastH3 VSA 4-Step（実験）",
        5,
        "sdpa",
        False,
        0.0,
        0.0,
        1.0,
        1,
        blocks_to_swap=0,
        activation_chunk_rows=0,
        vae_tile_size=256,
        approximate=True,
    ),
    "fasth3_fast": GenerationProfile(
        "fasth3_fast",
        "FastH3 VSA Blackwell最速（実験）",
        5,
        "sdpa",
        False,
        0.0,
        0.0,
        1.0,
        1,
        blocks_to_swap=0,
        activation_chunk_rows=0,
        vae_tile_size=256,
        approximate=True,
    ),
    # Recommended A100 operating point: the validated Fast Sage Detail
    # schedule (20 points, EasyCache 0.4 ending at 85%) with a resident DiT,
    # streamed conditioner, and INT8 tensor-core GEMMs.
    "a100_detail": GenerationProfile(
        "a100_detail",
        "A100 高速・高画質",
        20,
        "sageattn",
        True,
        0.4,
        0.15,
        0.85,
        2,
        blocks_to_swap=0,
        approximate=True,
        text_encoder_gpu_layers=0,
        text_encoder_stream=True,
        int8_fast=True,
        min_vram_gib=LARGE_GPU_MIN_VRAM_GIB,
        keep_model_warm=True,
    ),
    # Fidelity reference for A/B checks on the same card: exact attention, no
    # cache, 50 points, and weight-only INT8 error (dequantized matmul).
    "a100_quality": GenerationProfile(
        "a100_quality",
        "A100 品質基準",
        50,
        "sdpa",
        False,
        0.2,
        0.15,
        0.95,
        2,
        blocks_to_swap=0,
        text_encoder_gpu_layers=0,
        text_encoder_stream=True,
        min_vram_gib=LARGE_GPU_MIN_VRAM_GIB,
    ),
    # PDD keeps SDPA (the PDD+Sage short-clip failure) and the dequantized
    # base matmul, matching the numerics its W4A8 validation relied on.
    "a100_pdd": GenerationProfile(
        "a100_pdd",
        "A100 PDD 8-Step",
        9,
        "sdpa",
        False,
        0.4,
        0.15,
        0.95,
        2,
        blocks_to_swap=0,
        approximate=True,
        pdd=True,
        text_encoder_gpu_layers=0,
        text_encoder_stream=True,
        min_vram_gib=LARGE_GPU_MIN_VRAM_GIB,
    ),
}


def get_generation_profile(name: str) -> GenerationProfile:
    try:
        return GENERATION_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown generation profile: {name}") from exc
