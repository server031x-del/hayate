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

    def to_dict(self) -> dict:
        return asdict(self)


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
}


def get_generation_profile(name: str) -> GenerationProfile:
    try:
        return GENERATION_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown generation profile: {name}") from exc
