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
}


def get_generation_profile(name: str) -> GenerationProfile:
    try:
        return GENERATION_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown generation profile: {name}") from exc
