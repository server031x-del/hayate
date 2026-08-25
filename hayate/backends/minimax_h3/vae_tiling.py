from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any


def validate_vae_tile_size(tile_size: int, *, label: str = "vae tile size") -> int:
    # Pinned upstream advances tiles by (size - 64) and compresses spatially 16x.
    # A tile <= 64 cannot advance its split loop; non-multiples of 16 misalign latents.
    if tile_size <= 64 or tile_size % 16:
        raise ValueError(f"{label} must be a multiple of 16 greater than 64")
    return tile_size


def vae_tile_size_from_environment() -> int | None:
    raw = os.environ.get("HAYATE_VAE_TILE_SIZE")
    if raw is None or not raw.strip():
        return None
    try:
        tile_size = int(raw)
    except ValueError as exc:
        raise ValueError("HAYATE_VAE_TILE_SIZE must be an integer") from exc
    return validate_vae_tile_size(tile_size, label="HAYATE_VAE_TILE_SIZE")


def install_vae_tiling_override(tile_size: int | None = None) -> bool:
    """Configure upstream-loaded video VAEs without copying its implementation."""

    if tile_size is None:
        tile_size = vae_tile_size_from_environment()
    if tile_size is None:
        return False

    from minimax_video import model_loader

    current: Callable[..., Any] = model_loader.load_vae
    if getattr(current, "_hayate_vae_tiling_override", False):
        installed_size = getattr(current, "_hayate_vae_tile_size", None)
        if installed_size != tile_size:
            raise RuntimeError(
                f"VAE tiling override is already installed for {installed_size}, not {tile_size}"
            )
        return False

    def load_vae_with_hayate_tiling(*args, **kwargs):
        vae = current(*args, **kwargs)
        vae.enable_tiling(
            tile_sample_min_height=tile_size,
            tile_sample_min_width=tile_size,
        )
        return vae

    load_vae_with_hayate_tiling._hayate_vae_tiling_override = True  # type: ignore[attr-defined]
    load_vae_with_hayate_tiling._hayate_vae_tile_size = tile_size  # type: ignore[attr-defined]
    model_loader.load_vae = load_vae_with_hayate_tiling
    return True
