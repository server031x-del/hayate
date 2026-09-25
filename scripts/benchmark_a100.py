"""Fixed-seed A100 speed/quality matrix for HAYATE's MiniMax H3 profiles.

Run on the target adapter after downloading the A100 configuration::

    python scripts/benchmark_a100.py --upstream /path/to/h3 \
        --ckpt-dir models/minimax-h3-snapshot --config configs/models.a100.yaml

Every variant changes one thing relative to ``a100_detail`` (or is the
reference), so the report separates the effect of SageAttention, INT8 tensor
cores, EasyCache, PDD, and resident placement.  Quality is measured against
``a100_quality`` (50 points, SDPA, no cache, weight-only INT8 error) with
decoded-frame SSIM/PSNR; these are difference scores, not perceptual ratings,
so inspect the contact sheets before changing a default.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hayate.backends.minimax_h3.generation import (
    ExternalH3GenerationBackend,
    GenerationRequest,
)
from hayate.benchmark.video_compare import compare_videos
from hayate.models import ModelRegistry
from hayate.profiles import get_generation_profile

DEFAULT_PROMPT = (
    "integrated_multimodal_description: [Shot 1] A silver sports car accelerates along a wet "
    "coastal highway at golden hour, low tracking shot, crisp reflections on the bodywork, "
    "photorealistic cinematic commercial, stable camera, no text, no logo.\n\n"
    "overall_soundscape: A rising engine note and tire spray on wet asphalt.\n\n"
    "non_diegetic_music: A restrained, pulsing orchestral cue."
)

# (name, base profile, overrides).  The reference must come first.
VARIANTS: tuple[tuple[str, str, dict], ...] = (
    ("reference", "a100_quality", {}),
    ("a100_detail", "a100_detail", {}),
    ("detail_sdpa", "a100_detail", {"attention_backend": "sdpa"}),
    ("detail_dequant", "a100_detail", {"int8_fast": False}),
    ("detail_nocache", "a100_detail", {"easycache": False}),
    ("a100_pdd", "a100_pdd", {}),
    # The consumer placement on the same card: what A100 residency buys.
    ("consumer_placement", "fast_sage_detail", {}),
)

REQUEST_FIELDS = {field.name for field in dataclasses.fields(GenerationRequest)}


def build_request(args: argparse.Namespace, profile_name: str, overrides: dict, output: Path) -> GenerationRequest:
    profile = get_generation_profile(profile_name).to_dict()
    profile.update(overrides)
    values = {key: value for key, value in profile.items() if key in REQUEST_FIELDS}
    if profile.get("pdd"):
        values["pdd_checkpoint"] = args.pdd_checkpoint
        values["pdd_adaln_affine"] = args.pdd_adaln_affine
    return GenerationRequest(
        prompt=args.prompt,
        checkpoint_dir=args.ckpt_dir,
        output=output,
        task="t2va",
        height=args.height,
        width=args.width,
        frames=args.frames,
        seed=args.seed,
        prompt_cache=args.output_dir / "prompt-cache.safetensors",
        gpu_device=args.gpu,
        **values,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "models.a100.yaml")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmarks" / "a100")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--gpu", default="auto", help="GPU UUID/index; pin it on mixed-GPU hosts")
    parser.add_argument("--pdd-checkpoint", type=Path, default=ROOT / "models" / "lora" / "MiniMax-H3-FL2VA-Acc-8Step.safetensors")
    parser.add_argument("--pdd-adaln-affine", type=Path, default=ROOT / "models" / "lora" / "adaln_affine.safetensors")
    parser.add_argument("--only", nargs="*", default=None, help="subset of variant names (reference is always kept)")
    parser.add_argument("--dry-run", action="store_true", help="print each preflight without generating")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    backend = ExternalH3GenerationBackend(args.upstream, ModelRegistry.load(args.config), python=args.python)
    selected = [v for v in VARIANTS if args.only is None or v[0] == "reference" or v[0] in args.only]
    results = []
    reference_video: Path | None = None
    for name, profile_name, overrides in selected:
        output = args.output_dir / f"{name}.mp4"
        plan = backend.plan(build_request(args, profile_name, overrides, output))
        print(f"\n== {name} ({profile_name} {overrides or ''})")
        if not plan.executable:
            for issue in plan.issues:
                print(f"  BLOCKER: {issue}")
            results.append({"name": name, "blocked": list(plan.issues)})
            continue
        if args.dry_run:
            print("  READY:", " ".join(plan.command[-12:]))
            continue
        started = time.perf_counter()
        result = backend.execute(plan)
        wall = time.perf_counter() - started
        metrics = result.runtime_metrics or {}
        row = {
            "name": name,
            "profile": profile_name,
            "overrides": overrides,
            "returncode": result.returncode,
            "wall_seconds": round(wall, 1),
            "cuda_peak_allocated_gib": round((metrics.get("cuda_peak_allocated_bytes") or 0) / 1024**3, 2),
            "easycache": metrics.get("easycache"),
            "output": str(output),
        }
        if result.returncode == 0 and output.is_file():
            if name == "reference":
                reference_video = output
            elif reference_video is not None:
                comparison = compare_videos(reference_video, output)
                row["ssim_mean"] = round(comparison["ssim"]["mean"], 4)
                row["ssim_min"] = round(comparison["ssim"]["minimum"], 4)
                row["psnr_db"] = comparison["psnr_db"]["finite_mean"]
        results.append(row)
        print("  ", json.dumps(row, ensure_ascii=False))

    if args.dry_run:
        return 0
    report = args.output_dir / "report.json"
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n| variant | wall s | VRAM peak GiB | SSIM vs ref (mean/min) | PSNR dB |")
    print("|---|---:|---:|---:|---:|")
    for row in results:
        if "blocked" in row:
            print(f"| {row['name']} | blocked | | | |")
            continue
        ssim = f"{row['ssim_mean']} / {row['ssim_min']}" if "ssim_mean" in row else "reference" if row["name"] == "reference" else "—"
        psnr = row.get("psnr_db")
        print(f"| {row['name']} | {row['wall_seconds']} | {row['cuda_peak_allocated_gib']} | {ssim} | {psnr if psnr is None else round(psnr, 2)} |")
    print(f"\nReport: {report}")
    return 0 if all(row.get("returncode", 1) == 0 for row in results if "blocked" not in row) else 1


if __name__ == "__main__":
    raise SystemExit(main())
