from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def frame_ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = reference.astype(np.float32)
    candidate = candidate.astype(np.float32)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_x = cv2.GaussianBlur(reference, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(candidate, (11, 11), 1.5)
    sigma_x = cv2.GaussianBlur(reference * reference, (11, 11), 1.5) - mu_x**2
    sigma_y = cv2.GaussianBlur(candidate * candidate, (11, 11), 1.5) - mu_y**2
    sigma_xy = cv2.GaussianBlur(reference * candidate, (11, 11), 1.5) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )
    return float(score.mean())


def _open_video(path: Path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    return capture


def _metric_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "minimum": None, "maximum": None, "p95": None}
    return {
        "mean": float(np.mean(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "p95": float(np.percentile(values, 95)),
    }


def compare_videos(reference: Path, candidate: Path) -> dict:
    ref_capture = _open_video(reference)
    candidate_capture = _open_video(candidate)
    ssim_values: list[float] = []
    psnr_values: list[float] = []
    mae_values: list[float] = []
    ref_motion: list[float] = []
    candidate_motion: list[float] = []
    previous_reference = previous_candidate = None
    width = height = None
    reference_fps = float(ref_capture.get(cv2.CAP_PROP_FPS))
    candidate_fps = float(candidate_capture.get(cv2.CAP_PROP_FPS))

    try:
        if reference_fps <= 0 or candidate_fps <= 0:
            raise ValueError(
                f"invalid video FPS: {reference_fps} and {candidate_fps}"
            )
        if not math.isclose(reference_fps, candidate_fps, rel_tol=0, abs_tol=1e-3):
            raise ValueError(
                f"video FPS mismatch: {reference_fps} != {candidate_fps}"
            )
        while True:
            ref_ok, ref_frame = ref_capture.read()
            candidate_ok, candidate_frame = candidate_capture.read()
            if ref_ok != candidate_ok:
                raise ValueError("videos have different frame counts")
            if not ref_ok:
                break
            if ref_frame.shape != candidate_frame.shape:
                raise ValueError(
                    f"frame shape mismatch: {ref_frame.shape} != {candidate_frame.shape}"
                )
            height, width = ref_frame.shape[:2]
            ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
            candidate_gray = cv2.cvtColor(candidate_frame, cv2.COLOR_BGR2GRAY)
            ssim_values.append(frame_ssim(ref_gray, candidate_gray))
            mse = float(np.mean((ref_frame.astype(np.float32) - candidate_frame) ** 2))
            psnr_values.append(math.inf if mse == 0 else 10 * math.log10(255**2 / mse))
            mae_values.append(float(np.mean(np.abs(ref_frame.astype(np.float32) - candidate_frame))))
            if previous_reference is not None:
                ref_motion.append(
                    float(np.mean(np.abs(ref_gray.astype(np.float32) - previous_reference)))
                )
                candidate_motion.append(
                    float(
                        np.mean(
                            np.abs(candidate_gray.astype(np.float32) - previous_candidate)
                        )
                    )
                )
            previous_reference = ref_gray.astype(np.float32)
            previous_candidate = candidate_gray.astype(np.float32)
    finally:
        ref_capture.release()
        candidate_capture.release()

    if not ssim_values:
        raise ValueError("videos contain no decodable frames")
    finite_psnr = [value for value in psnr_values if math.isfinite(value)]
    psnr_summary = _metric_summary(finite_psnr)
    return {
        "reference": str(reference.resolve(strict=False)),
        "candidate": str(candidate.resolve(strict=False)),
        "comparison_domain": "decoded_mp4_bgr8",
        "frames": len(ssim_values),
        "fps": reference_fps,
        "width": width,
        "height": height,
        "ssim": {
            "mean": float(np.mean(ssim_values)),
            "minimum": float(np.min(ssim_values)),
            "maximum": float(np.max(ssim_values)),
        },
        "psnr_db": {
            "finite_mean": psnr_summary["mean"],
            "finite_minimum": psnr_summary["minimum"],
            "infinite_frames": len(psnr_values) - len(finite_psnr),
        },
        "mae_8bit": {
            "mean": float(np.mean(mae_values)),
            "maximum": float(np.max(mae_values)),
        },
        "temporal_luma_delta": {
            "reference": _metric_summary(ref_motion),
            "candidate": _metric_summary(candidate_motion),
            "candidate_minus_reference": (
                float(np.mean(candidate_motion) - np.mean(ref_motion)) if ref_motion else 0.0
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two videos frame by frame")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = compare_videos(args.reference, args.candidate)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
