from __future__ import annotations

import json

import numpy as np
import pytest

from hayate.benchmark import video_compare
from hayate.benchmark.video_compare import _metric_summary, compare_videos, frame_ssim


class _Capture:
    def __init__(self, frames: list[np.ndarray], fps: float = 24.0):
        self.frames = iter(frames)
        self.fps = fps
        self.released = False

    def get(self, _property: int) -> float:
        return self.fps

    def read(self):
        try:
            return True, next(self.frames).copy()
        except StopIteration:
            return False, None

    def release(self) -> None:
        self.released = True


def _patch_captures(monkeypatch, reference: _Capture, candidate: _Capture) -> None:
    captures = iter((reference, candidate))
    monkeypatch.setattr(video_compare, "_open_video", lambda _path: next(captures))


def test_frame_ssim_is_one_for_identical_frames():
    frame = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    assert frame_ssim(frame, frame) == 1.0


def test_frame_ssim_detects_visible_difference():
    reference = np.zeros((64, 64), dtype=np.uint8)
    candidate = reference.copy()
    candidate[16:48, 16:48] = 255
    assert frame_ssim(reference, candidate) < 0.8


def test_metric_summary_is_json_safe_when_empty():
    assert _metric_summary([]) == {
        "mean": None,
        "minimum": None,
        "maximum": None,
        "p95": None,
    }


def test_metric_summary_includes_distribution_bounds():
    summary = _metric_summary([1.0, 2.0, 3.0, 4.0])
    assert summary["mean"] == 2.5
    assert summary["minimum"] == 1.0
    assert summary["maximum"] == 4.0
    assert summary["p95"] == 3.8499999999999996


def test_compare_videos_reports_domain_fps_and_json_safe_identical_count(
    monkeypatch, tmp_path
):
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    reference = _Capture([frame, frame])
    candidate = _Capture([frame, frame])
    _patch_captures(monkeypatch, reference, candidate)

    result = compare_videos(tmp_path / "reference.mp4", tmp_path / "candidate.mp4")

    assert result["comparison_domain"] == "decoded_mp4_bgr8"
    assert result["fps"] == 24.0
    assert result["frames"] == 2
    assert result["psnr_db"]["infinite_frames"] == 2
    assert result["psnr_db"]["finite_mean"] is None
    json.dumps(result, allow_nan=False)
    assert reference.released and candidate.released


def test_compare_videos_rejects_fps_mismatch(monkeypatch, tmp_path):
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    reference = _Capture([frame], fps=24.0)
    candidate = _Capture([frame], fps=30.0)
    _patch_captures(monkeypatch, reference, candidate)

    with pytest.raises(ValueError, match="FPS mismatch"):
        compare_videos(tmp_path / "reference.mp4", tmp_path / "candidate.mp4")
    assert reference.released and candidate.released


def test_compare_videos_rejects_frame_count_mismatch(monkeypatch, tmp_path):
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    reference = _Capture([frame, frame])
    candidate = _Capture([frame])
    _patch_captures(monkeypatch, reference, candidate)

    with pytest.raises(ValueError, match="different frame counts"):
        compare_videos(tmp_path / "reference.mp4", tmp_path / "candidate.mp4")


def test_compare_videos_rejects_empty_streams(monkeypatch, tmp_path):
    reference = _Capture([])
    candidate = _Capture([])
    _patch_captures(monkeypatch, reference, candidate)

    with pytest.raises(ValueError, match="no decodable frames"):
        compare_videos(tmp_path / "reference.mp4", tmp_path / "candidate.mp4")
