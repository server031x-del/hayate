from __future__ import annotations

import json
import re
from dataclasses import dataclass

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
VAE_BINDING_RE = re.compile(r"Video VAE binding:\s*(\d+)/(\d+)")
W4A8_BINDING_RE = re.compile(r"W4A8 binding:\s*(\d+)/(\d+)")
DENOISE_RE = re.compile(r"denoise:\s*(\d+)%")
EASYCACHE_RE = re.compile(r"HAYATE_EASYCACHE_(SKIP|FULL)\s+step=(\d+)/(\d+)")
METRICS_PREFIX = "HAYATE_RUNTIME_METRICS "
EVENT_PREFIX = "HAYATE_EVENT "


def _clock_seconds(value: str) -> int | None:
    try:
        parts = [int(part) for part in value.split(":")]
    except ValueError:
        return None
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return None


@dataclass(frozen=True)
class ProgressUpdate:
    progress: float
    stage: str
    detail: str
    eta_seconds: int | None = None
    runtime_metrics: dict | None = None


class H3ProgressParser:
    """Map upstream console output to stable user-facing generation stages."""

    def __init__(self):
        self.progress = 1.0
        self.stage = "起動準備"
        self.detail = "生成プロセスを開始しています"
        self.eta_seconds: int | None = None
        self.runtime_metrics: dict | None = None

    def feed(self, raw_line: str) -> ProgressUpdate | None:
        line = ANSI_RE.sub("", raw_line).strip()
        if not line:
            return None
        if EVENT_PREFIX in line:
            try:
                event_text = line.split(EVENT_PREFIX, 1)[1].lstrip()
                event, _ = json.JSONDecoder().raw_decode(event_text)
            except (IndexError, json.JSONDecodeError):
                return None
            if event.get("type") == "metrics":
                self.runtime_metrics = event.get("runtime_metrics")
                self.progress, self.stage, self.detail = (
                    99.0,
                    "最終処理",
                    "実行統計を保存しています",
                )
            elif event.get("type") in {"progress", "process"}:
                self.progress = float(event.get("progress", self.progress))
                self.stage = str(event.get("stage", self.stage))
                self.detail = str(event.get("detail", self.detail))
                eta = event.get("eta_seconds")
                self.eta_seconds = int(eta) if eta is not None else self.eta_seconds
            else:
                return None
            return ProgressUpdate(
                round(self.progress, 2),
                self.stage,
                self.detail,
                self.eta_seconds,
                self.runtime_metrics,
            )
        changed = False

        if "task=" in line and "frames=" in line:
            self.progress, self.stage, self.detail = (
                3.0,
                "入力準備",
                "生成条件を確定しました",
            )
            changed = True
        elif "prompt cache hit" in line:
            self.progress, self.stage, self.detail = (
                6.0,
                "プロンプト処理",
                "キャッシュを再利用しています",
            )
            changed = True
        elif "Loading VAE" in line:
            self.progress, self.stage, self.detail = (
                8.0,
                "VAE読込",
                "映像デコーダーを準備しています",
            )
            changed = True
        elif match := VAE_BINDING_RE.search(line):
            current, total = map(int, match.groups())
            self.progress = 8.0 + 10.0 * current / max(total, 1)
            self.stage, self.detail = "VAE読込", f"INT8レイヤー {current}/{total}"
            changed = True
        elif "Audio VAE" in line and (
            "load" in line.lower() or "tensor source" in line
        ):
            self.progress, self.stage, self.detail = (
                19.0,
                "音声VAE読込",
                "音声デコーダーを準備しています",
            )
            changed = True
        elif "Loading DiT weights" in line:
            self.progress, self.stage, self.detail = (
                22.0,
                "Transformer読込",
                "W4A8モデルを展開しています",
            )
            changed = True
        elif match := W4A8_BINDING_RE.search(line):
            current, total = map(int, match.groups())
            self.progress = 22.0 + 13.0 * current / max(total, 1)
            self.stage, self.detail = (
                "Transformer読込",
                f"W4A8レイヤー {current}/{total}",
            )
            changed = True
        elif "transformer loaded" in line.lower():
            self.progress, self.stage, self.detail = (
                36.0,
                "推論準備完了",
                "デノイズを開始します",
            )
            changed = True
        elif match := DENOISE_RE.search(line):
            percent = min(100, int(match.group(1)))
            self.progress = 36.0 + 58.0 * percent / 100.0
            self.stage = "動画生成"
            self.detail = f"デノイズ {percent}%"
            eta_match = re.search(r"<([0-9:]+),", line)
            if eta_match:
                self.eta_seconds = _clock_seconds(eta_match.group(1))
            changed = True
        elif match := EASYCACHE_RE.search(line):
            kind, current, total = match.groups()
            action = "再利用" if kind == "SKIP" else "高精度計算"
            self.stage = "動画生成"
            self.detail = f"EasyCache {action} · {current}/{total}"
            changed = True
        elif "Video saved to:" in line or ":saved:" in line:
            self.progress, self.stage, self.detail = (
                98.0,
                "書き出し",
                "映像と音声を保存しました",
            )
            self.eta_seconds = 0
            changed = True
        elif line.startswith(METRICS_PREFIX):
            try:
                self.runtime_metrics = json.loads(line.removeprefix(METRICS_PREFIX))
            except json.JSONDecodeError:
                self.runtime_metrics = None
            self.progress, self.stage, self.detail = (
                99.0,
                "最終処理",
                "実行統計を保存しています",
            )
            changed = True

        if not changed:
            return None
        return ProgressUpdate(
            round(self.progress, 2),
            self.stage,
            self.detail,
            self.eta_seconds,
            self.runtime_metrics,
        )
