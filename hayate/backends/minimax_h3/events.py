from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

EVENT_PREFIX = "HAYATE_EVENT "


def emit_event(event_type: str, **payload: Any) -> None:
    event = {"schema": 1, "type": event_type, "timestamp": time.time(), **payload}
    print(
        "\n" + EVENT_PREFIX + json.dumps(event, ensure_ascii=True, sort_keys=True),
        flush=True,
    )


class _ProgressLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            event = self._event_for_message(message)
            if event is not None:
                emit_event("progress", **event)
        except (OSError, RuntimeError, TypeError, ValueError):
            self.handleError(record)

    @staticmethod
    def _event_for_message(message: str) -> dict[str, Any] | None:
        if "task=" in message and "frames=" in message:
            return {
                "phase": "input",
                "progress": 3.0,
                "stage": "入力準備",
                "detail": message,
            }
        if "prompt cache hit" in message:
            return {
                "phase": "conditioning",
                "progress": 6.0,
                "stage": "プロンプト処理",
                "detail": "キャッシュを再利用しています",
            }
        if "Loading VAE" in message:
            return {
                "phase": "load_vae",
                "progress": 8.0,
                "stage": "VAE読込",
                "detail": "映像VAEを準備しています",
            }
        if match := re.search(r"Video VAE binding:\s*(\d+)/(\d+)", message):
            current, total = map(int, match.groups())
            return {
                "phase": "load_vae",
                "progress": 8.0 + 10.0 * current / max(total, 1),
                "stage": "VAE読込",
                "detail": f"INT8レイヤー {current}/{total}",
                "current": current,
                "total": total,
            }
        if "Audio VAE" in message and (
            "load" in message.lower() or "tensor source" in message
        ):
            return {
                "phase": "load_audio_vae",
                "progress": 19.0,
                "stage": "音声VAE読込",
                "detail": "音声VAEを準備しています",
            }
        if "Loading DiT weights" in message:
            return {
                "phase": "load_transformer",
                "progress": 22.0,
                "stage": "Transformer読込",
                "detail": "W4A8モデルを展開しています",
            }
        if match := re.search(r"W4A8 binding:\s*(\d+)/(\d+)", message):
            current, total = map(int, match.groups())
            return {
                "phase": "load_transformer",
                "progress": 22.0 + 13.0 * current / max(total, 1),
                "stage": "Transformer読込",
                "detail": f"W4A8レイヤー {current}/{total}",
                "current": current,
                "total": total,
            }
        if "transformer loaded" in message.lower():
            return {
                "phase": "ready",
                "progress": 36.0,
                "stage": "推論準備完了",
                "detail": "デノイズを開始します",
            }
        if "stop requested: decoding" in message:
            return {
                "phase": "decode",
                "progress": 95.0,
                "stage": "途中保存",
                "detail": "現在の状態をデコードしています",
            }
        if ":saved:" in message:
            return {
                "phase": "save",
                "progress": 98.0,
                "stage": "書き出し",
                "detail": "映像と音声を保存しました",
            }
        return None


def install_structured_events(module: Any) -> None:
    """Emit a stable HAYATE protocol while leaving upstream execution unchanged."""

    root = logging.getLogger()
    if not any(isinstance(handler, _ProgressLogHandler) for handler in root.handlers):
        root.addHandler(_ProgressLogHandler())

    original_tqdm = module.tqdm

    class EventTqdm(original_tqdm):
        def update(self, n: float = 1):
            result = super().update(n)
            if str(getattr(self, "desc", "")).strip() == "denoise":
                total = int(self.total or 0)
                current = int(self.n)
                rate = float(self.format_dict.get("rate") or 0.0)
                eta = int(max(0, (total - current) / rate)) if rate > 0 else None
                percent = 100.0 * current / max(total, 1)
                emit_event(
                    "progress",
                    phase="denoise",
                    progress=36.0 + 58.0 * percent / 100.0,
                    stage="動画生成",
                    detail=f"デノイズ {current}/{total}",
                    current=current,
                    total=total,
                    eta_seconds=eta,
                )
            return result

    module.tqdm = EventTqdm
