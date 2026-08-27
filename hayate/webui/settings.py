from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class WebUISettings:
    config_path: str
    upstream_path: str
    checkpoint_dir: str
    output_dir: str
    python_path: str
    prompt_cache_dir: str
    pdd_checkpoint_path: str
    pdd_adaln_affine_path: str

    @classmethod
    def defaults(cls, workspace: Path) -> WebUISettings:
        workspace = workspace.resolve(strict=False)
        upstream = os.environ.get("HAYATE_H3_CHECKOUT") or str(
            workspace / "upstream" / "h3"
        )
        return cls(
            config_path=str(workspace / "configs" / "models.yaml"),
            upstream_path=upstream,
            checkpoint_dir=str(workspace / "models" / "minimax-h3-snapshot"),
            output_dir=str(workspace / "outputs"),
            python_path=str(Path(sys.executable).resolve(strict=False)),
            prompt_cache_dir=str(workspace / "outputs" / "prompt_cache"),
            pdd_checkpoint_path=str(
                workspace / "models" / "lora" / "MiniMax-H3-FL2VA-Acc-8Step.safetensors"
            ),
            pdd_adaln_affine_path=str(workspace / "models" / "lora" / "adaln_affine.safetensors"),
        )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    def resolved(self) -> WebUISettings:
        return WebUISettings(
            **{
                key: str(Path(value).expanduser().resolve(strict=False))
                for key, value in self.to_dict().items()
            }
        )


class SettingsStore:
    def __init__(self, path: Path, workspace: Path):
        self.path = path.resolve(strict=False)
        self.workspace = workspace.resolve(strict=False)
        self._lock = threading.RLock()

    def load(self) -> WebUISettings:
        with self._lock:
            defaults = WebUISettings.defaults(self.workspace)
            if not self.path.is_file():
                return defaults.resolved()
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            values = defaults.to_dict()
            values.update(
                {key: str(value) for key, value in payload.items() if key in values}
            )
            return WebUISettings(**values).resolved()

    def save(self, settings: WebUISettings) -> WebUISettings:
        resolved = settings.resolved()
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(resolved.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(self.path)
        return resolved

    def update(self, values: dict[str, str]) -> WebUISettings:
        current = self.load().to_dict()
        unknown = set(values) - set(current)
        if unknown:
            raise ValueError(f"unknown settings: {', '.join(sorted(unknown))}")
        current.update({key: str(value) for key, value in values.items()})
        return self.save(WebUISettings(**current))

    @staticmethod
    def readiness(settings: WebUISettings) -> dict[str, dict[str, object]]:
        checks = {
            "config_path": (Path(settings.config_path).is_file(), "file"),
            "upstream_path": (Path(settings.upstream_path).is_dir(), "directory"),
            "checkpoint_dir": (Path(settings.checkpoint_dir).is_dir(), "directory"),
            "output_dir": (Path(settings.output_dir).is_dir(), "directory"),
            "python_path": (Path(settings.python_path).is_file(), "file"),
            "prompt_cache_dir": (Path(settings.prompt_cache_dir).is_dir(), "directory"),
            "pdd_checkpoint_path": (Path(settings.pdd_checkpoint_path).is_file(), "file"),
            "pdd_adaln_affine_path": (Path(settings.pdd_adaln_affine_path).is_file(), "file"),
        }
        return {
            key: {"ready": ready, "expected": expected, "value": getattr(settings, key)}
            for key, (ready, expected) in checks.items()
        }
