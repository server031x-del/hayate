from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from hayate.errors import UpstreamContractError

AUDITED_COMMIT = "94220c1fdf14d6d9d40be06fb99f55b27c0d9024"
REPOSITORY = "https://github.com/maybleMyers/h3"

REQUIRED_COMPONENTS = {
    "pipeline": "minimax_engine/minimax_video/pipeline.py",
    "scheduler": "minimax_engine/minimax_video/scheduler.py",
    "transformer": "minimax_engine/minimax_video/transformer.py",
    "video_vae": "minimax_engine/minimax_video/vae_video.py",
    "audio_vae": "minimax_engine/minimax_video/vae_audio.py",
    "model_loader": "minimax_engine/minimax_video/model_loader.py",
    "conditioner": "minimax_engine/minimax_video/conditioner.py",
    "int8_quant": "minimax_engine/minimax_video/int8_quant.py",
    "progressive_load": "minimax_engine/minimax_video/progressive_load.py",
    "generation_cli": "minimax_engine/minimax_generate_video.py",
    "job_queue": "minimax_engine/wan_job_queue.py",
    "job_worker": "minimax_engine/wan_worker.py",
}


@dataclass(frozen=True)
class UpstreamValidation:
    checkout: Path
    valid: bool
    commit: str | None
    audited_commit: str
    missing_components: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "checkout": str(self.checkout),
            "valid": self.valid,
            "commit": self.commit,
            "audited_commit": self.audited_commit,
            "missing_components": list(self.missing_components),
            "warnings": list(self.warnings),
        }


class H3UpstreamAdapter:
    """Validate an external h3 checkout without importing or copying its engine."""

    def __init__(self, checkout: str | Path):
        self.checkout = Path(checkout).expanduser().resolve(strict=False)

    def component_path(self, name: str) -> Path:
        try:
            relative = REQUIRED_COMPONENTS[name]
        except KeyError as exc:
            raise UpstreamContractError(f"unknown upstream component: {name}") from exc
        return self.checkout / relative

    def _git_commit(self) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(self.checkout), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    def validate(self, *, require_audited_commit: bool = False) -> UpstreamValidation:
        missing = tuple(name for name in REQUIRED_COMPONENTS if not self.component_path(name).is_file())
        commit = self._git_commit()
        warnings: list[str] = []
        if commit is None:
            warnings.append("checkout commit could not be determined")
        elif commit != AUDITED_COMMIT:
            warnings.append(f"checkout commit {commit} differs from audited commit {AUDITED_COMMIT}")
        valid = not missing and (not require_audited_commit or commit == AUDITED_COMMIT)
        return UpstreamValidation(
            self.checkout, valid, commit, AUDITED_COMMIT, missing, tuple(warnings)
        )

    def require_valid(self, *, require_audited_commit: bool = False) -> UpstreamValidation:
        result = self.validate(require_audited_commit=require_audited_commit)
        if not result.valid:
            reasons = []
            if result.missing_components:
                reasons.append("missing: " + ", ".join(result.missing_components))
            reasons.extend(result.warnings)
            raise UpstreamContractError("invalid maybleMyers/h3 checkout: " + "; ".join(reasons))
        return result

