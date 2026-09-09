from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


HF_ENDPOINT = "https://huggingface.co"
MIN_FREE_SPACE_MARGIN = 512 * 1024 * 1024
DOWNLOAD_PROGRESS_INTERVAL_SECONDS = 1.0


class ModelSetupError(RuntimeError):
    """A safe, user-facing model setup failure."""


@dataclass(frozen=True)
class ModelArtifact:
    remote_path: str
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ModelAsset:
    id: str
    role: str
    label: str
    repo_id: str
    revision: str
    license: str
    license_url: str
    artifacts: tuple[ModelArtifact, ...]
    # ``execution_backend`` is descriptive metadata for the WebUI catalog. It
    # is deliberately not used to silently select a loader: experimental
    # checkpoints must pass their own backend preflight first.
    execution_backend: str = "mayble_h3"
    experimental: bool = False
    notes: tuple[str, ...] = ()
    # Some upstream artifacts are registered for provenance/header auditing
    # only (for example a ComfyUI single-file conversion).  Such an asset can
    # still be downloaded and verified, but must never enable a HAYATE profile.
    execution_supported: bool = True

    @property
    def size_bytes(self) -> int:
        return sum(item.size_bytes for item in self.artifacts)

    @property
    def source_url(self) -> str:
        return f"{HF_ENDPOINT}/{self.repo_id}/tree/{self.revision}"

    @property
    def manifest_sha256(self) -> str:
        if len(self.artifacts) == 1:
            return self.artifacts[0].sha256
        digest = hashlib.sha256()
        for item in sorted(self.artifacts, key=lambda value: value.relative_path):
            digest.update(
                f"{item.relative_path}\0{item.size_bytes}\0{item.sha256}\n".encode()
            )
        return digest.hexdigest()


MINIMAX_LICENSE = "MiniMax H3 Community License Agreement / repository terms"
MINIMAX_LICENSE_URL = f"{HF_ENDPOINT}/MiniMaxAI/MiniMax-H3/blob/main/LICENSE"


def _artifact(remote: str, relative: str, size: int, sha256: str) -> ModelArtifact:
    return ModelArtifact(remote, relative, size, sha256)


MODEL_ASSETS: tuple[ModelAsset, ...] = (
    ModelAsset(
        "transformer_w4a8",
        "transformer",
        "MiniMax H3 FL2VA Pruned W4A8 Mixed",
        "Kijai/MiniMax-H3-experimental",
        "f9c521b15b6883fa1c353f86f806ad76311aa5a4",
        MINIMAX_LICENSE,
        MINIMAX_LICENSE_URL,
        (
            _artifact(
                "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
                "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
                12_540_858_008,
                "01aa7b92c007c599890461c325f9b7e3c96fb06c36f242f95b62f7f20e538dec",
            ),
        ),
    ),
    ModelAsset(
        "transformer_fastvideo_vsa_4step",
        "transformer",
        "MiniMax H3 FastH3 VSA DataFree 4-Step INT8 ConvRot（外部ComfyUI用）",
        "Kijai/MiniMax-H3-experimental",
        "f4cac997f880e93cf6940af61ee8d58ef31ff7f7",
        MINIMAX_LICENSE,
        MINIMAX_LICENSE_URL,
        (
            _artifact(
                "minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors",
                "minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors",
                22_898_594_920,
                "7221ae65d78780354d51e5048d29728d9f1f8fb9baf50b1dd3df85f5101413d",
            ),
        ),
        execution_backend="external_comfyui_vsa",
        experimental=True,
        execution_supported=False,
        notes=(
            "ComfyUI単一ファイル形式。現行maybleMyers/h3 W4A8ローダーでは使用しません",
            "HAYATEから直接生成する場合は公式FastVideoディレクトリ型スナップショットを別途配置します",
            "このファイルはHAYATEではヘッダー診断・完全性確認のみ（外部ComfyUI/VSA用）",
        ),
    ),
    ModelAsset(
        "text_encoder_nvfp4_awq",
        "text_encoder",
        "Qwen3-VL 32B MiniMax H3 NVFP4 AWQ",
        "Comfy-Org/MiniMax-H3",
        "4cc1d817b6184899b41293954329f576cb5ae86b",
        MINIMAX_LICENSE,
        MINIMAX_LICENSE_URL,
        (
            _artifact(
                "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                15_687_142_551,
                "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6",
            ),
        ),
    ),
    ModelAsset(
        "video_vae_int8_convrot",
        "video_vae",
        "MiniMax H3 Video VAE INT8 ConvRot",
        "Kijai/MiniMax-H3-experimental",
        "f9c521b15b6883fa1c353f86f806ad76311aa5a4",
        MINIMAX_LICENSE,
        MINIMAX_LICENSE_URL,
        (
            _artifact(
                "minimax_h3_video_vae_int8_convrot.safetensors",
                "minimax_h3_video_vae_int8_convrot.safetensors",
                3_171_670_912,
                "9bb2d96f218c76babd85e0611b85ca8fb330a90546c01a0005e8a58a59593410",
            ),
        ),
    ),
    ModelAsset(
        "audio_vae_fp32",
        "audio_vae",
        "MiniMax H3 Audio VAE FP32",
        "Comfy-Org/MiniMax-H3",
        "4cc1d817b6184899b41293954329f576cb5ae86b",
        MINIMAX_LICENSE,
        MINIMAX_LICENSE_URL,
        (
            _artifact(
                "vae/minimax_h3_audio_vae_fp32.safetensors",
                "vae/minimax_h3_audio_vae_fp32.safetensors",
                605_254_808,
                "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
            ),
        ),
    ),
    ModelAsset(
        "pdd_fl2va_8step",
        "pdd_lora",
        "MiniMax H3 FL2VA Acc 8-Step LoRA",
        "alibaba-pai/MiniMax-H3-Acc-LoRAs",
        "335001fb9e5455d68a0caa18ec2e319072150328",
        "Repository license metadata and MiniMax-derived weight terms",
        f"{HF_ENDPOINT}/alibaba-pai/MiniMax-H3-Acc-LoRAs",
        (
            _artifact(
                "MiniMax-H3-FL2VA-Acc-8Step.safetensors",
                "lora/MiniMax-H3-FL2VA-Acc-8Step.safetensors",
                1_372_450_680,
                "0b29be7042d883970eb0c20774a9ba03d95669ed80a721bb4d21be8ea0d0a196",
            ),
        ),
    ),
    ModelAsset(
        "pdd_adaln_affine",
        "pdd_affine",
        "MiniMax H3 Pruned AdaLN affine map",
        "multimodalart/MiniMax-H3-Pruned",
        "1a0ef5e65b639e84af81d883817968532180e9c7",
        "Repository license metadata and MiniMax-derived weight terms",
        f"{HF_ENDPOINT}/multimodalart/MiniMax-H3-Pruned",
        (
            _artifact(
                "transformer/adaln_affine.safetensors",
                "lora/adaln_affine.safetensors",
                96_960,
                "34f285e7aeae741666868bf5912506ab4793098357a01a9d8358f6ce7704d532",
            ),
        ),
    ),
    ModelAsset(
        "checkpoint_support",
        "checkpoint_support",
        "MiniMax H3 support files (no base weight shards)",
        "MiniMaxAI/MiniMax-H3",
        "42ed227ee7df40d41602854ae760620d6eb651fe",
        MINIMAX_LICENSE,
        MINIMAX_LICENSE_URL,
        (
            _artifact("transformer/config.json", "minimax-h3-snapshot/transformer/config.json", 546, "74c11bff524336576096993cbfcdcdc2ef4fa2fa4409df693bdcbc6c666282ae"),
            _artifact("vae/config.json", "minimax-h3-snapshot/vae/config.json", 2_011, "78f67deec3d63aae807f2bfe7154bc1e26f6372cb20b63265fcbae1b62bb5745"),
            _artifact("audio_vae/config.json", "minimax-h3-snapshot/audio_vae/config.json", 2_271, "9a3c645ff892b376c6f5f4c8685964cd75474731af594ff058492a0000caabb6"),
            _artifact("text_encoder/config.json", "minimax-h3-snapshot/text_encoder/config.json", 1_474, "d2dd0c60d01b9e195d9447c52da61c7302d28828524914c044d9c6e1b81d0427"),
            _artifact("scheduler/scheduler_config.json", "minimax-h3-snapshot/scheduler/scheduler_config.json", 97, "8fa6c3aa70dc9e691e1a6df899fd1b6f75f70481a27cee6e18a303817075c304"),
            _artifact("audio_scheduler/scheduler_config.json", "minimax-h3-snapshot/audio_scheduler/scheduler_config.json", 96, "804780f7133477067bd6bbfbc02dc8b3cf9feeb400f97c08f5b1d5f6cbab3840"),
            _artifact("tokenizer/merges.txt", "minimax-h3-snapshot/tokenizer/merges.txt", 1_671_839, "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"),
            _artifact("tokenizer/tokenizer_config.json", "minimax-h3-snapshot/tokenizer/tokenizer_config.json", 11_003, "a07e942ac874baa13758de8d1fbdb186683cc03416b5589e1b6671c6b3057c68"),
            _artifact("tokenizer/vocab.json", "minimax-h3-snapshot/tokenizer/vocab.json", 2_776_833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
            _artifact("processor/chat_template.json", "minimax-h3-snapshot/processor/chat_template.json", 5_499, "5c72a170d2a4a1a3bc5adad2e689ae28138a9700e5b8c96c0266331e86c0acce"),
            _artifact("processor/merges.txt", "minimax-h3-snapshot/processor/merges.txt", 1_671_839, "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"),
            _artifact("processor/preprocessor_config.json", "minimax-h3-snapshot/processor/preprocessor_config.json", 390, "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"),
            _artifact("processor/tokenizer_config.json", "minimax-h3-snapshot/processor/tokenizer_config.json", 11_003, "a07e942ac874baa13758de8d1fbdb186683cc03416b5589e1b6671c6b3057c68"),
            _artifact("processor/video_preprocessor_config.json", "minimax-h3-snapshot/processor/video_preprocessor_config.json", 385, "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"),
            _artifact("processor/vocab.json", "minimax-h3-snapshot/processor/vocab.json", 2_776_833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
            _artifact("model_index.json", "minimax-h3-snapshot/model_index.json", 2_936, "5a587fe13b2371427415ac892463142683aefcd8d322e274a3a095eac37ac7d2"),
        ),
    ),
)


Downloader = Callable[[ModelAsset, ModelArtifact, Path], Path]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{value} B"


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds <= 0:
        return "計算中"
    total = max(1, int(round(seconds)))
    minutes, remainder = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}時間{minutes:02d}分"
    if minutes:
        return f"{minutes}分{remainder:02d}秒"
    return f"{remainder}秒"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _default_downloader(asset: ModelAsset, artifact: ModelArtifact, target: Path) -> Path:
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=asset.repo_id,
        filename=artifact.remote_path,
        revision=asset.revision,
        local_dir=str(target),
        endpoint=HF_ENDPOINT,
        token=False,
    )
    return Path(downloaded)


class ModelSetupService:
    """Detect and install only the audited HAYATE model catalog."""

    def __init__(
        self,
        workspace: Path,
        *,
        state_path: Path | None = None,
        downloader: Downloader | None = None,
        assets: tuple[ModelAsset, ...] = MODEL_ASSETS,
    ):
        self.workspace = workspace.resolve(strict=False)
        self.models_root = (self.workspace / "models").resolve(strict=False)
        self.state_path = (
            state_path or self.workspace / "data" / "webui" / "model-setup.json"
        ).resolve(strict=False)
        self._downloader = downloader or _default_downloader
        self._catalog = tuple(assets)
        self._assets = {asset.id: asset for asset in self._catalog}
        if len(self._assets) != len(self._catalog):
            raise ValueError("model asset ids must be unique")
        self._lock = threading.RLock()
        self._state = self._load_state()
        self._active_assets: set[str] = set()
        self._recover_interrupted()

    def _load_state(self) -> dict:
        empty = {"downloads": {}, "verified": {}, "license_acceptance": {}}
        if not self.state_path.is_file():
            return empty
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return empty
        if not isinstance(payload, dict):
            return empty

        def mapping(name: str) -> dict:
            value = payload.get(name)
            if not isinstance(value, dict):
                return {}
            # State is persisted by HAYATE itself, but tolerate hand-edits or
            # an interrupted migration so a malformed child entry cannot
            # prevent the WebUI from starting.
            return {
                str(key): dict(item)
                for key, item in value.items()
                if isinstance(item, dict)
            }

        return {
            "downloads": mapping("downloads"),
            "verified": mapping("verified"),
            "license_acceptance": mapping("license_acceptance"),
        }

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.state_path)

    def _recover_interrupted(self) -> None:
        changed = False
        for download in self._state["downloads"].values():
            if download.get("status") in {"queued", "downloading", "verifying"}:
                raw_temp = download.get("temp_dir")
                if isinstance(raw_temp, str) and raw_temp:
                    temporary = Path(raw_temp).resolve(strict=False)
                    try:
                        temporary.relative_to(self.models_root)
                    except ValueError:
                        temporary = None
                    if (
                        temporary is not None
                        and temporary.name.startswith(".hayate-download-")
                        and temporary.is_dir()
                    ):
                        shutil.rmtree(temporary, ignore_errors=True)
                download.update(
                    status="interrupted",
                    message="前回のモデル取得は中断されました。再実行できます。",
                    finished_at=_utc_now(),
                    temp_dir=None,
                )
                changed = True
        if changed:
            self._save_state()

    def _target(self, relative_path: str) -> Path:
        candidate = (self.models_root / relative_path).resolve(strict=False)
        try:
            candidate.relative_to(self.models_root)
        except ValueError as exc:
            raise ModelSetupError("モデル保存先が標準modelsフォルダの外です") from exc
        return candidate

    def prepare(self) -> dict:
        directories = (
            self.models_root,
            self.models_root / "minimax-h3-snapshot",
            self.models_root / "text_encoders",
            self.models_root / "vae",
            self.models_root / "lora",
            self.models_root / "fastvideo",
        )
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        return self.status()

    def _verification_is_current(self, asset: ModelAsset) -> bool:
        record = self._state["verified"].get(asset.id)
        if not isinstance(record, dict) or record.get("revision") != asset.revision:
            return False
        files = record.get("files")
        if not isinstance(files, dict):
            return False
        for artifact in asset.artifacts:
            path = self._target(artifact.relative_path)
            entry = files.get(artifact.relative_path)
            if not path.is_file() or not isinstance(entry, dict):
                return False
            stat = path.stat()
            if (
                stat.st_size != artifact.size_bytes
                or entry.get("size_bytes") != stat.st_size
                or entry.get("mtime_ns") != stat.st_mtime_ns
                or entry.get("sha256") != artifact.sha256
            ):
                return False
        return True

    def _asset_status(self, asset: ModelAsset) -> dict:
        paths = [self._target(item.relative_path) for item in asset.artifacts]
        exists_count = sum(path.is_file() for path in paths)
        invalid_size = any(
            path.is_file() and path.stat().st_size != artifact.size_bytes
            for path, artifact in zip(paths, asset.artifacts, strict=True)
        )
        verified = self._verification_is_current(asset)
        latest_download = next(
            (
                value
                for value in reversed(tuple(self._state["downloads"].values()))
                if value.get("asset_id") == asset.id
            ),
            None,
        )
        if asset.id in self._active_assets:
            status = "downloading"
        elif latest_download and latest_download.get("status") in {"failed", "interrupted"}:
            status = "error"
        elif invalid_size:
            status = "invalid"
        elif verified:
            status = "verified"
        elif exists_count == len(paths):
            status = "present_unverified"
        elif exists_count:
            status = "partial"
        else:
            status = "missing"
        first = asset.artifacts[0]
        first_path = self._target(first.relative_path)
        result = {
            "id": asset.id,
            "role": asset.role,
            "label": asset.label,
            "filename": first_path.name,
            "path": str(first_path if len(paths) == 1 else first_path.parents[len(Path(first.relative_path).parts) - 2]),
            "relative_path": first.relative_path,
            "size_bytes": asset.size_bytes,
            "size_label": _format_bytes(asset.size_bytes),
            "sha256": asset.manifest_sha256,
            "downloadable": True,
            "exists": exists_count == len(paths),
            "verified": verified,
            "sha256_ok": True if verified else (False if invalid_size else None),
            "status": status,
            "source_url": asset.source_url,
            "license": asset.license,
            "license_url": asset.license_url,
            "revision": asset.revision,
            "execution_backend": asset.execution_backend,
            "experimental": asset.experimental,
            "execution_supported": asset.execution_supported,
            "notes": list(asset.notes),
        }
        if latest_download:
            result.update(
                download_id=latest_download.get("id"),
                download_status=latest_download.get("status"),
                download_progress=latest_download.get("progress"),
                progress=latest_download.get("progress"),
                bytes_downloaded=latest_download.get("bytes_downloaded"),
                bytes_total=latest_download.get("bytes_total", asset.size_bytes),
                download_speed_bytes_per_sec=latest_download.get("download_speed_bytes_per_sec"),
                download_eta_seconds=latest_download.get("download_eta_seconds"),
                download_elapsed_seconds=latest_download.get("download_elapsed_seconds"),
                message=latest_download.get("message"),
            )
            if latest_download.get("status") in {"failed", "interrupted"}:
                result["error"] = latest_download.get("message")
        return result

    def status(self) -> dict:
        with self._lock:
            assets = [self._asset_status(asset) for asset in self._catalog]
            downloads = {
                key: dict(value) for key, value in self._state["downloads"].items()
            }
        return {
            "assets": assets,
            "downloads": downloads,
            "directories": {
                "models": {"path": str(self.models_root), "exists": self.models_root.is_dir()},
                "checkpoint": {"path": str(self.models_root / "minimax-h3-snapshot"), "exists": (self.models_root / "minimax-h3-snapshot").is_dir()},
                "text_encoders": {"path": str(self.models_root / "text_encoders"), "exists": (self.models_root / "text_encoders").is_dir()},
                "vae": {"path": str(self.models_root / "vae"), "exists": (self.models_root / "vae").is_dir()},
                "lora": {"path": str(self.models_root / "lora"), "exists": (self.models_root / "lora").is_dir()},
                "fastvideo": {"path": str(self.models_root / "fastvideo"), "exists": (self.models_root / "fastvideo").is_dir()},
            },
        }

    def downloads(self) -> dict:
        with self._lock:
            return {key: dict(value) for key, value in self._state["downloads"].items()}

    def start_download(self, asset_id: str, *, license_accepted: bool) -> dict:
        asset = self._assets.get(asset_id)
        if asset is None:
            raise KeyError(asset_id)
        if not license_accepted:
            raise ModelSetupError("ライセンス条件への同意が必要です")
        with self._lock:
            if asset_id in self._active_assets:
                raise ModelSetupError("このモデルは既にダウンロード中です")
            download_id = uuid.uuid4().hex
            job = {
                "id": download_id,
                "asset_id": asset.id,
                "status": "queued",
                "bytes_downloaded": 0,
                "bytes_total": asset.size_bytes,
                "progress": 0.0,
                "message": "ダウンロード待機中",
                "download_speed_bytes_per_sec": 0.0,
                "download_eta_seconds": None,
                "download_elapsed_seconds": 0.0,
                "started_at": _utc_now(),
                "finished_at": None,
            }
            self._active_assets.add(asset_id)
            self._state["downloads"][download_id] = job
            self._state["license_acceptance"][asset_id] = {
                "revision": asset.revision,
                "license": asset.license,
                "accepted_at": _utc_now(),
            }
            self._save_state()
        thread = threading.Thread(
            target=self._run_download,
            args=(download_id, asset),
            name=f"hayate-model-{asset.id}",
            daemon=True,
        )
        thread.start()
        return dict(job)

    def _update_download(self, download_id: str, **values: object) -> None:
        with self._lock:
            self._state["downloads"][download_id].update(values)
            self._save_state()

    @staticmethod
    def _artifact_download_bytes(temp_root: Path, artifact: ModelArtifact) -> int:
        """Return the largest visible partial file for an active artifact.

        huggingface_hub writes either the target file or a temporary
        ``.incomplete`` file below ``local_dir``.  Looking at both names lets
        the WebUI show real progress without depending on a downloader
        callback that the public API does not expose.
        """
        names = {Path(artifact.remote_path).name, Path(artifact.relative_path).name}
        candidates: list[Path] = []
        for relative in (artifact.remote_path, artifact.relative_path):
            target = temp_root / relative
            candidates.extend((target, target.with_name(f"{target.name}.incomplete")))
            candidates.extend(target.parent.glob(f"{target.name}.*.incomplete"))
        try:
            for path in temp_root.rglob("*"):
                if path.is_file() and any(
                    path.name == name or path.name.startswith(f"{name}.")
                    for name in names
                ):
                    candidates.append(path)
        except OSError:
            pass
        sizes: list[int] = []
        seen: set[str] = set()
        for path in candidates:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            try:
                if path.is_file():
                    sizes.append(path.stat().st_size)
            except OSError:
                continue
        return min(artifact.size_bytes, max(sizes, default=0))

    def _monitor_artifact_download(
        self,
        download_id: str,
        asset: ModelAsset,
        artifact: ModelArtifact,
        temp_root: Path,
        completed: int,
        stop: threading.Event,
    ) -> None:
        started = time.monotonic()
        name = Path(artifact.remote_path).name
        while not stop.wait(DOWNLOAD_PROGRESS_INTERVAL_SECONDS):
            try:
                partial = self._artifact_download_bytes(temp_root, artifact)
                elapsed = max(0.1, time.monotonic() - started)
                total = min(asset.size_bytes, completed + partial)
                speed = partial / elapsed if partial else 0.0
                remaining = max(0, asset.size_bytes - total)
                eta = remaining / speed if speed > 0 else None
                speed_label = f"{_format_bytes(int(speed))}/秒" if speed > 0 else "速度計測中"
                message = (
                    f"取得中: {name} · {_format_bytes(total)} / {_format_bytes(asset.size_bytes)}"
                    f" · {speed_label} · 経過 {_format_duration(elapsed)}"
                    f" · 残り {_format_duration(eta)}"
                )
                self._update_download(
                    download_id,
                    status="downloading",
                    bytes_downloaded=total,
                    progress=round(total * 100 / asset.size_bytes, 2),
                    download_speed_bytes_per_sec=round(speed, 2),
                    download_eta_seconds=round(eta, 1) if eta is not None else None,
                    download_elapsed_seconds=round(elapsed, 1),
                    message=message,
                )
            except Exception:
                # A disappearing temporary file or a service shutdown must
                # never interrupt the actual downloader.
                return

    @staticmethod
    def _replace_with_retry(source: Path, target: Path) -> None:
        last_error: OSError | None = None
        for delay in (0.0, 0.15, 0.5, 1.0):
            if delay:
                time.sleep(delay)
            try:
                os.replace(source, target)
                return
            except OSError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def _run_download(self, download_id: str, asset: ModelAsset) -> None:
        temp_root: Path | None = None
        try:
            self.prepare()
            missing: list[ModelArtifact] = []
            verified_files: dict[str, dict[str, object]] = {}
            self._update_download(
                download_id, status="verifying", message="既存ファイルを確認中"
            )
            for artifact in asset.artifacts:
                target = self._target(artifact.relative_path)
                if not target.exists():
                    missing.append(artifact)
                    continue
                if not target.is_file():
                    raise ModelSetupError(f"保存先がファイルではありません: {target}")
                if target.stat().st_size != artifact.size_bytes or _sha256(target) != artifact.sha256:
                    raise ModelSetupError(
                        f"既存ファイルが確認済み内容と一致しないため上書きしません: {target}"
                    )
                stat = target.stat()
                verified_files[artifact.relative_path] = {
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": artifact.sha256,
                }

            missing_bytes = sum(item.size_bytes for item in missing)
            if missing_bytes:
                free = shutil.disk_usage(self.models_root).free
                if free < missing_bytes + MIN_FREE_SPACE_MARGIN:
                    raise ModelSetupError(
                        f"空き容量が不足しています。必要: {_format_bytes(missing_bytes + MIN_FREE_SPACE_MARGIN)} / 空き: {_format_bytes(free)}"
                    )
                # Persist the exact cleanup target before creating it so a
                # process exit cannot leave an untracked large partial tree.
                temp_root = (
                    self.models_root / f".hayate-download-{download_id}"
                ).resolve(strict=False)
                self._update_download(download_id, temp_dir=str(temp_root))
                temp_root.mkdir(parents=False, exist_ok=False)
                completed = asset.size_bytes - missing_bytes
                for artifact in missing:
                    self._update_download(
                        download_id,
                        status="downloading",
                        bytes_downloaded=completed,
                        progress=round(completed * 100 / asset.size_bytes, 2),
                        message=f"取得中: {Path(artifact.remote_path).name}",
                    )
                    progress_stop = threading.Event()
                    progress_thread = threading.Thread(
                        target=self._monitor_artifact_download,
                        args=(
                            download_id,
                            asset,
                            artifact,
                            temp_root,
                            completed,
                            progress_stop,
                        ),
                        name=f"hayate-model-progress-{asset.id}",
                        daemon=True,
                    )
                    progress_thread.start()
                    try:
                        downloaded = self._downloader(asset, artifact, temp_root).resolve()
                    finally:
                        progress_stop.set()
                        progress_thread.join(timeout=2)
                    try:
                        downloaded.relative_to(temp_root)
                    except ValueError as exc:
                        raise ModelSetupError("ダウンローダーが一時フォルダ外を返しました") from exc
                    if downloaded.stat().st_size != artifact.size_bytes:
                        raise ModelSetupError(
                            f"ダウンロードサイズが一致しません: {artifact.remote_path}"
                        )
                    self._update_download(
                        download_id,
                        status="verifying",
                        bytes_downloaded=completed + artifact.size_bytes,
                        progress=round((completed + artifact.size_bytes) * 100 / asset.size_bytes, 2),
                        message=f"SHA256確認中: {Path(artifact.remote_path).name}",
                    )
                    if _sha256(downloaded) != artifact.sha256:
                        raise ModelSetupError(
                            f"SHA256が一致しません: {artifact.remote_path}"
                        )
                    target = self._target(artifact.relative_path)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        raise ModelSetupError(
                            f"確認中に保存先が作成されたため上書きしません: {target}"
                        )
                    self._replace_with_retry(downloaded, target)
                    stat = target.stat()
                    verified_files[artifact.relative_path] = {
                        "size_bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "sha256": artifact.sha256,
                    }
                    completed += artifact.size_bytes

            with self._lock:
                self._state["verified"][asset.id] = {
                    "revision": asset.revision,
                    "verified_at": _utc_now(),
                    "files": verified_files,
                }
            self._update_download(
                download_id,
                status="completed",
                bytes_downloaded=asset.size_bytes,
                progress=100.0,
                message="検証済みモデルを配置しました",
                finished_at=_utc_now(),
            )
        except Exception as exc:
            message = str(exc).strip() or type(exc).__name__
            self._update_download(
                download_id,
                status="failed",
                message=message[:600],
                finished_at=_utc_now(),
            )
        finally:
            if temp_root is not None:
                shutil.rmtree(temp_root, ignore_errors=True)
            with self._lock:
                self._active_assets.discard(asset.id)
                self._state["downloads"][download_id]["temp_dir"] = None
                self._save_state()
