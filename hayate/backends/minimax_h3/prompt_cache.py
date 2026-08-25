from __future__ import annotations

import copy
import hashlib
import json
import os
import struct
import uuid
from pathlib import Path
from typing import Any, Callable


CACHE_SCHEMA_VERSION = 1
_MAX_HEADER_BYTES = 128 * 1024 * 1024
_SAMPLE_BYTES = 1024 * 1024


def _sampled_file_fingerprint(value: str | os.PathLike[str] | None) -> dict[str, Any] | None:
    """Return a bounded content/stat fingerprint without reading a multi-GB payload."""
    if not value:
        return None
    path = Path(value).expanduser().resolve(strict=False)
    if not path.is_file():
        return {"path": str(path), "missing": True}

    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", stat.st_size))
    with path.open("rb") as handle:
        prefix = handle.read(8)
        digest.update(prefix)
        header_size = struct.unpack("<Q", prefix)[0] if len(prefix) == 8 else 0
        if 2 <= header_size <= _MAX_HEADER_BYTES and 8 + header_size <= stat.st_size:
            digest.update(handle.read(header_size))
        else:
            digest.update(handle.read(_SAMPLE_BYTES))
        if stat.st_size > _SAMPLE_BYTES:
            handle.seek(max(0, stat.st_size - _SAMPLE_BYTES))
            digest.update(handle.read(_SAMPLE_BYTES))

    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sampled_sha256": digest.hexdigest(),
    }


def _input_fingerprints(args: Any) -> dict[str, Any]:
    return {
        "image": _sampled_file_fingerprint(getattr(args, "image_path", None)),
        "last_image": _sampled_file_fingerprint(getattr(args, "last_image_path", None)),
        "references": [
            _sampled_file_fingerprint(path) for path in (getattr(args, "reference", None) or [])
        ],
    }


def _hayate_metadata(args: Any, upstream_commit: str) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "upstream_commit": upstream_commit,
        "text_encoder": _sampled_file_fingerprint(getattr(args, "text_encoder", None)),
        "inputs": _input_fingerprints(args),
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def install_prompt_cache_override(module: Any, *, upstream_commit: str) -> None:
    """Add HAYATE fingerprints and crash-safe writes around upstream's prompt cache."""
    original_key: Callable[..., str] = module._prompt_cache_key
    original_encode: Callable[..., Any] = module.encode_prompt_stage

    def prompt_cache_key(args: Any, task: str, plan: Any, prompt: str) -> str:
        payload = {
            "upstream_key": original_key(args, task, plan, prompt),
            "hayate": _hayate_metadata(args, upstream_commit),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:32]

    def encode_prompt_stage(args: Any, task: str, plan: Any, prompt: str, device: Any):
        cache_value = getattr(args, "prompt_cache", None)
        if not cache_value:
            return original_encode(args, task, plan, prompt, device)

        target = Path(cache_value).expanduser().resolve(strict=False)
        sidecar = Path(str(target) + ".json")
        expected_key = prompt_cache_key(args, task, plan, prompt)
        expected_metadata = _hayate_metadata(args, upstream_commit)
        existing = _read_json(sidecar)
        if (
            target.is_file()
            and existing is not None
            and existing.get("key") == expected_key
            and existing.get("hayate") == expected_metadata
        ):
            module.logger.info("HAYATE prompt cache fingerprint validated: %s", target)
            return original_encode(args, task, plan, prompt, device)

        target.parent.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        temporary = target.with_name(f".{target.name}.{token}.tmp")
        temporary_sidecar = Path(str(temporary) + ".json")
        temporary_args = copy.copy(args)
        temporary_args.prompt_cache = str(temporary)
        module.logger.info("HAYATE prompt cache miss; rebuilding atomically: %s", target)
        try:
            result = original_encode(temporary_args, task, plan, prompt, device)
            metadata = _read_json(temporary_sidecar)
            if not temporary.is_file() or metadata is None:
                raise RuntimeError("upstream prompt cache did not produce both cache files")
            metadata["key"] = expected_key
            metadata["hayate"] = expected_metadata
            _write_json_atomic(temporary_sidecar, metadata)
            # Data first is deliberate: if the second replace is interrupted, the old
            # sidecar key causes a safe miss instead of accepting mismatched embeddings.
            os.replace(temporary, target)
            os.replace(temporary_sidecar, sidecar)
            return result
        finally:
            temporary.unlink(missing_ok=True)
            temporary_sidecar.unlink(missing_ok=True)

    module._prompt_cache_key = prompt_cache_key
    module.encode_prompt_stage = encode_prompt_stage

