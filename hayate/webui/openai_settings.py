"""Local OpenAI configuration for the HAYATE WebUI.

The model is an ordinary application setting. Requests use the official
OpenAI endpoint; custom endpoints are intentionally not supported because a
stored key must never be redirected to an arbitrary host. The API key is
deliberately kept out of settings.json, SQLite, job manifests, and
HTTP responses.  On Windows it is stored in Credential Manager through
``keyring`` when that package/backend is available; otherwise a key entered
from the UI is retained only for the current server process.  An
``OPENAI_API_KEY`` environment variable remains a supported fallback.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

DEFAULT_OPENAI_MODEL = "gpt-5.6-terra"
KEYRING_SERVICE = "HAYATE"
KEYRING_USERNAME = "openai-api-key"
_UNSET = object()
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def validate_model(value: str) -> str:
    model = str(value or "").strip()
    if not model or not _MODEL_RE.fullmatch(model):
        raise ValueError(
            "OpenAIモデルIDは英数字・ハイフン・アンダースコア・ドット・コロンで指定してください"
        )
    return model


@dataclass(frozen=True)
class OpenAISettings:
    model: str = DEFAULT_OPENAI_MODEL

    def validated(self) -> "OpenAISettings":
        return OpenAISettings(model=validate_model(self.model))


class OpenAISettingsStore:
    """Persist non-secret OpenAI settings and resolve a secret at request time."""

    def __init__(self, path: Path, *, secret_backend: object = _UNSET):
        self.path = path.resolve(strict=False)
        self._lock = threading.RLock()
        self._session_key: str | None = None
        # The injectable backend keeps tests deterministic without touching a
        # developer's real Credential Manager.
        self._secret_backend = secret_backend

    def _backend(self):
        if self._secret_backend is not _UNSET:
            return self._secret_backend
        try:
            import keyring  # type: ignore

            return keyring
        except Exception:
            return None

    def load(self) -> OpenAISettings:
        with self._lock:
            defaults = OpenAISettings()
            if not self.path.is_file():
                return defaults
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                return defaults
            if not isinstance(payload, dict):
                return defaults
            try:
                return OpenAISettings(
                    model=str(payload.get("model", defaults.model)),
                ).validated()
            except ValueError:
                return defaults

    def save(self, settings: OpenAISettings) -> OpenAISettings:
        validated = settings.validated()
        payload = {"model": validated.model}
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary.replace(self.path)
        return validated

    def update(self, *, model: str | None = None) -> OpenAISettings:
        current = self.load()
        return self.save(
            OpenAISettings(
                model=current.model if model is None else model,
            )
        )

    def _read_keyring(self) -> str | None:
        backend = self._backend()
        if backend is None:
            return None
        try:
            value = backend.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
        except Exception:
            return None
        return value.strip() if isinstance(value, str) and value.strip() else None

    def _write_keyring(self, value: str) -> bool:
        backend = self._backend()
        if backend is None:
            return False
        try:
            backend.set_password(KEYRING_SERVICE, KEYRING_USERNAME, value)
            return True
        except Exception:
            # Do not fall back to a plaintext file.  The caller can retain the
            # value for this process only and the UI reports that source.
            return False

    def _delete_keyring(self) -> bool:
        """Delete and verify the Credential Manager entry.

        A missing entry is already the desired end state.  A backend error or
        a value that remains after deletion is reported to the caller so the
        WebUI cannot show a false "cleared" status.
        """

        backend = self._backend()
        if backend is None:
            return True
        try:
            existing = backend.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
        except Exception:
            return False
        if not existing:
            return True
        try:
            backend.delete_password(KEYRING_SERVICE, KEYRING_USERNAME)
            remaining = backend.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
        except Exception:
            return False
        return not remaining

    def set_api_key(self, value: str) -> str:
        key = str(value or "").strip()
        if not key:
            raise ValueError("APIキーが空です")
        with self._lock:
            if self._write_keyring(key):
                # Credential Manager is the durable source; do not retain a
                # second plaintext copy in the WebUI process after saving.
                self._session_key = None
                return "keyring"
            self._session_key = key
            return "session"

    def clear_api_key(self) -> bool:
        with self._lock:
            self._session_key = None
            return self._delete_keyring()

    def credentials(self) -> tuple[str | None, str]:
        """Return ``(key, source)`` without ever serializing the key."""

        with self._lock:
            if self._session_key:
                return self._session_key, "session"
            keyring_key = self._read_keyring()
            if keyring_key:
                return keyring_key, "keyring"
            environment_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if environment_key:
                return environment_key, "environment"
            return None, "none"

    def public_dict(self) -> dict[str, object]:
        settings = self.load()
        key, source = self.credentials()
        return {
            "model": settings.model,
            "api_key_configured": bool(key),
            "api_key_source": source,
        }
