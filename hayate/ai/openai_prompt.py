"""OpenAI-backed MiniMax H3 prompt authoring.

This module only creates text.  It does not call the H3 generation process or
send a user's API key to that process.  The Responses API Structured Outputs
path keeps the result predictable enough for the WebUI to preview and apply
only the validated ``final_prompt`` field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict


class H3PromptResult(BaseModel):
    """The reviewed prompt parts returned by the authoring model."""

    model_config = ConfigDict(extra="forbid")

    subject: str
    action: str
    environment: str
    camera: str
    lighting: str
    style: str
    soundscape: str
    music: str
    negative: str
    final_prompt: str


@dataclass(frozen=True)
class PromptRequest:
    brief: str
    task: str = "auto"
    duration_seconds: float = 5.0
    width: int = 512
    height: int = 512
    include_audio: bool = True
    language: str = "ja"
    current_prompt: str = ""

    def as_input(self) -> str:
        current = self.current_prompt.strip()
        current_section = (
            f"\nCurrent H3 prompt to improve (preserve intent unless it conflicts):\n{current}\n"
            if current
            else ""
        )
        return (
            "Create a MiniMax H3 video prompt from this user brief.\n"
            f"User brief: {self.brief.strip()}\n"
            f"Task: {self.task}\n"
            f"Duration: {self.duration_seconds:g} seconds at 24 fps\n"
            f"Canvas: {self.width}x{self.height}\n"
            f"Audio requested: {'yes' if self.include_audio else 'no'}\n"
            f"UI language: {self.language}\n"
            f"{current_section}"
        )


H3_PROMPT_INSTRUCTIONS = """You are HAYATE's MiniMax H3 prompt director.
Turn the user's brief into a precise, production-ready prompt for the existing
MiniMax H3 engine. Return only the requested structured fields; do not add
metadata or discuss your reasoning.

The final_prompt must be in clear English because it is passed to the H3 text
encoder. It must be self-contained and temporally coherent. Include:
- an integrated_multimodal_description with [Shot 1], [Shot 2], ... where useful;
- consistent subject identity, action, environment, camera framing/lens/movement,
  composition, lighting, materials, motion continuity, and cinematic style;
- separate overall_soundscape and non_diegetic_music when audio is requested;
- the exact duration and requested canvas only as helpful generation context;
- a natural-language negative_prompt clause inside final_prompt covering flicker,
  temporal identity drift, deformed anatomy/objects, blur, compression artifacts,
  text, logos, watermarks, and unwanted cuts.

Do not invent dialogue, lyrics, brand names, or on-screen text unless the brief
explicitly asks for them. If audio is not requested, set soundscape and music to
"N/A" rather than inventing them. The negative field is a concise review aid;
final_prompt remains the only field the UI will apply. Keep every field concrete
and useful, avoid vague adjectives, and keep final_prompt below 6000 characters."""


class PromptAssistantError(RuntimeError):
    """Safe, user-facing classification for an OpenAI request failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class OpenAIClientConfig:
    model: str
    api_key: str


def _classify_openai_error(exc: Exception) -> PromptAssistantError:
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__.lower()
    if status in {401, 403} or "authentication" in name:
        return PromptAssistantError(
            "openai_authentication", "OpenAIの認証に失敗しました。設定画面のAPIキーを確認してください。"
        )
    if status == 429 or "rate" in name:
        return PromptAssistantError(
            "openai_rate_limit", "OpenAIの利用上限またはレート制限に達しました。少し待って再試行してください。"
        )
    if status == 404 or "notfound" in name:
        return PromptAssistantError(
            "openai_model_unavailable", "指定したOpenAIモデルが利用できないか、権限がありません。モデルIDを確認してください。"
        )
    if "timeout" in name or "connection" in name:
        return PromptAssistantError(
            "openai_connection", "OpenAIへ接続できませんでした。ネットワークとAPIエンドポイントを確認してください。"
        )
    return PromptAssistantError(
        "openai_request", "OpenAIへのプロンプト作成要求に失敗しました。設定とモデルを確認してください。"
    )


class MiniMaxH3PromptAssistant:
    def __init__(
        self,
        config: OpenAIClientConfig,
        *,
        client_factory: Any | None = None,
    ):
        self.config = config
        self._client_factory = client_factory

    def _client(self):
        factory = self._client_factory
        if factory is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise PromptAssistantError(
                    "openai_sdk_missing",
                    "OpenAI SDKが未インストールです。HAYATEのWebUI依存関係を更新してください。",
                ) from exc
            factory = OpenAI
        kwargs: dict[str, Any] = {
            "api_key": self.config.api_key,
            "timeout": 60.0,
            "max_retries": 1,
        }
        return factory(**kwargs)

    def generate(self, request: PromptRequest) -> H3PromptResult:
        if not self.config.api_key.strip():
            raise PromptAssistantError(
                "openai_key_missing", "設定画面でOpenAI APIキーを登録してください。"
            )
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "instructions": H3_PROMPT_INSTRUCTIONS,
            "input": request.as_input(),
            "text_format": H3PromptResult,
            # Leave enough room for the ten structured fields and a detailed
            # temporal prompt.  The model is still instructed to keep the
            # final prompt below 6000 characters.
            "max_output_tokens": 4096,
            "store": False,
        }
        if self.config.model.lower().startswith("gpt-5"):
            kwargs["reasoning"] = {"effort": "low"}
        client = None
        try:
            client = self._client()
            response = client.responses.parse(**kwargs)
        except PromptAssistantError:
            raise
        except Exception as exc:
            raise _classify_openai_error(exc) from exc
        finally:
            close = getattr(client, "close", None) if client is not None else None
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

        status = getattr(response, "status", "completed")
        if status != "completed":
            detail = getattr(getattr(response, "incomplete_details", None), "reason", None)
            if detail:
                raise PromptAssistantError(
                    "openai_incomplete", "OpenAIの応答が途中で終了しました。もう一度試してください。"
                )
            raise PromptAssistantError(
                "openai_refused", "OpenAIがこのプロンプト作成要求を完了できませんでした。内容を調整してください。"
            )
        result = getattr(response, "output_parsed", None)
        if result is None:
            for item in getattr(response, "output", ()) or ():
                for content in getattr(item, "content", ()) or ():
                    if getattr(content, "type", "") == "refusal":
                        raise PromptAssistantError(
                            "openai_refused",
                            "OpenAIがこのプロンプト作成要求を拒否しました。内容を調整してください。",
                        )
            raise PromptAssistantError(
                "openai_structured_output", "OpenAIから有効なH3プロンプトを受け取れませんでした。もう一度試してください。"
            )
        try:
            parsed = result if isinstance(result, H3PromptResult) else H3PromptResult.model_validate(result)
        except Exception as exc:
            raise PromptAssistantError(
                "openai_structured_output", "OpenAIの構造化出力を検証できませんでした。もう一度試してください。"
            ) from exc
        for field_name in H3PromptResult.model_fields:
            value = getattr(parsed, field_name).strip()
            if not value or len(value) > 6000:
                raise PromptAssistantError(
                    "openai_structured_output", "生成されたH3プロンプトの内容が不正です。もう一度試してください。"
                )
        return parsed
