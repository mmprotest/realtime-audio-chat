"""Environment-backed configuration for the realtime audio chat app."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    """Configuration values loaded from environment variables."""

    openai_model: str
    openai_max_tokens: int
    openai_base_url: str | None
    openai_api_key: str | None
    f5_tts_url: str
    f5_tts_voice: str
    f5_tts_output_format: str
    whisper_url: str
    whisper_language: str | None
    http_timeout: float
    tts_sample_rate: int
    input_sample_rate: int


def _get_env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


def _get_optional_env(name: str) -> str | None:
    value = os.getenv(name)
    return value if value else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load :class:`Settings` from the current process environment."""

    openai_max_tokens = int(_get_env("OPENAI_MAX_TOKENS", "512"))
    http_timeout = float(_get_env("HTTP_TIMEOUT", "30"))
    tts_sample_rate = int(_get_env("OUTPUT_SAMPLE_RATE", "24000"))
    input_sample_rate = int(_get_env("INPUT_SAMPLE_RATE", "16000"))

    return Settings(
        openai_model=_get_env("OPENAI_MODEL", "gpt-4o-mini"),
        openai_max_tokens=openai_max_tokens,
        openai_base_url=_get_optional_env("OPENAI_BASE_URL"),
        openai_api_key=_get_optional_env("OPENAI_API_KEY"),
        f5_tts_url=_get_env("F5_TTS_URL", "http://localhost:9880"),
        f5_tts_voice=_get_env("F5_TTS_VOICE", "default"),
        f5_tts_output_format=_get_env("F5_TTS_OUTPUT_FORMAT", "pcm_s16le"),
        whisper_url=_get_env("WHISPER_URL", "http://localhost:9000"),
        whisper_language=os.getenv("WHISPER_LANGUAGE"),
        http_timeout=http_timeout,
        tts_sample_rate=tts_sample_rate,
        input_sample_rate=input_sample_rate,
    )
