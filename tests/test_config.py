from __future__ import annotations

import os

import pytest

from src.config import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.delenv("F5_TTS_URL", raising=False)
    monkeypatch.delenv("WHISPER_URL", raising=False)

    settings = get_settings()
    assert settings.groq_model == "llama-3.1-8b-instant"
    assert settings.f5_tts_url == "http://localhost:9880"
    assert settings.whisper_url == "http://localhost:9000"
    assert settings.tts_sample_rate == 24000
    assert settings.input_sample_rate == 16000


def test_settings_respect_environment(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "llama-custom")
    monkeypatch.setenv("F5_TTS_URL", "http://tts.local")
    monkeypatch.setenv("WHISPER_URL", "http://stt.local")
    monkeypatch.setenv("OUTPUT_SAMPLE_RATE", "44100")
    monkeypatch.setenv("INPUT_SAMPLE_RATE", "8000")
    monkeypatch.setenv("GROQ_MAX_TOKENS", "256")
    monkeypatch.setenv("HTTP_TIMEOUT", "12.5")

    settings = get_settings()
    assert settings.groq_model == "llama-custom"
    assert settings.f5_tts_url == "http://tts.local"
    assert settings.whisper_url == "http://stt.local"
    assert settings.tts_sample_rate == 44100
    assert settings.input_sample_rate == 8000
    assert settings.groq_max_tokens == 256
    assert settings.http_timeout == 12.5
