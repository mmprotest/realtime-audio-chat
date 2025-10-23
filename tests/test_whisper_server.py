from __future__ import annotations

import base64
import io
import wave

import pytest

pytest.importorskip("fastapi")
np = pytest.importorskip("numpy")

from fastapi.testclient import TestClient

from types import SimpleNamespace

from src.servers.whisper import create_app, _default_transcriber_factory


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.astype(np.int16).tobytes())
    return buffer.getvalue()


def test_transcribe_resamples_and_returns_text():
    captured: dict[str, object] = {}

    def factory(model: object):
        def _transcribe(audio: np.ndarray, language: str | None) -> str:
            captured["length"] = len(audio)
            captured["language"] = language
            return "hello world"

        return _transcribe

    app = create_app(model_loader=lambda: object(), transcriber_factory=factory)
    audio = (np.sin(np.linspace(0, 1, 8000) * 2 * np.pi * 220) * 0.5 * 32767).astype(np.int16)
    payload = {
        "audio": base64.b64encode(_wav_bytes(audio, 8000)).decode("ascii"),
        "sample_rate": 8000,
        "language": "en",
    }
    with TestClient(app) as client:
        response = client.post("/transcribe", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "hello world"
    assert body["duration"] > 0
    assert captured["length"] == 16000  # default target sample rate
    assert captured["language"] == "en"


def test_transcribe_rejects_invalid_base64():
    app = create_app(model_loader=lambda: object(), transcriber_factory=lambda model: lambda a, b: "")
    with TestClient(app) as client:
        response = client.post("/transcribe", json={"audio": "@@@"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid base64 audio"


def test_transcribe_rejects_empty_audio():
    app = create_app(model_loader=lambda: object(), transcriber_factory=lambda model: lambda a, b: "")
    silence = np.zeros(0, dtype=np.int16)
    payload = {
        "audio": base64.b64encode(_wav_bytes(silence, 16000)).decode("ascii"),
    }
    with TestClient(app) as client:
        response = client.post("/transcribe", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == "Audio payload contains no data"


def test_transcribe_falls_back_to_default_language(monkeypatch):
    monkeypatch.setenv("WHISPER_FALLBACK_LANGUAGE", "en")
    calls: list[str | None] = []

    def factory(model: object):
        def _transcribe(audio: np.ndarray, language: str | None) -> str:
            calls.append(language)
            if language is None:
                raise ValueError("max() arg is an empty sequence")
            return "fallback succeeded"

        return _transcribe

    app = create_app(model_loader=lambda: object(), transcriber_factory=factory)
    audio = (np.sin(np.linspace(0, 1, 16000) * 2 * np.pi * 220) * 0.5 * 32767).astype(np.int16)
    payload = {"audio": base64.b64encode(_wav_bytes(audio, 16000)).decode("ascii")}

    with TestClient(app) as client:
        response = client.post("/transcribe", json=payload)

    assert response.status_code == 200
    assert response.json()["text"] == "fallback succeeded"
    assert calls == [None, "en"]


def test_transcribe_surfaces_language_detection_failure(monkeypatch):
    monkeypatch.setenv("WHISPER_FALLBACK_LANGUAGE", "en")

    def factory(model: object):
        def _transcribe(audio: np.ndarray, language: str | None) -> str:
            raise ValueError("max() arg is an empty sequence")

        return _transcribe

    app = create_app(model_loader=lambda: object(), transcriber_factory=factory)
    audio = (np.sin(np.linspace(0, 1, 16000) * 2 * np.pi * 220) * 0.5 * 32767).astype(np.int16)
    payload = {"audio": base64.b64encode(_wav_bytes(audio, 16000)).decode("ascii")}

    with TestClient(app) as client:
        response = client.post("/transcribe", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"] == "Language detection failed; specify the language parameter"


def test_transcribe_rejects_empty_transcription():
    def factory(model: object):
        def _transcribe(audio: np.ndarray, language: str | None) -> str:
            return "  "

        return _transcribe

    app = create_app(model_loader=lambda: object(), transcriber_factory=factory)
    audio = (np.sin(np.linspace(0, 1, 16000) * 2 * np.pi * 220) * 0.5 * 32767).astype(np.int16)
    payload = {"audio": base64.b64encode(_wav_bytes(audio, 16000)).decode("ascii")}

    with TestClient(app) as client:
        response = client.post("/transcribe", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"] == "Transcription produced no text"


def test_transcribe_retries_with_fallback_when_blank(monkeypatch):
    monkeypatch.setenv("WHISPER_FALLBACK_LANGUAGE", "en")
    calls: list[str | None] = []

    def factory(model: object):
        def _transcribe(audio: np.ndarray, language: str | None) -> str:
            calls.append(language)
            if language is None:
                return "   "
            if language == "en":
                return "hello world"
            return ""

        return _transcribe

    app = create_app(model_loader=lambda: object(), transcriber_factory=factory)
    audio = (np.sin(np.linspace(0, 1, 16000) * 2 * np.pi * 220) * 0.5 * 32767).astype(np.int16)
    payload = {"audio": base64.b64encode(_wav_bytes(audio, 16000)).decode("ascii")}

    with TestClient(app) as client:
        response = client.post("/transcribe", json=payload)

    assert response.status_code == 200
    assert response.json()["text"] == "hello world"
    assert calls == [None, "en"]


def test_transcribe_treats_blank_language_as_missing():
    observed: list[str | None] = []

    def factory(model: object):
        def _transcribe(audio: np.ndarray, language: str | None) -> str:
            observed.append(language)
            return "ok"

        return _transcribe

    app = create_app(model_loader=lambda: object(), transcriber_factory=factory)
    audio = (np.sin(np.linspace(0, 1, 16000) * 2 * np.pi * 220) * 0.5 * 32767).astype(np.int16)
    payload = {
        "audio": base64.b64encode(_wav_bytes(audio, 16000)).decode("ascii"),
        "language": "  ",
    }

    with TestClient(app) as client:
        response = client.post("/transcribe", json=payload)

    assert response.status_code == 200
    assert response.json()["text"] == "ok"
    assert observed == [None]


def test_default_transcriber_factory_normalises_segments(monkeypatch):
    faster_whisper = pytest.importorskip("faster_whisper")

    class DummyModel:
        def transcribe(self, audio: np.ndarray, language: str | None = None):
            return iter(
                [
                    SimpleNamespace(text="▁Hello"),
                    SimpleNamespace(text="  world  "),
                    SimpleNamespace(text=""),
                ]
            ), None

    monkeypatch.setattr(faster_whisper, "WhisperModel", DummyModel)
    transcriber = _default_transcriber_factory(DummyModel())
    result = transcriber(np.zeros(10, dtype=np.float32), "en")
    assert result == "Hello world"
