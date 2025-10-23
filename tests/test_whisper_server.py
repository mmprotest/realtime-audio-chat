from __future__ import annotations

import base64
import io
import wave

import pytest

pytest.importorskip("fastapi")
np = pytest.importorskip("numpy")

from fastapi.testclient import TestClient

from src.servers.whisper import create_app


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
