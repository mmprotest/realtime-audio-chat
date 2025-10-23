from __future__ import annotations

import io
import sys
import wave
from types import ModuleType

import pytest

pytest.importorskip("fastapi")
np = pytest.importorskip("numpy")

from fastapi.testclient import TestClient

from src.servers.f5 import VoiceConfig, _SimpleF5Model, _default_model_loader, create_app


class StubF5Model:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def infer(self, ref_file: str, ref_text: str, gen_text: str, **_: object):
        self.calls.append((ref_file, ref_text, gen_text))
        sample_rate = 24_000
        t = np.linspace(0, 1, sample_rate, endpoint=False)
        wav = (0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        return wav, sample_rate, None


@pytest.fixture()
def voice_files(tmp_path):
    audio_path = tmp_path / "voice.wav"
    text_path = tmp_path / "voice.txt"
    with wave.open(audio_path.open("wb"), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(b"\x00\x00" * 24000)
    text_path.write_text("reference text", encoding="utf-8")
    return VoiceConfig(
        voice_id="demo",
        audio_path=str(audio_path),
        text_path=str(text_path),
        description="demo voice",
    )


def test_tts_streams_audio(tmp_path, voice_files):
    model = StubF5Model()

    def voice_loader():
        return {"demo": voice_files}

    app = create_app(model_loader=lambda: model, voice_loader=voice_loader, default_voice_id="demo")
    with TestClient(app) as client:
        response = client.post("/tts", json={"text": "Say something"})
        assert response.status_code == 200
        audio_bytes = b"".join(response.iter_bytes())

    assert len(audio_bytes) > 0
    assert response.headers["X-Sample-Rate"] == "24000"
    assert model.calls and model.calls[0][2] == "Say something"


def test_tts_rejects_unknown_voice(tmp_path, voice_files):
    model = StubF5Model()

    def voice_loader():
        return {"demo": voice_files}

    app = create_app(model_loader=lambda: model, voice_loader=voice_loader, default_voice_id="demo")
    with TestClient(app) as client:
        response = client.post("/tts", json={"text": "Hi", "voice_id": "missing"})
    assert response.status_code == 404


def test_tts_accepts_default_alias(tmp_path, voice_files):
    model = StubF5Model()

    def voice_loader():
        return {"demo": voice_files}

    app = create_app(model_loader=lambda: model, voice_loader=voice_loader, default_voice_id="demo")
    with TestClient(app) as client:
        response = client.post("/tts", json={"text": "Hi", "voice_id": "default"})
        audio_bytes = b"".join(response.iter_bytes())

    assert response.status_code == 200
    assert audio_bytes
    # The stub model should still be invoked with the actual demo voice assets.
    assert model.calls
    ref_file, _, gen_text = model.calls[0]
    assert ref_file == voice_files.audio_path
    assert gen_text == "Hi"


def test_voices_endpoint_lists_config(tmp_path, voice_files):
    model = StubF5Model()

    def voice_loader():
        return {"demo": voice_files}

    app = create_app(model_loader=lambda: model, voice_loader=voice_loader, default_voice_id="demo")
    with TestClient(app) as client:
        response = client.get("/voices")
    assert response.status_code == 200
    payload = response.json()
    assert payload["default"] == "demo"
    assert "demo" in payload["voices"]


def test_default_model_loader_falls_back_when_torchcodec_missing(monkeypatch):
    module = ModuleType("f5_tts")
    api_module = ModuleType("f5_tts.api")

    class DummyF5:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("No module named 'torchcodec'")

    api_module.F5TTS = DummyF5
    module.api = api_module
    monkeypatch.setitem(sys.modules, "f5_tts", module)
    monkeypatch.setitem(sys.modules, "f5_tts.api", api_module)
    monkeypatch.setattr("src.servers.f5._ensure_ffmpeg", lambda: None)

    model = _default_model_loader()
    assert isinstance(model, _SimpleF5Model)
