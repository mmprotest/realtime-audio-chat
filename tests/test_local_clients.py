from __future__ import annotations

import base64
import json
from typing import Iterable

import pytest

np = pytest.importorskip("numpy")

from src.local_clients import F5LocalTTS, WhisperSTTClient, _ensure_mono


def test_whisper_client_posts_audio():
    captured = {}

    def requester(url: str, body: bytes, headers: dict[str, str], timeout: float):
        captured["url"] = url
        captured["headers"] = headers
        payload = json.loads(body.decode("utf-8"))
        captured["payload"] = payload
        return {"text": "hello"}

    client = WhisperSTTClient(base_url="http://stt.local", requester=requester)
    text = client.stt((16000, np.zeros(1600, dtype=np.float32)))
    client.close()

    assert text == "hello"
    assert captured["url"] == "http://stt.local/transcribe"
    assert captured["headers"]["Content-Type"] == "application/json"
    audio_b64 = captured["payload"]["audio"]
    assert isinstance(audio_b64, str)
    decoded = base64.b64decode(audio_b64.encode("ascii"))
    assert decoded.startswith(b"RIFF")  # wav header


def test_f5_tts_streams_bytes():
    chunks = [b"\x01\x00\x02\x00", b"\x03\x00"]

    def streamer(url: str, body: bytes, headers: dict[str, str], timeout: float) -> Iterable[bytes]:
        payload = json.loads(body.decode("utf-8"))
        assert payload["voice_id"] == "test-voice"
        assert url == "http://tts.local/tts"
        assert headers["Accept"] == "application/octet-stream"
        for chunk in chunks:
            yield chunk

    client = F5LocalTTS(
        base_url="http://tts.local",
        voice_id="test-voice",
        output_format="pcm_s16le",
        streamer=streamer,
    )
    try:
        streamed = list(client.text_to_speech.convert_as_stream(text="hi"))
    finally:
        client.close()

    assert streamed == chunks


def test_whisper_client_validates_response():
    client = WhisperSTTClient(base_url="http://stt.local", requester=lambda *a, **k: {})
    with pytest.raises(ValueError):
        client.stt((16000, np.zeros(1600, dtype=np.float32)))
    client.close()


def test_ensure_mono_preserves_singleton_channel_last():
    samples = np.arange(1600, dtype=np.float32).reshape(-1, 1)
    mono = _ensure_mono(samples)
    assert mono.shape == (1600,)
    assert mono[0] == pytest.approx(samples[0, 0])


def test_ensure_mono_preserves_singleton_channel_first():
    samples = np.arange(1600, dtype=np.float32).reshape(1, -1)
    mono = _ensure_mono(samples)
    assert mono.shape == (1600,)
    assert mono[-1] == pytest.approx(samples[0, -1])
