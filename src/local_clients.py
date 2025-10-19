"""HTTP clients that integrate local Whisper STT and F5-TTS services."""
from __future__ import annotations

import base64
import io
import json
import urllib.request
import wave
from typing import Callable, Iterable

import numpy as np
from numpy.typing import NDArray

AudioArray = NDArray[np.float32 | np.int16]

WhisperRequester = Callable[[str, bytes, dict[str, str], float], dict[str, object]]
TTSStreamer = Callable[[str, bytes, dict[str, str], float], Iterable[bytes]]


class WhisperSTTClient:
    """Client for a local Whisper transcription API."""

    def __init__(
        self,
        *,
        base_url: str,
        language: str | None = None,
        timeout: float = 30.0,
        requester: WhisperRequester | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._language = language
        self._timeout = timeout
        self._requester = requester or _default_whisper_requester

    def close(self) -> None:  # pragma: no cover - retained for API parity
        """Included for parity with external clients; no persistent resources."""

    def stt(self, audio: tuple[int, AudioArray]) -> str:
        sample_rate, samples = audio
        wav_bytes = _audio_to_wav_bytes(samples, sample_rate)
        payload: dict[str, object] = {
            "audio": base64.b64encode(wav_bytes).decode("ascii"),
            "sample_rate": sample_rate,
        }
        if self._language:
            payload["language"] = self._language

        url = f"{self._base_url}/transcribe"
        headers = {"Content-Type": "application/json"}
        response = self._requester(url, json.dumps(payload).encode("utf-8"), headers, self._timeout)
        text = response.get("text")
        if not isinstance(text, str):  # pragma: no cover - defensive guard
            raise ValueError("Whisper server response missing 'text' field")
        return text


class F5LocalTTS:
    """Client for a local F5-TTS streaming server."""

    def __init__(
        self,
        *,
        base_url: str,
        voice_id: str,
        output_format: str = "pcm_s16le",
        timeout: float = 30.0,
        streamer: TTSStreamer | None = None,
    ) -> None:
        self.voice_id = voice_id
        self.output_format = output_format
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self.text_to_speech = _F5TextToSpeech(self, streamer or _default_tts_streamer)

    def close(self) -> None:  # pragma: no cover - retained for API parity
        """Included for parity with external clients; no persistent resources."""


class _F5TextToSpeech:
    def __init__(self, parent: F5LocalTTS, streamer: TTSStreamer) -> None:
        self._parent = parent
        self._streamer = streamer

    def convert_as_stream(
        self,
        *,
        text: str,
        voice_id: str | None = None,
        output_format: str | None = None,
    ) -> Iterable[bytes]:
        payload = {
            "text": text,
            "voice_id": voice_id or self._parent.voice_id,
            "output_format": output_format or self._parent.output_format,
        }
        url = f"{self._parent._base_url}/tts"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/octet-stream",
        }
        body = json.dumps(payload).encode("utf-8")
        yield from self._streamer(url, body, headers, self._parent._timeout)


def _default_whisper_requester(
    url: str, body: bytes, headers: dict[str, str], timeout: float
) -> dict[str, object]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # type: ignore[arg-type]
        charset = response.headers.get_content_charset() or "utf-8"
        payload = response.read().decode(charset)
    return json.loads(payload)


def _default_tts_streamer(
    url: str, body: bytes, headers: dict[str, str], timeout: float
) -> Iterable[bytes]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # type: ignore[arg-type]
        while True:
            chunk = response.read(8192)
            if not chunk:
                break
            yield chunk


def _audio_to_wav_bytes(samples: AudioArray, sample_rate: int) -> bytes:
    mono = _ensure_mono(samples)
    pcm16 = _to_pcm16(mono)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    return buffer.getvalue()


def _ensure_mono(samples: AudioArray) -> np.ndarray:
    array = np.asarray(samples)
    if array.ndim == 1:
        return array
    if array.ndim > 2:
        raise ValueError("Audio arrays with more than 2 dimensions are not supported")
    return array.mean(axis=-1)


def _to_pcm16(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples)
    if array.dtype == np.int16:
        return array
    if np.issubdtype(array.dtype, np.floating):
        clipped = np.clip(array, -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16)
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        normalized = array.astype(np.float32) / float(info.max)
        return (np.clip(normalized, -1.0, 1.0) * 32767.0).astype(np.int16)
    raise TypeError(f"Unsupported audio dtype: {array.dtype}")
