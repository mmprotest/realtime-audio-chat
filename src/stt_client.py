import io
import json
import mimetypes
import os
import wave
from typing import Tuple

import requests

DEFAULT_STT_URL = os.getenv("STT_URL", "http://127.0.0.1:5007")


class RemoteSTT:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or DEFAULT_STT_URL).rstrip("/")
        self._session = requests.Session()
        self._check_health()

    @property
    def _health_endpoint(self) -> str:
        return f"{self.base_url}/health"

    @property
    def _transcribe_endpoint(self) -> str:
        return f"{self.base_url}/v1/transcribe"

    def _check_health(self) -> None:
        response = self._session.get(self._health_endpoint, timeout=5)
        response.raise_for_status()

    def transcribe(self, sr_and_pcm: Tuple[int, "np.ndarray"] | bytes | io.BytesIO, ext: str = ".wav") -> str:
        """Send audio to the remote STT service and return the transcript text."""

        try:
            import numpy as np  # type: ignore
        except Exception:  # pragma: no cover - optional dependency guard
            np = None

        files = None
        if isinstance(sr_and_pcm, tuple):
            if np is None:
                raise RuntimeError(
                    "NumPy is required to send in-memory audio to the STT service. Install numpy<2 in the app venv."
                )
            sr, pcm = sr_and_pcm
            if not isinstance(pcm, np.ndarray):
                raise TypeError("Audio tuple must contain a NumPy array")
            pcm = pcm.squeeze()
            if pcm.ndim != 1:
                raise ValueError("RemoteSTT only supports mono audio arrays")
            if pcm.dtype != np.int16:
                pcm = np.clip(pcm, -1.0, 1.0)
                pcm = (pcm * 32767).astype("<i2")
            else:
                pcm = pcm.astype("<i2", copy=False)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(int(sr))
                wav_file.writeframes(pcm.tobytes())
            buf.seek(0)
            files = {"file": ("audio.wav", buf, "audio/wav")}
        else:
            if isinstance(sr_and_pcm, (bytes, bytearray)):
                buf = io.BytesIO(sr_and_pcm)
            elif isinstance(sr_and_pcm, io.BytesIO):
                buf = sr_and_pcm
            else:
                raise TypeError("Unsupported audio input for RemoteSTT.transcribe")
            buf.seek(0)
            mime = mimetypes.guess_type(f"dummy{ext}")[0] or "application/octet-stream"
            files = {"file": (f"audio{ext}", buf, mime)}

        response = self._session.post(self._transcribe_endpoint, files=files, timeout=90)
        response.raise_for_status()
        try:
            data = response.json()
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
            raise RuntimeError("STT service returned a non-JSON response") from exc
        return data.get("text", "")
