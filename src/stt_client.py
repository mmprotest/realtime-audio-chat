import io
import mimetypes
import os
from typing import Tuple

import requests

DEFAULT_STT_URL = os.getenv("STT_URL", "http://127.0.0.1:5007")


class RemoteSTT:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or DEFAULT_STT_URL).rstrip("/")
        self._check_health()

    @property
    def _health_endpoint(self) -> str:
        return f"{self.base_url}/health"

    @property
    def _transcribe_endpoint(self) -> str:
        return f"{self.base_url}/v1/transcribe"

    def _check_health(self) -> None:
        response = requests.get(self._health_endpoint, timeout=3)
        response.raise_for_status()

    def transcribe(self, sr_and_pcm: Tuple[int, "np.ndarray"] | bytes | io.BytesIO, ext: str = ".wav") -> str:
        """
        Accepts either:
          - (sample_rate, numpy_array) 16-bit or float32 mono audio, or
          - raw bytes/BytesIO of an audio file (wav/mp3/flac)
        Returns transcript text.
        """
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
            try:
                import soundfile as sf
            except Exception as exc:  # pragma: no cover - optional dependency guard
                raise RuntimeError(
                    "soundfile is required to encode audio buffers before sending them to the STT service."
                ) from exc
            buf = io.BytesIO()
            sf.write(buf, pcm, sr, format="WAV", subtype="PCM_16")
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

        response = requests.post(self._transcribe_endpoint, files=files, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("text", "")
