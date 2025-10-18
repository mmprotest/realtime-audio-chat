"""F5-TTS wrapper for voice cloning and streaming."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO
from typing import Generator, Optional

import numpy as np
import soundfile as sf

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class VoiceProfile:
    """Stores reference audio and style instructions."""

    speaker_wav: Optional[np.ndarray] = None
    speaker_sr: Optional[int] = None
    style_notes: str = ""


class F5Cloner:
    """Lazy-loading helper for F5-TTS voice cloning."""

    def __init__(self, *, device: str = "cpu") -> None:
        self.device = device
        self._pipe = None
        LOGGER.info("F5Cloner initialized | device=%s", device)

    def _load_pipe(self):  # pragma: no cover - small wrapper
        if self._pipe is None:
            try:
                from f5_tts import F5TTS
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("f5-tts is not installed. Please install dependencies.") from exc
            self._pipe = F5TTS(device=self.device)
            LOGGER.info("Loaded F5-TTS pipeline")
        return self._pipe

    def set_voice(self, wav: np.ndarray, sr: int) -> VoiceProfile:
        """Prepare and store a normalized mono reference voice."""

        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)
        wav = wav.astype(np.float32, copy=False)
        peak = np.max(np.abs(wav))
        if peak > 0:
            wav = wav * (0.95 / peak)
        profile = VoiceProfile(speaker_wav=wav, speaker_sr=sr)
        LOGGER.info("Voice profile updated | sr=%s duration=%.2fs", sr, len(wav) / float(sr))
        return profile

    def stream_tts_sync(
        self,
        text: str,
        profile: VoiceProfile,
        out_sr: int,
        chunk_ms: int,
    ) -> Generator[bytes, None, None]:
        """Synthesize speech and yield WAV-encoded chunks."""

        if not text:
            return
            yield  # pragma: no cover - generator formality

        pipe = self._load_pipe()
        LOGGER.debug(
            "Synthesizing text | length=%s ref_sr=%s chunk_ms=%s", len(text), profile.speaker_sr, chunk_ms
        )
        audio = pipe.tts(
            text=text,
            ref_audio=profile.speaker_wav,
            ref_sr=profile.speaker_sr,
            sr=out_sr,
            prompt_style=profile.style_notes or None,
        )
        audio_np = np.array(audio, dtype=np.float32).reshape(-1)
        if audio_np.size == 0:
            return

        chunk_samples = max(int(out_sr * (chunk_ms / 1000.0)), 1)
        for start in range(0, audio_np.size, chunk_samples):
            end = min(start + chunk_samples, audio_np.size)
            chunk = audio_np[start:end]
            buffer = BytesIO()
            sf.write(buffer, chunk, out_sr, format="WAV", subtype="PCM_16")
            yield buffer.getvalue()


__all__ = ["VoiceProfile", "F5Cloner"]
