"""F5-TTS wrapper for voice cloning and streaming."""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from io import BytesIO
from typing import Generator, Optional

import numpy as np

try:  # pragma: no cover - dependency import is trivial when present
    import soundfile as sf
except Exception:  # pragma: no cover - exercised when soundfile missing
    sf = None

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class VoiceProfile:
    """Stores reference audio and style instructions."""

    speaker_wav: Optional[np.ndarray] = None
    speaker_sr: Optional[int] = None
    style_notes: str = ""
    reference_text: str = ""


class F5Cloner:
    """Lazy-loading helper for F5-TTS voice cloning."""

    def __init__(self, *, device: str = "cpu") -> None:
        self.device = device
        self._pipe = None
        LOGGER.info("F5Cloner initialized | device=%s", device)

    def _load_pipe(self):  # pragma: no cover - small wrapper
        if self._pipe is None:
            F5TTS_cls = _resolve_f5_api()
            self._pipe = F5TTS_cls(device=self.device)
            LOGGER.info("Loaded F5-TTS pipeline")
        return self._pipe

    def ensure_pipeline(self) -> None:
        """Ensure the underlying F5 pipeline is instantiated."""

        self._load_pipe()

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
        audio_np = _synthesize(pipe, text, profile, out_sr)
        if audio_np.size == 0:
            return

        chunk_samples = max(int(out_sr * (chunk_ms / 1000.0)), 1)
        for start in range(0, audio_np.size, chunk_samples):
            end = min(start + chunk_samples, audio_np.size)
            chunk = audio_np[start:end]
            yield _encode_wav(chunk, out_sr)


def _encode_wav(chunk: np.ndarray, sr: int) -> bytes:
    """Encode the provided samples into a mono 16-bit WAV payload."""

    buffer = BytesIO()
    if sf is not None:
        sf.write(buffer, chunk, sr, format="WAV", subtype="PCM_16")
        return buffer.getvalue()

    import struct
    import wave

    samples = [float(x) for x in chunk]
    # Clamp to [-1, 1] before scaling to 16-bit PCM
    pcm = [
        max(-32768, min(32767, int(round(sample * 32767.0))))
        for sample in (_clamp(sample, -1.0, 1.0) for sample in samples)
    ]

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        wav_file.writeframes(b"".join(struct.pack("<h", value) for value in pcm))

    return buffer.getvalue()


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return minimum if value < minimum else maximum if value > maximum else value


def _synthesize(pipe: object, text: str, profile: VoiceProfile, out_sr: int) -> np.ndarray:
    """Invoke the available F5 API and return mono audio samples."""

    if hasattr(pipe, "tts"):
        audio = pipe.tts(
            text=text,
            ref_audio=profile.speaker_wav,
            ref_sr=profile.speaker_sr,
            sr=out_sr,
            prompt_style=profile.style_notes or None,
        )
        return np.array(audio, dtype=np.float32).reshape(-1)

    if hasattr(pipe, "infer"):
        return _infer_v1(pipe, text, profile, out_sr)

    if callable(pipe):  # pragma: no cover - extremely defensive
        audio = pipe(text)
        return np.array(audio, dtype=np.float32).reshape(-1)

    raise AttributeError("F5-TTS pipeline exposes no supported synthesis method")


class _SilentProgress:
    """Minimal tqdm-compatible shim to silence console output."""

    def tqdm(self, iterable, *args, **kwargs):  # pragma: no cover - trivial container
        return iterable


def _infer_v1(pipe: object, text: str, profile: VoiceProfile, out_sr: int) -> np.ndarray:
    """Compatibility bridge for packages exposing an ``infer`` entry point."""

    if profile.speaker_wav is None or profile.speaker_sr is None:
        LOGGER.warning("Cannot synthesize without a reference voice sample")
        return np.array([], dtype=np.float32)

    wav = profile.speaker_wav.astype(np.float32, copy=False)
    sr = int(profile.speaker_sr)
    ref_text = getattr(profile, "reference_text", "") or ""

    try:
        path = _write_temp_wav(wav, sr)
    except ValueError:
        LOGGER.warning("Reference audio contained no samples after preprocessing")
        return np.array([], dtype=np.float32)
    try:
        show_info = lambda *args, **kwargs: None  # noqa: E731 - small inline stub
        progress = _SilentProgress()
        result = pipe.infer(
            ref_file=path,
            ref_text=ref_text,
            gen_text=text,
            show_info=show_info,
            progress=progress,
        )
    except TypeError:
        # Older builds may omit keyword support; retry positionally.
        result = pipe.infer(path, ref_text, text)
    finally:
        try:
            os.remove(path)
        except OSError:  # pragma: no cover - cleanup best effort
            LOGGER.debug("Temporary reference file already cleaned up", exc_info=True)

    if not isinstance(result, tuple):
        audio = np.array(result, dtype=np.float32).reshape(-1)
        native_sr = getattr(pipe, "target_sample_rate", out_sr)
    else:
        if len(result) == 3:
            audio, native_sr, _ = result
        elif len(result) >= 2:
            audio, native_sr = result[:2]
        else:  # pragma: no cover - defensive guard
            audio, native_sr = result[0], out_sr
        audio = np.array(audio, dtype=np.float32).reshape(-1)

    native_sr = int(native_sr) if native_sr else out_sr
    if native_sr != out_sr and audio.size:
        audio = _resample_linear(audio, native_sr, out_sr)

    return audio.astype(np.float32, copy=False)


def _write_temp_wav(wav: np.ndarray, sr: int) -> str:
    """Persist the reference clip to disk for APIs that require a filepath."""

    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim != 1:
        wav = wav.reshape(-1)
    if wav.size == 0:
        raise ValueError("Reference audio is empty")

    # NamedTemporaryFile(delete=False) for cross-platform compatibility.
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    path = tmp.name
    tmp.close()

    if sf is not None:
        sf.write(path, wav, sr, format="WAV", subtype="PCM_16")
    else:  # pragma: no cover - fallback
        import wave

        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sr)
            samples = (_clamp(sample, -1.0, 1.0) for sample in wav)
            pcm = [int(round(value * 32767.0)) for value in samples]
            wav_file.writeframes(b"".join(int(sample).to_bytes(2, "little", signed=True) for sample in pcm))

    return path


def _resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Lightweight linear resampler to avoid heavy dependencies."""

    if orig_sr == target_sr or audio.size == 0:
        return audio.astype(np.float32, copy=False)

    duration = audio.size / float(orig_sr)
    target_size = max(int(round(duration * target_sr)), 1)
    # Use linspace to map samples over the same duration.
    x_old = np.linspace(0.0, duration, num=audio.size, endpoint=False, dtype=np.float32)
    x_new = np.linspace(0.0, duration, num=target_size, endpoint=False, dtype=np.float32)
    resampled = np.interp(x_new, x_old, audio.astype(np.float32, copy=False))
    return resampled.astype(np.float32, copy=False)


__all__ = ["VoiceProfile", "F5Cloner"]


def _resolve_f5_api():  # pragma: no cover - exercised only with dependency installed
    """Locate the F5-TTS entry point across published package layouts."""

    try:
        import f5_tts  # type: ignore
    except ImportError as exc:
        raise RuntimeError("f5-tts is not installed. Please install dependencies.") from exc

    candidate_attrs = ("F5TTS", "F5TTSInference", "F5TTSPipeline")
    for attr in candidate_attrs:
        cls = getattr(f5_tts, attr, None)
        if cls is not None:
            return cls

    from importlib import import_module

    candidate_modules = (
        "f5_tts.api",
        "f5_tts.inference",
    )
    for module_name in candidate_modules:
        try:
            module = import_module(module_name)
        except ImportError:
            continue
        for attr in candidate_attrs:
            cls = getattr(module, attr, None)
            if cls is not None:
                return cls

    raise RuntimeError(
        "Installed f5-tts package does not expose an F5TTS-compatible class. "
        "Please upgrade/downgrade f5-tts or install extras that include the inference API."
    )
