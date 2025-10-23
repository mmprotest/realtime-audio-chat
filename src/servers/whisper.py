"""FastAPI application that exposes a local Whisper transcription API."""
from __future__ import annotations

import base64
import binascii
import io
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

AudioArray = np.ndarray
Transcriber = Callable[[AudioArray, str | None], str]
ModelLoader = Callable[[], object]

_DEFAULT_SAMPLE_RATE = 16_000


class TranscribeRequest(BaseModel):
    """Request payload for the `/transcribe` endpoint."""

    audio: str
    sample_rate: int | None = None
    language: str | None = None

    @classmethod
    def normalise_language(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    # Pydantic v1 model hook (kept simple for compatibility with pydantic<1.10)
    def __init__(self, **data):  # type: ignore[override]
        if "language" in data:
            data = {**data, "language": self.normalise_language(data.get("language"))}
        super().__init__(**data)


class TranscribeResponse(BaseModel):
    """Response payload containing the recognized text."""

    text: str
    duration: float


def create_app(
    *,
    model_loader: ModelLoader | None = None,
    transcriber_factory: Callable[[object], Transcriber] | None = None,
) -> FastAPI:
    """Create a FastAPI app that serves Whisper transcriptions."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        loader = model_loader or _default_model_loader
        app.state.model = loader()
        factory = transcriber_factory or _default_transcriber_factory
        app.state.transcribe = factory(app.state.model)
        app.state.target_sample_rate = _target_sample_rate()
        app.state.fallback_language = _fallback_language()
        try:
            yield
        finally:
            transcribe = getattr(app.state, "transcribe", None)
            closer = getattr(transcribe, "close", None)
            if callable(closer):
                closer()
            model = getattr(app.state, "model", None)
            closer = getattr(model, "close", None)
            if callable(closer):
                closer()

    app = FastAPI(title="Local Whisper Server", version="1.0.0", lifespan=lifespan)

    @app.get("/healthz")
    async def _health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/transcribe", response_model=TranscribeResponse)
    async def _transcribe(request: TranscribeRequest) -> TranscribeResponse:
        if not request.audio:
            raise HTTPException(status_code=400, detail="Missing audio payload")

        try:
            audio_bytes = base64.b64decode(request.audio, validate=True)
        except (ValueError, binascii.Error):  # type: ignore[attr-defined]
            raise HTTPException(status_code=400, detail="Invalid base64 audio") from None

        samples, source_rate = _decode_wav(audio_bytes)
        target_rate = app.state.target_sample_rate  # type: ignore[attr-defined]
        resampled = _resample(samples, source_rate, target_rate)

        if resampled.size == 0 or np.allclose(resampled, 0.0):
            raise HTTPException(status_code=400, detail="Audio payload contains no data")

        transcribe = app.state.transcribe  # type: ignore[attr-defined]
        fallback_language = app.state.fallback_language  # type: ignore[attr-defined]
        language = request.language
        try:
            text = transcribe(resampled, language)
        except ValueError as exc:
            if (
                language is None
                and _language_detection_failed(exc)
                and fallback_language
            ):
                try:
                    text = transcribe(resampled, fallback_language)
                except ValueError as fallback_exc:
                    raise HTTPException(
                        status_code=422,
                        detail="Language detection failed; specify the language parameter",
                    ) from fallback_exc
            elif language is None and _language_detection_failed(exc):
                raise HTTPException(
                    status_code=422,
                    detail="Language detection failed; specify the language parameter",
                ) from exc
            else:
                raise
        cleaned_text = text.strip()
        if not cleaned_text and language is None and fallback_language:
            cleaned_text = transcribe(resampled, fallback_language).strip()

        if not cleaned_text:
            raise HTTPException(status_code=422, detail="Transcription produced no text")
        duration = len(resampled) / float(target_rate)
        return TranscribeResponse(text=cleaned_text, duration=duration)

    return app


def _default_model_loader() -> object:
    """Load a Whisper model using the faster-whisper backend."""

    from faster_whisper import WhisperModel

    model_name = os.getenv("WHISPER_MODEL", "base")
    device = os.getenv("WHISPER_DEVICE")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "default")

    kwargs: dict[str, object] = {}
    if device:
        kwargs["device"] = device
    if compute_type and compute_type.lower() != "default":
        kwargs["compute_type"] = compute_type

    return WhisperModel(model_name, **kwargs)


def _default_transcriber_factory(model: object) -> Transcriber:
    from faster_whisper import WhisperModel

    if not isinstance(model, WhisperModel):  # pragma: no cover - defensive
        raise TypeError("Expected model to be an instance of faster_whisper.WhisperModel")

    def _transcribe(audio: AudioArray, language: str | None) -> str:
        segments_iter, _ = model.transcribe(audio, language=language)
        segments = list(segments_iter)
        cleaned: list[str] = []
        for segment in segments:
            text = getattr(segment, "text", "")
            if not isinstance(text, str):
                continue
            normalised = _normalise_segment_text(text)
            if normalised:
                cleaned.append(normalised)
        if cleaned:
            return " ".join(cleaned)

        fallback = " ".join(
            text.strip()
            for text in (getattr(segment, "text", "") for segment in segments)
            if isinstance(text, str) and text.strip()
        )
        return fallback

    return _transcribe


def _decode_wav(data: bytes) -> tuple[AudioArray, int]:
    import wave

    try:
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            sample_width = wav_file.getsampwidth()
            if sample_width != 2:
                raise HTTPException(status_code=400, detail="Only 16-bit PCM WAV is supported")
            channels = wav_file.getnchannels()
            frames = wav_file.readframes(wav_file.getnframes())
            sample_rate = wav_file.getframerate()
    except (wave.Error, EOFError) as exc:
        raise HTTPException(status_code=400, detail="Invalid WAV payload") from exc

    array = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        array = array.reshape(-1, channels).mean(axis=1)
    return array, sample_rate


def _resample(samples: AudioArray, source_rate: int, target_rate: int) -> AudioArray:
    if source_rate == target_rate:
        return samples
    duration = len(samples) / float(source_rate)
    target_length = max(int(duration * target_rate), 1)
    source_times = np.linspace(0.0, duration, num=len(samples), endpoint=False)
    target_times = np.linspace(0.0, duration, num=target_length, endpoint=False)
    return np.interp(target_times, source_times, samples).astype(np.float32)


def _target_sample_rate() -> int:
    try:
        return int(os.getenv("WHISPER_TARGET_SAMPLE_RATE", str(_DEFAULT_SAMPLE_RATE)))
    except ValueError:  # pragma: no cover - defensive
        return _DEFAULT_SAMPLE_RATE


def _fallback_language() -> str | None:
    fallback = os.getenv("WHISPER_FALLBACK_LANGUAGE", "en").strip()
    return fallback or None


def _language_detection_failed(error: ValueError) -> bool:
    message = str(error).lower()
    return "max()" in message and "empty" in message


def _normalise_segment_text(text: str) -> str:
    cleaned = text.replace("▁", " ")
    return " ".join(cleaned.split())


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.getenv("WHISPER_HOST", "0.0.0.0"),
        port=int(os.getenv("WHISPER_PORT", "9000")),
    )
