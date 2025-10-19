"""FastAPI application that exposes a local Whisper transcription API."""
from __future__ import annotations

import base64
import binascii
import io
import os
from typing import Callable

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

    app = FastAPI(title="Local Whisper Server", version="1.0.0")

    @app.on_event("startup")
    async def _load_model() -> None:  # pragma: no cover - simple startup hook
        loader = model_loader or _default_model_loader
        app.state.model = loader()
        factory = transcriber_factory or _default_transcriber_factory
        app.state.transcribe = factory(app.state.model)
        app.state.target_sample_rate = _target_sample_rate()

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

        transcribe = app.state.transcribe  # type: ignore[attr-defined]
        text = transcribe(resampled, request.language)
        duration = len(resampled) / float(target_rate)
        return TranscribeResponse(text=text, duration=duration)

    return app


def _default_model_loader() -> object:
    import whisper

    model_name = os.getenv("WHISPER_MODEL", "base")
    device = os.getenv("WHISPER_DEVICE")
    kwargs = {"device": device} if device else {}
    return whisper.load_model(model_name, **kwargs)


def _default_transcriber_factory(model: object) -> Transcriber:
    import whisper

    def _transcribe(audio: AudioArray, language: str | None) -> str:
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)
        options = whisper.DecodingOptions(language=language, fp16=False)
        result = whisper.decode(model, mel, options)
        return result.text.strip()

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


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.getenv("WHISPER_HOST", "0.0.0.0"),
        port=int(os.getenv("WHISPER_PORT", "9000")),
    )
