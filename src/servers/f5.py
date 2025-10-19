"""FastAPI application that serves streaming audio from a local F5-TTS model."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from importlib import resources

VoiceMap = Dict[str, "VoiceConfig"]
ModelLoader = Callable[[], "F5Model"]


@dataclass
class VoiceConfig:
    """Configuration mapping a voice id to reference audio/text."""

    voice_id: str
    audio_path: str
    text_path: str
    description: str | None = None


class F5Model:
    """Protocol describing the portion of the F5-TTS API we consume."""

    def infer(
        self,
        ref_file: str,
        ref_text: str,
        gen_text: str,
        **kwargs,
    ) -> tuple[np.ndarray, int, object]: ...


class TTSRequest(yaml.YAMLObject):
    pass


class TTSRequestModel:
    def __init__(self, *, text: str, voice_id: str | None, output_format: str | None) -> None:
        self.text = text
        self.voice_id = voice_id
        self.output_format = output_format or "pcm_s16le"


def create_app(
    *,
    model_loader: ModelLoader | None = None,
    voice_loader: Callable[[], VoiceMap] | None = None,
    default_voice_id: str | None = None,
) -> FastAPI:
    """Create a FastAPI server that exposes `/tts` for streaming audio."""

    app = FastAPI(title="Local F5-TTS Server", version="1.0.0")

    @app.on_event("startup")
    async def _load() -> None:  # pragma: no cover - simple initialization
        loader = model_loader or _default_model_loader
        app.state.model = loader()
        loader_voice = voice_loader or _load_default_voices
        voices = loader_voice()
        if not voices:
            raise RuntimeError("No voices are configured for the F5-TTS server")
        app.state.voices = voices
        app.state.default_voice_id = default_voice_id or os.getenv("F5_TTS_VOICE", next(iter(voices)))
        if app.state.default_voice_id not in voices:
            raise RuntimeError(f"Default voice '{app.state.default_voice_id}' is not available")

    @app.get("/voices")
    async def _voices() -> dict[str, object]:
        voices: VoiceMap = app.state.voices  # type: ignore[attr-defined]
        return {
            "default": app.state.default_voice_id,  # type: ignore[attr-defined]
            "voices": {
                vid: {
                    "audio_path": voice.audio_path,
                    "text_path": voice.text_path,
                    "description": voice.description,
                }
                for vid, voice in voices.items()
            },
        }

    @app.post("/tts")
    async def _tts(payload: dict[str, object]):
        request = _parse_request(payload)
        voice_id = request.voice_id or app.state.default_voice_id  # type: ignore[attr-defined]
        voices: VoiceMap = app.state.voices  # type: ignore[attr-defined]
        if voice_id not in voices:
            raise HTTPException(status_code=404, detail=f"Unknown voice '{voice_id}'")
        if request.output_format != "pcm_s16le":
            raise HTTPException(status_code=400, detail="Only pcm_s16le output is supported")

        voice = voices[voice_id]
        ref_audio = voice.audio_path
        ref_text = _read_text_file(voice.text_path)

        model: F5Model = app.state.model  # type: ignore[attr-defined]
        wav, sample_rate, _ = model.infer(ref_audio, ref_text, request.text)
        stream = _pcm_stream(wav, sample_rate)
        headers = {"X-Sample-Rate": str(sample_rate)}
        return StreamingResponse(stream, media_type="application/octet-stream", headers=headers)

    return app


def _default_model_loader() -> F5Model:
    from f5_tts import F5TTS

    model_name = os.getenv("F5_TTS_MODEL", "F5TTS_v1_Base")
    device = os.getenv("F5_TTS_DEVICE")
    kwargs = {"device": device} if device else {}
    return F5TTS(model=model_name, **kwargs)


def _load_default_voices() -> VoiceMap:
    voices_file_env = os.getenv("F5_TTS_VOICES_FILE")
    if voices_file_env:
        path = os.path.abspath(voices_file_env)
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
            return _parse_voice_map(data, base_dir=os.path.dirname(path))

    resource_root = resources.files("resources.voices")
    with resources.as_file(resource_root.joinpath("voices.yaml")) as config_path:
        data = yaml.safe_load(config_path.read_text("utf-8"))
        base_dir = str(config_path.parent)
    return _parse_voice_map(data, base_dir=base_dir)


def _parse_voice_map(data: dict[str, object], base_dir: str) -> VoiceMap:
    voices: VoiceMap = {}
    entries = data.get("voices") if isinstance(data, dict) else None
    if not isinstance(entries, dict):
        raise RuntimeError("voices.yaml must contain a top-level 'voices' mapping")
    for voice_id, config in entries.items():
        if not isinstance(config, dict):
            continue
        audio = config.get("ref_audio")
        text = config.get("ref_text")
        description = config.get("description")
        if not isinstance(audio, str) or not isinstance(text, str):
            continue
        audio_path = os.path.join(base_dir, audio)
        text_path = os.path.join(base_dir, text)
        voices[voice_id] = VoiceConfig(
            voice_id=voice_id,
            audio_path=audio_path,
            text_path=text_path,
            description=description if isinstance(description, str) else None,
        )
    return voices


def _parse_request(payload: dict[str, object]) -> TTSRequestModel:
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="Field 'text' must be a non-empty string")
    voice_id = payload.get("voice_id")
    if voice_id is not None and not isinstance(voice_id, str):
        raise HTTPException(status_code=400, detail="Field 'voice_id' must be a string")
    output_format = payload.get("output_format")
    if output_format is not None and not isinstance(output_format, str):
        raise HTTPException(status_code=400, detail="Field 'output_format' must be a string")
    return TTSRequestModel(text=text, voice_id=voice_id, output_format=output_format)


def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except FileNotFoundError as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=500, detail=f"Reference text not found at {path}") from exc


def _pcm_stream(wav: np.ndarray, sample_rate: int, chunk_size: int = 8192) -> Iterable[bytes]:
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    pcm = np.clip(wav, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16).tobytes()
    buffer = memoryview(pcm16)
    for start in range(0, len(buffer), chunk_size):
        yield bytes(buffer[start : start + chunk_size])


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.getenv("F5_TTS_HOST", "0.0.0.0"),
        port=int(os.getenv("F5_TTS_PORT", "9880")),
    )
