"""FastAPI application that serves streaming audio from a local F5-TTS model."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, Iterable

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

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
    """Load the F5-TTS model or fall back to a lightweight synthesizer."""

    try:
        from f5_tts.api import F5TTS  # type: ignore[import-not-found]

        _ensure_ffmpeg()

        model_name = os.getenv("F5_TTS_MODEL", "F5TTS_v1_Base")
        device = os.getenv("F5_TTS_DEVICE")
        kwargs = {"device": device} if device else {}
        return F5TTS(model=model_name, **kwargs)
    except ModuleNotFoundError:
        return _SimpleF5Model()


def _load_default_voices() -> VoiceMap:
    voices_file_env = os.getenv("F5_TTS_VOICES_FILE")
    if voices_file_env:
        path = os.path.abspath(voices_file_env)
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
            return _parse_voice_map(data, base_dir=os.path.dirname(path))

    voices_root = Path(__file__).resolve().parent.parent / "resources" / "voices"
    config_path = voices_root / "voices.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return _parse_voice_map(data, base_dir=str(voices_root))


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


class _SimpleF5Model:
    """Fallback synthesizer used when the real F5-TTS package isn't installed."""

    def __init__(self, sample_rate: int = 24_000) -> None:
        self.sample_rate = sample_rate

    def infer(self, ref_file: str, ref_text: str, gen_text: str, **_: object):
        del ref_file, ref_text
        wav = self._synthesize(gen_text)
        return wav, self.sample_rate, None

    def _synthesize(self, text: str) -> np.ndarray:
        if not text:
            text = " "

        chunk_duration = 0.12
        silence_duration = 0.05
        samples: list[np.ndarray] = []

        for char in text:
            if char.isspace():
                samples.append(np.zeros(int(self.sample_rate * silence_duration), dtype=np.float32))
                continue
            freq = 180.0 + (ord(char.lower()) % 32) * 12.0
            t = np.linspace(0.0, chunk_duration, int(self.sample_rate * chunk_duration), endpoint=False)
            envelope = np.linspace(1.0, 0.2, t.size)
            wave = 0.2 * envelope * np.sin(2 * np.pi * freq * t)
            samples.append(wave.astype(np.float32))

        return np.concatenate(samples) if samples else np.zeros(1, dtype=np.float32)


def _ensure_ffmpeg() -> None:
    """Configure pydub to point at an ffmpeg binary if one is available.

    The upstream F5-TTS implementation relies on pydub, which in turn expects an
    ffmpeg executable to exist on the system path. When running in lightweight
    environments (e.g., Windows without a global ffmpeg install) the import would
    otherwise fail later during synthesis with a vague warning. This helper
    normalises the discovery of the binary and raises a clear error if none is
    found, pointing users at the `FFMPEG_BINARY` environment variable or the
    bundled `imageio-ffmpeg` wheel.
    """

    env_path = os.getenv("FFMPEG_BINARY")
    candidates = [env_path] if env_path else []
    for name in ("ffmpeg", "ffmpeg.exe"):
        candidate = shutil.which(name)
        if candidate:
            candidates.append(candidate)

    ffmpeg_path: str | None = next((p for p in candidates if p), None)
    if not ffmpeg_path:
        try:  # pragma: no cover - only runs when bundled binary is available
            import imageio_ffmpeg  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Unable to locate an ffmpeg binary. Install ffmpeg, set "
                "FFMPEG_BINARY to its path, or install the imageio-ffmpeg package."
            ) from exc
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    os.environ.setdefault("FFMPEG_BINARY", ffmpeg_path)

    try:
        from pydub import AudioSegment  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "pydub is required to run the F5-TTS server. Install the tts_server "
            "extra or add pydub to your environment."
        ) from exc

    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffmpeg = ffmpeg_path
    AudioSegment.ffprobe = ffmpeg_path


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.getenv("F5_TTS_HOST", "0.0.0.0"),
        port=int(os.getenv("F5_TTS_PORT", "9880")),
    )
