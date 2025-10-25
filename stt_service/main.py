import io
import os
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency guard
    torch = None

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from faster_whisper import WhisperModel

from .service_config import STTSettings

app = FastAPI(title="STT Service", version="1.0.0")


# Resolve device & compute_type robustly
def _torch_cuda_available() -> bool:
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - defensive
        return False


def pick_device() -> str:
    want = os.getenv("STT_DEVICE", "auto").lower()
    if want == "cuda":
        return "cuda" if _torch_cuda_available() else "cpu"
    if want == "cpu":
        return "cpu"
    # auto
    return "cuda" if _torch_cuda_available() else "cpu"


def pick_compute_type(device: str) -> str:
    ct_env = os.getenv("STT_COMPUTE_TYPE", "").lower()
    if ct_env:
        return ct_env
    return "float16" if device == "cuda" else "int8"


MODEL_NAME = os.getenv("STT_MODEL", "small")
DEVICE = pick_device()
COMPUTE_TYPE = pick_compute_type(DEVICE)
SETTINGS = STTSettings(model_name=MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)

model = WhisperModel(
    SETTINGS.model_name,
    device=SETTINGS.device,
    compute_type=SETTINGS.compute_type,
)


class TranscribeResponse(BaseModel):
    text: str
    language: str | None = None
    segments: list[dict[str, Any]]


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "device": SETTINGS.device,
        "compute_type": SETTINGS.compute_type,
        "model": SETTINGS.model_name,
    }


@app.post("/v1/transcribe", response_model=TranscribeResponse)
async def transcribe(file: UploadFile = File(...), language: str | None = None):
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(400, "Empty audio file")
        audio_buf = io.BytesIO(raw)
        # faster-whisper accepts PCM/WAV/MP3/FLAC etc. Let it detect
        segments, info = model.transcribe(
            audio_buf,
            language=language or SETTINGS.language,
            vad_filter=SETTINGS.vad_filter,
            beam_size=SETTINGS.beam_size,
        )
        segs = []
        full_text: list[str] = []
        for s in segments:
            d = {
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text,
                "avg_logprob": float(getattr(s, "avg_logprob", 0.0)),
                "no_speech_prob": float(getattr(s, "no_speech_prob", 0.0)),
            }
            segs.append(d)
            full_text.append(s.text)
        return TranscribeResponse(
            text=" ".join(full_text).strip(),
            language=getattr(info, "language", None),
            segments=segs,
        )
    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover - runtime errors bubble up
        raise HTTPException(500, f"Transcription failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "stt_service.main:app",
        host="127.0.0.1",
        port=int(os.getenv("STT_PORT", "5007")),
        reload=False,
    )
