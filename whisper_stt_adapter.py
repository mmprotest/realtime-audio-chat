"""Local Whisper.cpp speech-to-text adapter without NumPy 2.x dependency conflicts."""
from __future__ import annotations

from functools import lru_cache
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

try:
    from pywhispercpp.constants import AVAILABLE_MODELS
    from pywhispercpp.model import Model
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - import guard
    raise ImportError(
        "pywhispercpp is required for speech-to-text. Install it with "
        "`pip install pywhispercpp`."
    ) from exc

try:
    import resampy
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - import guard
    raise ImportError(
        "resampy is required for resampling non-16kHz audio. Install it with "
        "`pip install resampy`."
    ) from exc


class STTModel(Protocol):
    """Protocol describing the speech-to-text interface consumed by the app."""

    def stt(self, audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str:
        ...


class WhisperCppSTT(STTModel):
    """Thin wrapper around pywhispercpp that normalises audio payloads."""

    def __init__(self, model: str = "base.en", models_dir: str | None = None) -> None:
        if model not in AVAILABLE_MODELS:
            formatted_models = "\n".join(f"  - {candidate}" for candidate in AVAILABLE_MODELS)
            raise ValueError(
                f"Model '{model}' not found. Available models:\n{formatted_models}"
            )

        self._model = Model(model=model, models_dir=models_dir)

    def stt(self, audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str:
        sample_rate, samples = audio

        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0

        if sample_rate != 16_000:
            samples = resampy.resample(samples, sample_rate, 16_000)

        if samples.ndim == 1:
            samples = samples.reshape(1, -1)

        segments = self._model.transcribe(samples)
        return " ".join(segment.text for segment in segments)

    def list_models(self) -> None:
        for candidate in AVAILABLE_MODELS:
            print(candidate)


@lru_cache
def get_stt_model(model: str = "base.en", models_dir: str | None = None) -> STTModel:
    """Lazily instantiate and warm up the Whisper.cpp model."""

    stt = WhisperCppSTT(model=model, models_dir=models_dir)

    # Warm up the model with a small chunk of silence to trigger graph initialisation.
    warmup_rate = 16_000
    warmup_audio = np.zeros(warmup_rate * 2, dtype=np.float32)
    stt.stt((warmup_rate, warmup_audio))

    return stt
