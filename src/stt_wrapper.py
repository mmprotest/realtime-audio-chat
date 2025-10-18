"""Moonshine STT wrapper using FastRTC's factory."""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from fastrtc import get_stt_model


LOGGER = logging.getLogger(__name__)


class MoonshineSTT:
    """Simple adapter for the FastRTC Moonshine speech-to-text model."""

    def __init__(self) -> None:
        self._model = get_stt_model()
        LOGGER.info("Moonshine STT model initialized")

    def transcribe(self, audio: Tuple[int, np.ndarray]) -> str:
        sr, data = audio
        if data is None or data.size == 0:
            return ""
        LOGGER.debug("Transcribing audio with sample_rate=%s length=%s", sr, data.shape)
        result = self._model.transcribe((sr, data))
        if isinstance(result, dict):
            return result.get("text", "")
        return str(result or "")


_stt_instance: MoonshineSTT | None = None


def get_stt() -> MoonshineSTT:
    global _stt_instance
    if _stt_instance is None:
        _stt_instance = MoonshineSTT()
    return _stt_instance


__all__ = ["MoonshineSTT", "get_stt"]
