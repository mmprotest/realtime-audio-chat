"""FastRTC-compatible adapter for the F5-TTS voice cloning model."""
from __future__ import annotations

import asyncio
import os
import re
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from f5_tts.api import F5TTS


@dataclass
class TTSOptions:
    """Placeholder for future configurable options."""

    kwargs: dict[str, Any] | None = None


class F5TTSModel:
    """Wrap the F5-TTS API to satisfy FastRTC's ``TTSModel`` protocol."""

    def __init__(
        self,
        ref_wav: str,
        ref_text: Optional[str] = None,
        *,
        model_name: str | None = None,
        target_sample_rate: int | None = None,
        transcription_kwargs: Optional[dict[str, Any]] = None,
        inference_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        if not os.path.exists(ref_wav):
            raise FileNotFoundError(f"Reference audio not found: {ref_wav}")

        self._f5 = F5TTS(model=model_name or "F5TTS_v1_Base")
        self._ref_wav = ref_wav
        self._target_sample_rate = target_sample_rate
        self._inference_kwargs = inference_kwargs or {}
        self._transcription_kwargs = transcription_kwargs or {}

        if ref_text is not None and ref_text.strip():
            self._ref_text = ref_text.strip()
        else:
            self._ref_text = self._f5.transcription(
                ref_audio=self._ref_wav, **self._transcription_kwargs
            )

    def _infer_once(self, text: str) -> tuple[int, NDArray[np.float32]]:
        infer_kwargs = {
            "ref_file": self._ref_wav,
            "ref_text": self._ref_text,
            "gen_text": text,
            **self._inference_kwargs,
        }

        if hasattr(self._f5, "inference"):
            wav, sample_rate, _ = self._f5.inference(**infer_kwargs)
        elif hasattr(self._f5, "infer"):
            wav, sample_rate, _ = self._f5.infer(**infer_kwargs)
        else:
            raise AttributeError(
                "F5TTS API changed: expected `infer` or `inference` method to be available."
            )

        audio = np.asarray(wav, dtype=np.float32).reshape(1, -1)
        if self._target_sample_rate is not None and sample_rate != self._target_sample_rate:
            raise ValueError(
                "F5-TTS returned an unexpected sample rate. Configure the model "
                "to emit the desired rate or disable target_sample_rate."
            )
        return sample_rate, audio

    def tts(
        self, text: str, options: Optional[TTSOptions] = None
    ) -> tuple[int, NDArray[np.float32]]:
        """Generate a single utterance synchronously."""

        _ = options  # options reserved for future use
        return self._infer_once(text)

    def stream_tts_sync(
        self, text: str, options: Optional[TTSOptions] = None
    ) -> Generator[tuple[int, NDArray[np.float32]], None, None]:
        """Yield audio for each sentence-sized segment synchronously."""

        _ = options
        for segment in _split_on_sentences(text):
            if not segment.strip():
                continue
            yield self._infer_once(segment.strip())

    async def stream_tts(
        self, text: str, options: Optional[TTSOptions] = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.float32]], None]:
        """Asynchronously yield audio segments using a background thread."""

        loop = asyncio.get_running_loop()
        for segment in _split_on_sentences(text):
            if not segment.strip():
                continue
            sr, audio = await loop.run_in_executor(None, self._infer_once, segment.strip())
            yield sr, audio


def _split_on_sentences(text: str) -> list[str]:
    pattern = re.compile(r"(?<=[.!?]\s)|(?<=\n)")
    parts = pattern.split(text)
    if not parts:
        return [text]
    return [part for part in parts if part]
