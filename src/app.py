"""FastAPI + Gradio application providing realtime audio chat with local STT/TTS."""
from __future__ import annotations

import os
import time
from collections.abc import Generator, Iterable
from functools import lru_cache

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastrtc import AdditionalOutputs, ReplyOnPause, Stream, get_twilio_turn_credentials
from gradio.utils import get_space
import httpx
from openai import OpenAI
from numpy.typing import NDArray

from .chat_history import (
    ChatEntry,
    ChatHistory,
    chatbot_to_messages,
    normalize_chat_history,
)
from .config import get_settings
from .local_clients import F5LocalTTS, WhisperSTTClient

load_dotenv()
settings = get_settings()

_openai_client_kwargs: dict[str, object] = {}
if settings.openai_base_url:
    _openai_client_kwargs["base_url"] = settings.openai_base_url
if settings.openai_api_key:
    _openai_client_kwargs["api_key"] = settings.openai_api_key


def _create_openai_client() -> OpenAI:
    """Instantiate an OpenAI client resilient to incompatible httpx versions."""

    try:
        return OpenAI(**_openai_client_kwargs)
    except TypeError as exc:  # pragma: no cover - defensive guard for Windows envs
        if "proxies" not in str(exc):
            raise
        http_client = httpx.Client(trust_env=True)
        return OpenAI(http_client=http_client, **_openai_client_kwargs)


@lru_cache(maxsize=1)
def _get_openai_client() -> OpenAI:
    return _create_openai_client()


@lru_cache(maxsize=1)
def _get_tts_client() -> F5LocalTTS:
    return F5LocalTTS(
        base_url=settings.f5_tts_url,
        voice_id=settings.f5_tts_voice,
        output_format=settings.f5_tts_output_format,
        timeout=settings.http_timeout,
    )


@lru_cache(maxsize=1)
def _get_stt_client() -> WhisperSTTClient:
    return WhisperSTTClient(
        base_url=settings.whisper_url,
        language=settings.whisper_language,
        timeout=settings.http_timeout,
    )


AudioTuple = tuple[int, NDArray[np.int16 | np.float32]]


def _pcm_chunks_to_arrays(
    chunks: Iterable[bytes],
    sample_rate: int,
    sample_width: int = 2,
) -> Generator[tuple[int, NDArray[np.int16]], None, None]:
    """Convert a stream of PCM bytes into numpy arrays suitable for Gradio."""

    remainder = b""
    for chunk in chunks:
        if not chunk:
            continue
        combined = remainder + chunk
        remainder_len = len(combined) % sample_width
        if remainder_len:
            remainder = combined[-remainder_len:]
            combined = combined[:-remainder_len]
        else:
            remainder = b""
        if combined:
            audio_array = np.frombuffer(combined, dtype="<i2")
            yield sample_rate, audio_array.reshape(1, -1)
    if remainder:
        padded = remainder + b"\x00" * (sample_width - len(remainder))
        audio_array = np.frombuffer(padded, dtype="<i2")
        yield sample_rate, audio_array.reshape(1, -1)


# Backwards compatibility for modules importing the old private helpers.
_normalize_chat_history = normalize_chat_history
_chatbot_to_messages = chatbot_to_messages


def response(
    audio: AudioTuple,
    chatbot: ChatHistory | ChatEntry | None = None,
    event: object | None = None,
):
    _ = event  # ReplyOnPause provides an interaction event we don't currently use.
    history = normalize_chat_history(chatbot)
    messages = chatbot_to_messages(history)
    overall_start = time.perf_counter()

    stt_start = time.perf_counter()
    stt_client = _get_stt_client()
    text = stt_client.stt(audio)
    stt_duration = time.perf_counter() - stt_start
    print(f"[STT] Transcription ({stt_duration:.2f}s): {text}")
    history.append({"role": "user", "content": text})
    yield AdditionalOutputs(history)
    messages.append({"role": "user", "content": text})

    llm_start = time.perf_counter()
    openai_client = _get_openai_client()
    completion = openai_client.chat.completions.create(
        model=settings.openai_model,
        max_tokens=settings.openai_max_tokens,
        messages=messages,  # type: ignore[arg-type]
    )
    llm_duration = time.perf_counter() - llm_start
    choice = completion.choices[0]
    response_text = choice.message.content or ""
    print(f"[LLM] Response ({llm_duration:.2f}s): {response_text}")

    history.append({"role": "assistant", "content": response_text})

    if not response_text.strip():
        overall_duration = time.perf_counter() - overall_start
        print("[TTS] Skipping audio generation (empty response)")
        print(f"[Pipeline] Total duration: {overall_duration:.2f}s")
        yield AdditionalOutputs(history)
        return

    tts_start = time.perf_counter()
    tts_client = _get_tts_client()
    chunks = tts_client.text_to_speech.convert_as_stream(
        text=response_text,
        voice_id=settings.f5_tts_voice,
        output_format=settings.f5_tts_output_format,
    )
    for sample_rate, audio_array in _pcm_chunks_to_arrays(chunks, settings.tts_sample_rate):
        yield (sample_rate, audio_array)
    tts_duration = time.perf_counter() - tts_start
    overall_duration = time.perf_counter() - overall_start
    print(f"[TTS] Audio generation ({tts_duration:.2f}s)")
    print(f"[Pipeline] Total duration: {overall_duration:.2f}s")
    yield AdditionalOutputs(history)


chatbot = gr.Chatbot(type="messages")
stream = Stream(
    modality="audio",
    mode="send-receive",
    handler=ReplyOnPause(response, input_sample_rate=settings.input_sample_rate),
    additional_outputs_handler=lambda a, b: b,
    additional_inputs=[chatbot],
    additional_outputs=[chatbot],
    rtc_configuration=get_twilio_turn_credentials() if get_space() else None,
    concurrency_limit=5 if get_space() else None,
    time_limit=90 if get_space() else None,
    ui_args={"title": "LLM Voice Chat (Powered by OpenAI-compatible LLMs, F5-TTS, Whisper, and WebRTC ⚡️)"},
)

# Mount the STREAM UI to the FastAPI app
# Because I don't want to build the UI manually
app = FastAPI()
app = gr.mount_gradio_app(app, stream.ui, path="/")


if __name__ == "__main__":
    os.environ["GRADIO_SSR_MODE"] = "false"

    if (mode := os.getenv("MODE")) == "UI":
        stream.ui.launch(server_port=7860)
    elif mode == "PHONE":
        stream.fastphone(host="0.0.0.0", port=7860)
    else:
        stream.ui.launch(server_port=7860)
