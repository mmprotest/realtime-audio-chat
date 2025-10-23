"""FastAPI + Gradio application providing realtime audio chat with local STT/TTS."""
from __future__ import annotations

import os
import time
from typing import Generator, Iterable, Optional, Sequence

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastrtc import AdditionalOutputs, ReplyOnPause, Stream, get_twilio_turn_credentials
from gradio.utils import get_space
import httpx
from openai import OpenAI
from numpy.typing import NDArray

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
        # Some installations ship a version of httpx that expects the newer
        # ``proxy=`` keyword instead of ``proxies=``.  In that case we bypass the
        # default client construction (which forwards the incompatible argument)
        # and build an httpx client ourselves. ``trust_env=True`` ensures that
        # any proxy configuration from the host environment is still honoured.
        http_client = httpx.Client(trust_env=True)
        return OpenAI(http_client=http_client, **_openai_client_kwargs)


openai_client = _create_openai_client()
tts_client = F5LocalTTS(
    base_url=settings.f5_tts_url,
    voice_id=settings.f5_tts_voice,
    output_format=settings.f5_tts_output_format,
    timeout=settings.http_timeout,
)
stt_model = WhisperSTTClient(
    base_url=settings.whisper_url,
    language=settings.whisper_language,
    timeout=settings.http_timeout,
)


AudioTuple = tuple[int, NDArray[np.int16 | np.float32]]
ChatMessageDict = dict[str, str]
ChatEntry = ChatMessageDict | Sequence[str] | str
ChatHistory = list[ChatEntry]


def _normalize_chat_history(chatbot: Optional[ChatHistory | ChatEntry]) -> ChatHistory:
    """Ensure the chatbot history is always a mutable list."""

    if chatbot is None:
        return []
    if isinstance(chatbot, list):
        return chatbot
    if isinstance(chatbot, dict):
        return [chatbot]
    if isinstance(chatbot, str):
        return [{"role": "user", "content": chatbot}]
    if isinstance(chatbot, Sequence):
        return list(chatbot)
    return []


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


# See "Talk to Claude" in Cookbook for an example of how to keep
# track of the chat history.
def _chatbot_to_messages(chatbot: ChatHistory) -> list[ChatMessageDict]:
    """Convert Gradio Chatbot history into OpenAI-style chat messages."""

    messages: list[ChatMessageDict] = []
    for entry in chatbot:
        if isinstance(entry, dict):
            role = entry.get("role")
            content = entry.get("content")
            if role and content is not None:
                messages.append({"role": role, "content": str(content)})
        elif isinstance(entry, str):
            # Some versions of the Chatbot component may pass bare strings.
            messages.append({"role": "user", "content": entry})
        elif isinstance(entry, Sequence):
            # Tuple/list pairs (user, assistant) are the historical default.
            if len(entry) >= 1 and entry[0]:
                messages.append({"role": "user", "content": str(entry[0])})
            if len(entry) >= 2 and entry[1]:
                messages.append({"role": "assistant", "content": str(entry[1])})
        # Ignore any other unrecognized formats silently.
    return messages


def response(
    audio: AudioTuple,
    chatbot: Optional[ChatHistory | ChatEntry] = None,
    event: object | None = None,
):
    _ = event  # ReplyOnPause provides an interaction event we don't currently use.
    history = _normalize_chat_history(chatbot)
    messages = _chatbot_to_messages(history)
    start = time.time()
    text = stt_model.stt(audio)
    print("transcription", time.time() - start)
    print("prompt", text)
    history.append({"role": "user", "content": text})
    yield AdditionalOutputs(history)
    messages.append({"role": "user", "content": text})
    completion = openai_client.chat.completions.create(
        model=settings.openai_model,
        max_tokens=settings.openai_max_tokens,
        messages=messages,  # type: ignore[arg-type]
    )
    choice = completion.choices[0]
    response_text = choice.message.content or ""

    history.append({"role": "assistant", "content": response_text})

    chunks = tts_client.text_to_speech.convert_as_stream(
        text=response_text,
        voice_id=settings.f5_tts_voice,
        output_format=settings.f5_tts_output_format,
    )
    for sample_rate, audio_array in _pcm_chunks_to_arrays(chunks, settings.tts_sample_rate):
        yield (sample_rate, audio_array)
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
