"""FastAPI + Gradio application providing realtime audio chat with local STT/TTS."""
from __future__ import annotations

import os
import time
from typing import Generator, Iterable, Optional

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastrtc import AdditionalOutputs, ReplyOnPause, Stream, get_twilio_turn_credentials
from gradio.utils import get_space
from groq import Groq
from numpy.typing import NDArray

from .config import get_settings
from .local_clients import F5LocalTTS, WhisperSTTClient

load_dotenv()
settings = get_settings()

groq_client = Groq()
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
ChatHistory = list[dict[str, str]]


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
def response(audio: AudioTuple, chatbot: Optional[ChatHistory] = None):
    chatbot = chatbot or []
    messages = [{"role": d["role"], "content": d["content"]} for d in chatbot]
    start = time.time()
    text = stt_model.stt(audio)
    print("transcription", time.time() - start)
    print("prompt", text)
    chatbot.append({"role": "user", "content": text})
    yield AdditionalOutputs(chatbot)
    messages.append({"role": "user", "content": text})
    response_text = (
        groq_client.chat.completions.create(
            model=settings.groq_model,
            max_tokens=settings.groq_max_tokens,
            messages=messages,  # type: ignore[arg-type]
        )
        .choices[0]
        .message.content
    )

    chatbot.append({"role": "assistant", "content": response_text})

    chunks = tts_client.text_to_speech.convert_as_stream(
        text=response_text,
        voice_id=settings.f5_tts_voice,
        output_format=settings.f5_tts_output_format,
    )
    for sample_rate, audio_array in _pcm_chunks_to_arrays(chunks, settings.tts_sample_rate):
        yield (sample_rate, audio_array)
    yield AdditionalOutputs(chatbot)


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
    ui_args={"title": "LLM Voice Chat (Powered by Groq, F5-TTS, Whisper, and WebRTC ⚡️)"},
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
