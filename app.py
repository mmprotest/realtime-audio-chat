import os
import time

import gradio as gr
from dotenv import load_dotenv
from fastapi import FastAPI
from fastrtc import (
    AdditionalOutputs,
    ReplyOnPause,
    Stream,
    get_twilio_turn_credentials,
)
from fastrtc.tracks import WebRTCData
from fastrtc_whisper_cpp import get_stt_model
from gradio.utils import get_space
import numpy as np
from numpy.typing import NDArray
from openai import OpenAI

from f5_adapter import F5TTSModel

load_dotenv()

openai_client = OpenAI(
    api_key=os.getenv("LOCAL_OPENAI_API_KEY", "dummy"),
    base_url=os.getenv("LOCAL_OPENAI_BASE_URL", "http://localhost:8000/v1"),
)

stt_model = get_stt_model()

f5_tts_model = F5TTSModel(
    ref_wav=os.getenv("F5_REFERENCE_WAV", "reference.mp3"),
    ref_text=os.getenv("F5_REFERENCE_TEXT"),
    model_name=os.getenv("F5_MODEL_NAME"),
    target_sample_rate=(
        int(os.getenv("F5_TARGET_SAMPLE_RATE"))
        if os.getenv("F5_TARGET_SAMPLE_RATE")
        else None
    ),
)

LOCAL_OPENAI_MODEL = os.getenv("LOCAL_OPENAI_MODEL", "llama-3.1-8b-instruct")


def _should_flush(sentence_buffer: str) -> bool:
    stripped = sentence_buffer.strip()
    if not stripped:
        return False
    if stripped.endswith((".", "!", "?", "…")):
        return True
    if stripped.endswith("\n"):
        return True
    return False


# See "Talk to Claude" in Cookbook for an example of how to keep
# track of the chat history.
def response(*args):
    if not args:
        raise ValueError("response handler received no arguments")

    raw_audio = args[0]
    remaining_args = list(args[1:])

    # FastRTC 0.0.20+ sends an initial "__webrtc_value__" marker before
    # the actual WebRTCData payload is streamed to the handler. Bail out early
    # for that sentinel call so the generator stays alive without raising.
    if isinstance(raw_audio, str) and raw_audio == "__webrtc_value__":
        return

    session_id: str | None = None
    chatbot: list[dict] | None = None

    # The first remaining argument is the session identifier when provided.
    if remaining_args:
        possible_session = remaining_args.pop(0)
        if isinstance(possible_session, str):
            session_id = possible_session
        else:
            remaining_args.insert(0, possible_session)

    # A chatbot history (list[dict]) may be passed either as the second
    # positional argument or via additional inputs.
    for candidate in remaining_args:
        if isinstance(candidate, list):
            chatbot = candidate
            break

    audio_payload: tuple[int, NDArray[np.int16 | np.float32]] | None
    if isinstance(raw_audio, WebRTCData):
        session_id = raw_audio.webrtc_id or session_id
        audio_payload = raw_audio.audio
    else:
        audio_payload = raw_audio

    if audio_payload is None:
        raise ValueError("Missing audio payload in response handler")

    _ = session_id  # session identifier reserved for future use

    chatbot = (chatbot or [])[:]

    audio = audio_payload
    messages = [{"role": d["role"], "content": d["content"]} for d in chatbot]
    start = time.time()
    text = stt_model.stt(audio)
    print("transcription", time.time() - start)
    print("prompt", text)
    chatbot.append({"role": "user", "content": text})
    yield AdditionalOutputs(chatbot)
    messages.append({"role": "user", "content": text})
    completion_stream = openai_client.chat.completions.create(
        model=LOCAL_OPENAI_MODEL,
        max_tokens=512,
        messages=messages,  # type: ignore
        stream=True,
    )

    full_text = ""
    sentence_buffer = ""

    for chunk in completion_stream:
        for choice in chunk.choices:
            token = (choice.delta.content or "") if choice.delta else ""
            if not token:
                continue
            sentence_buffer += token
            full_text += token
            if _should_flush(sentence_buffer):
                for audio_chunk in f5_tts_model.stream_tts_sync(
                    sentence_buffer.strip()
                ):
                    yield audio_chunk
                sentence_buffer = ""

    if sentence_buffer.strip():
        for audio_chunk in f5_tts_model.stream_tts_sync(sentence_buffer.strip()):
            yield audio_chunk

    response_text = " ".join(full_text.strip().split())
    chatbot.append({"role": "assistant", "content": response_text})
    yield AdditionalOutputs(chatbot)


chatbot = gr.Chatbot(type="messages")
stream = Stream(
    modality="audio",
    mode="send-receive",
    handler=ReplyOnPause(response, input_sample_rate=16000),
    additional_outputs_handler=lambda a, b: b,
    additional_inputs=[chatbot],
    additional_outputs=[chatbot],
    rtc_configuration=get_twilio_turn_credentials() if get_space() else None,
    concurrency_limit=5 if get_space() else None,
    time_limit=90 if get_space() else None,
    ui_args={"title": "LLM Voice Chat (Local LLM, Whisper, and F5-TTS ⚡️)"},
)

# Mount the STREAM UI to the FastAPI app
# Because I don't want to build the UI manually
app = FastAPI()
app = gr.mount_gradio_app(app, stream.ui, path="/")


if __name__ == "__main__":
    import os

    os.environ["GRADIO_SSR_MODE"] = "false"

    if (mode := os.getenv("MODE")) == "UI":
        stream.ui.launch(server_port=7860)
    elif mode == "PHONE":
        stream.fastphone(host="0.0.0.0", port=7860)
    else:
        stream.ui.launch(server_port=7860)
