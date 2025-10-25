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
from gradio.utils import get_space
import numpy as np
from numpy.typing import NDArray
from openai import OpenAI

from fish_speech_adapter import FishSpeechTTSModel
from src.stt_client import RemoteSTT

load_dotenv()

openai_client = OpenAI(
    api_key=os.getenv("LOCAL_OPENAI_API_KEY", "dummy"),
    base_url=os.getenv("LOCAL_OPENAI_BASE_URL", "http://127.0.0.1:1234/v1"),
)


def get_stt_callable():
    stt_url = os.getenv("STT_URL")
    if stt_url:
        try:
            return RemoteSTT(stt_url).transcribe
        except Exception as exc:
            raise RuntimeError(
                f"Failed to connect to remote STT at {stt_url}. Ensure the STT service is running and reachable."
            ) from exc
    try:
        from whisper_stt_adapter import get_stt_model

        return get_stt_model().stt
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "No STT configured. Set STT_URL to a running STT service or install the local Whisper adapter."
        ) from exc


stt_transcribe = get_stt_callable()

def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _build_inference_kwargs() -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if chunk_length := _parse_optional_int(os.getenv("FISH_SPEECH_CHUNK_LENGTH")):
        kwargs["chunk_length"] = chunk_length
    if max_tokens := _parse_optional_int(os.getenv("FISH_SPEECH_MAX_NEW_TOKENS")):
        kwargs["max_new_tokens"] = max_tokens
    if top_p := os.getenv("FISH_SPEECH_TOP_P"):
        try:
            kwargs["top_p"] = float(top_p)
        except ValueError:
            pass
    if penalty := os.getenv("FISH_SPEECH_REPETITION_PENALTY"):
        try:
            kwargs["repetition_penalty"] = float(penalty)
        except ValueError:
            pass
    if temperature := os.getenv("FISH_SPEECH_TEMPERATURE"):
        try:
            kwargs["temperature"] = float(temperature)
        except ValueError:
            pass
    if seed := _parse_optional_int(os.getenv("FISH_SPEECH_SEED")):
        kwargs["seed"] = seed
    if use_memory_cache := os.getenv("FISH_SPEECH_USE_MEMORY_CACHE"):
        kwargs["use_memory_cache"] = use_memory_cache
    if normalize := os.getenv("FISH_SPEECH_NORMALIZE"):
        kwargs["normalize"] = _parse_bool(normalize, True)
    return kwargs


fish_tts_model = FishSpeechTTSModel(
    ref_wav=os.getenv("F5_REFERENCE_WAV", "morgan.mp3"),
    ref_text=os.getenv("F5_REFERENCE_TEXT"),
    checkpoint_dir=os.getenv("FISH_SPEECH_CHECKPOINT_DIR"),
    download=_parse_bool(os.getenv("FISH_SPEECH_AUTO_DOWNLOAD"), True),
    device=os.getenv("FISH_SPEECH_DEVICE"),
    precision=os.getenv("FISH_SPEECH_PRECISION"),
    compile=_parse_bool(os.getenv("FISH_SPEECH_COMPILE"), False),
    inference_kwargs=_build_inference_kwargs(),
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
def response(
    raw_audio=None,
    maybe_session_or_chatbot=None,
    *extra_args,
    chatbot: list[dict] | None = None,
    **kwargs,
):
    """Handle audio frames from FastRTC with backwards-compatible signatures."""

    if raw_audio is None and not extra_args and maybe_session_or_chatbot is None:
        raise ValueError("response handler received no arguments")

    # FastRTC 0.0.20+ sends an initial "__webrtc_value__" marker before
    # the actual WebRTCData payload is streamed to the handler. Bail out early
    # for that sentinel call so the generator stays alive without raising.
    if isinstance(raw_audio, str) and raw_audio == "__webrtc_value__":
        return

    session_id: str | None = kwargs.get("session_id")
    chat_history: list[dict] | None = kwargs.get("chat_history")

    positional_candidates: list[object] = []
    if maybe_session_or_chatbot is not None:
        positional_candidates.append(maybe_session_or_chatbot)
    positional_candidates.extend(extra_args)

    # Prefer explicitly supplied chatbot keyword argument, otherwise fall back to
    # chat_history kwarg or positional list payloads from FastRTC.
    chatbot = chatbot or chat_history

    # Walk through positional args to collect session identifiers, chatbot
    # histories, or fallback audio payloads for older calling conventions.
    for candidate in positional_candidates:
        if isinstance(candidate, str) and session_id is None:
            session_id = candidate
            continue
        if isinstance(candidate, list) and chatbot is None:
            chatbot = candidate
            continue
        if isinstance(candidate, WebRTCData) and not isinstance(raw_audio, WebRTCData):
            raw_audio = candidate

    audio_payload: tuple[int, NDArray[np.int16 | np.float32]] | None = None
    if isinstance(raw_audio, WebRTCData):
        session_id = raw_audio.webrtc_id or session_id
        audio_payload = raw_audio.audio
    elif isinstance(raw_audio, tuple):
        audio_payload = raw_audio

    if audio_payload is None:
        raise ValueError("Missing audio payload in response handler")

    _ = session_id  # session identifier reserved for future use

    chatbot = (chatbot or [])[:]

    audio = audio_payload
    messages = [{"role": d["role"], "content": d["content"]} for d in chatbot]
    start = time.time()
    text = stt_transcribe(audio)
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
                for audio_chunk in fish_tts_model.stream_tts_sync(
                    sentence_buffer.strip()
                ):
                    yield audio_chunk
                sentence_buffer = ""

    if sentence_buffer.strip():
        for audio_chunk in fish_tts_model.stream_tts_sync(sentence_buffer.strip()):
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
    ui_args={"title": "LLM Voice Chat (Local LLM, Whisper, and Fish-Speech ⚡️)"},
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
