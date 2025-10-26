"""Realtime audio chat via FastRTC using local Whisper STT and F5-TTS."""
from __future__ import annotations

import base64
import io
import os
import time
from typing import Iterable, List, Sequence

import gradio as gr
import numpy as np
import requests
import soundfile as sf
from dotenv import load_dotenv
from fastapi import FastAPI
from fastrtc import AdditionalOutputs, ReplyOnPause, Stream, get_twilio_turn_credentials
from gradio.utils import get_space
from numpy.typing import NDArray
from openai import OpenAI

load_dotenv()

# --- Configuration --------------------------------------------------------------------
FASTER_WHISPER_API_URL = os.getenv("FASTER_WHISPER_API_URL", "http://0.0.0.0:8080").rstrip("/")
FASTER_WHISPER_MODEL = os.getenv("FASTER_WHISPER_MODEL", "medium")
FASTER_WHISPER_TASK = os.getenv("FASTER_WHISPER_TASK", "transcribe")
FASTER_WHISPER_LANGUAGE = os.getenv("FASTER_WHISPER_LANGUAGE", "").strip()
FASTER_WHISPER_BEAM_SIZE = int(os.getenv("FASTER_WHISPER_BEAM_SIZE", "5"))
FASTER_WHISPER_TIMEOUT = float(os.getenv("FASTER_WHISPER_TIMEOUT", "120"))

F5_TTS_API_URL = os.getenv("F5_TTS_API_URL", "http://127.0.0.1:8000").rstrip("/")
F5_REFERENCE_AUDIO = os.getenv("F5_REFERENCE_AUDIO", "morgan.mp3")
F5_REFERENCE_TEXT = os.getenv("F5_REFERENCE_TEXT", "")
F5_REMOVE_SILENCE = os.getenv("F5_REMOVE_SILENCE", "false").lower() in {"1", "true", "yes"}
F5_SEED = os.getenv("F5_SEED")
F5_REQUEST_TIMEOUT = float(os.getenv("F5_REQUEST_TIMEOUT", "300"))
F5_CHUNK_DURATION = float(os.getenv("F5_CHUNK_DURATION", "0.5"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","blah")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL","http://127.0.0.1:1234/v1") or None
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "512"))
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required.")

if not F5_REFERENCE_AUDIO or not os.path.exists(F5_REFERENCE_AUDIO):
    raise RuntimeError(
        "F5_REFERENCE_AUDIO environment variable must point to an existing audio file."
    )

if not F5_REFERENCE_TEXT:
    raise RuntimeError("F5_REFERENCE_TEXT environment variable must contain the reference text.")

openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# --- Helpers ---------------------------------------------------------------------------

def _to_wav_bytes(audio: tuple[int, NDArray[np.int16 | np.float32]]) -> io.BytesIO:
    sample_rate, samples = audio
    if samples.ndim == 2:
        samples = samples[0]
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32) / np.iinfo(np.int16).max
    buffer = io.BytesIO()
    sf.write(buffer, samples, samplerate=sample_rate, format="WAV")
    buffer.seek(0)
    return buffer


def transcribe_audio(audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str:
    """Send audio to the local Faster Whisper API and return the transcription."""
    wav_buffer = _to_wav_bytes(audio)
    files = {"file": ("audio.wav", wav_buffer, "audio/wav")}
    data = {
        "model_name": FASTER_WHISPER_MODEL,
        "task": FASTER_WHISPER_TASK,
        "beam_size": str(FASTER_WHISPER_BEAM_SIZE),
    }
    if FASTER_WHISPER_LANGUAGE:
        data["language"] = FASTER_WHISPER_LANGUAGE

    response = requests.post(
        f"{FASTER_WHISPER_API_URL}/transcribe",
        files=files,
        data=data,
        timeout=FASTER_WHISPER_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    transcription = payload.get("transcription")
    if not transcription:
        raise RuntimeError("Faster Whisper API did not return a transcription.")
    return transcription


def _iter_audio_chunks(audio: np.ndarray, sample_rate: int) -> Iterable[NDArray[np.float32]]:
    chunk_size = max(int(sample_rate * F5_CHUNK_DURATION), 1)
    for start in range(0, audio.shape[-1], chunk_size):
        end = start + chunk_size
        yield audio[..., start:end]


def synthesize_speech(text: str) -> tuple[int, Iterable[NDArray[np.float32]]]:
    """Use the F5-TTS API to synthesize speech for the given text."""
    with open(F5_REFERENCE_AUDIO, "rb") as ref_file:
        files = {
            "reference_audio": (
                os.path.basename(F5_REFERENCE_AUDIO) or "reference.wav",
                ref_file,
                "audio/wav",
            )
        }
        data = {
            "reference_text": F5_REFERENCE_TEXT,
            "target_text": text,
            "remove_silence": str(F5_REMOVE_SILENCE).lower(),
        }
        if F5_SEED:
            data["seed"] = F5_SEED
        response = requests.post(
            f"{F5_TTS_API_URL}/infer",
            data=data,
            files=files,
            timeout=F5_REQUEST_TIMEOUT,
        )
    response.raise_for_status()
    payload = response.json()
    encoded_audio = payload.get("audio_base64")
    sample_rate = int(payload.get("sample_rate", 24000))
    if not encoded_audio:
        raise RuntimeError("F5-TTS API response missing audio data.")

    audio_bytes = base64.b64decode(encoded_audio)
    wav, returned_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if returned_sr != sample_rate:
        sample_rate = returned_sr
    if wav.ndim == 1:
        wav = wav.reshape(1, -1)
    else:
        wav = wav.T
    return sample_rate, _iter_audio_chunks(wav, sample_rate)


def generate_response(messages: Sequence[dict[str, str]]) -> str:
    completion = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=list(messages),
        temperature=OPENAI_TEMPERATURE,
        max_tokens=OPENAI_MAX_TOKENS,
    )
    message = completion.choices[0].message
    if message.content is None:
        raise RuntimeError("LLM response did not contain any content.")
    return message.content


# --- FastRTC Handler ------------------------------------------------------------------

def response(
    audio: tuple[int, NDArray[np.int16 | np.float32]] | None,
    chatbot: List[dict[str, str]] | None = None,
    event: object | None = None,
):
    """Handle audio from the user, returning streamed audio + chatbot updates."""

    if event is not None:
        print("ReplyOnPause event:", event)

    if audio is None:
        # FastRTC can invoke the handler with no audio payload during setup/teardown.
        return

    chatbot = list(chatbot or [])
    messages: List[dict[str, str]] = [
        {"role": entry["role"], "content": entry["content"]} for entry in chatbot
    ]

    start = time.time()
    user_text = transcribe_audio(audio)
    print("Transcription latency:", time.time() - start)
    print("User said:", user_text)

    user_message = {"role": "user", "content": user_text}
    chatbot.append(user_message)
    messages.append(user_message)
    yield AdditionalOutputs(chatbot)

    assistant_text = generate_response(messages)
    chatbot.append({"role": "assistant", "content": assistant_text})

    sample_rate, audio_chunks = synthesize_speech(assistant_text)
    for chunk in audio_chunks:
        yield (sample_rate, chunk.astype(np.float32))
    yield AdditionalOutputs(chatbot)


chatbot_component = gr.Chatbot(type="messages")
stream = Stream(
    modality="audio",
    mode="send-receive",
    handler=ReplyOnPause(response, input_sample_rate=16000),
    additional_outputs_handler=lambda audio, updated_history: updated_history,
    additional_inputs=[chatbot_component],
    additional_outputs=[chatbot_component],
    rtc_configuration=get_twilio_turn_credentials() if get_space() else None,
    concurrency_limit=5 if get_space() else None,
    time_limit=90 if get_space() else None,
    ui_args={"title": "Realtime Audio Chat (Whisper STT + F5-TTS + OpenAI-compatible LLM)"},
)

app = FastAPI()
app = gr.mount_gradio_app(app, stream.ui, path="/")


if __name__ == "__main__":
    os.environ.setdefault("GRADIO_SSR_MODE", "false")
    mode = os.getenv("MODE", "UI").upper()
    if mode == "PHONE":
        stream.fastphone(host="0.0.0.0", port=7860)
    else:
        stream.ui.launch(server_port=7860)
