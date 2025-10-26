"""Realtime audio chat via FastRTC using local Whisper STT and F5-TTS."""
from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

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
FASTER_WHISPER_API_URL = os.getenv("FASTER_WHISPER_API_URL", "http://localhost:8080").rstrip("/")
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
    F5_REFERENCE_AUDIO = ""

if not F5_REFERENCE_TEXT:
    F5_REFERENCE_TEXT = ""

def _convert_env_seed(seed: str | None) -> int | None:
    try:
        return int(seed) if seed not in (None, "") else None
    except (TypeError, ValueError):
        return None


DEFAULT_VOICE_SETTINGS: Dict[str, Any] = {
    "api_url": F5_TTS_API_URL,
    "request_timeout": F5_REQUEST_TIMEOUT,
    "reference_audio": F5_REFERENCE_AUDIO,
    "reference_text": F5_REFERENCE_TEXT,
    "remove_silence": F5_REMOVE_SILENCE,
    "seed": _convert_env_seed(F5_SEED),
}

DEFAULT_STT_SETTINGS: Dict[str, Any] = {
    "api_url": FASTER_WHISPER_API_URL,
    "model_name": FASTER_WHISPER_MODEL,
    "task": FASTER_WHISPER_TASK,
    "language": FASTER_WHISPER_LANGUAGE,
    "beam_size": FASTER_WHISPER_BEAM_SIZE,
    "timeout": FASTER_WHISPER_TIMEOUT,
}

openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# --- Helpers ---------------------------------------------------------------------------


def _merge_voice_settings(overrides: Dict[str, Any] | None) -> Dict[str, Any]:
    settings = dict(DEFAULT_VOICE_SETTINGS)
    if overrides:
        for key, value in overrides.items():
            if value not in (None, ""):
                settings[key] = value
    return settings


def _merge_stt_settings(overrides: Dict[str, Any] | None) -> Dict[str, Any]:
    settings = dict(DEFAULT_STT_SETTINGS)
    if overrides:
        for key, value in overrides.items():
            if value not in (None, ""):
                settings[key] = value
    return settings


def _parse_seed_value(seed: Any) -> int | None:
    if seed in (None, ""):
        return None
    try:
        if isinstance(seed, str) and seed.strip() == "":
            return None
        return int(float(seed))
    except (TypeError, ValueError) as exc:
        raise gr.Error("Seed must be an integer value.") from exc


def _validate_reference_audio(path: str | None) -> str:
    if not path:
        raise gr.Error(
            "Please upload a reference audio sample (wav format recommended) or set F5_REFERENCE_AUDIO."
        )
    resolved = Path(path)
    if not resolved.exists():
        raise gr.Error(f"Reference audio file not found: {path}")
    return str(resolved)

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


def transcribe_audio_with_settings(
    audio: tuple[int, NDArray[np.int16 | np.float32]],
    stt_settings: Dict[str, Any] | None,
) -> str:
    """Send audio to the Faster Whisper API using the provided settings."""

    settings = _merge_stt_settings(stt_settings)
    wav_buffer = _to_wav_bytes(audio)
    files = {"file": ("audio.wav", wav_buffer, "audio/wav")}
    data: Dict[str, str] = {
        "model_name": str(settings["model_name"]),
        "task": str(settings["task"]),
        "beam_size": str(int(settings["beam_size"])),
    }
    language = str(settings.get("language", "")).strip()
    if language:
        data["language"] = language

    response = requests.post(
        f"{settings['api_url']}/transcribe",
        files=files,
        data=data,
        timeout=float(settings["timeout"]),
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


def synthesize_speech_with_settings(
    text: str, voice_settings: Dict[str, Any] | None
) -> tuple[int, Iterable[NDArray[np.float32]]]:
    """Use the F5-TTS API to synthesize speech for the given text and settings."""

    settings = _merge_voice_settings(voice_settings)
    reference_audio = _validate_reference_audio(settings.get("reference_audio"))
    reference_text = settings.get("reference_text", "").strip()
    if not reference_text:
        raise gr.Error("Reference text cannot be empty.")

    seed_value = settings.get("seed")
    if seed_value not in (None, ""):
        seed_value = _parse_seed_value(seed_value)

    with open(reference_audio, "rb") as ref_file:
        files = {
            "reference_audio": (
                os.path.basename(reference_audio) or "reference.wav",
                ref_file,
                "audio/wav",
            )
        }
        data: Dict[str, str] = {
            "reference_text": reference_text,
            "target_text": text,
            "remove_silence": str(bool(settings.get("remove_silence", False))).lower(),
        }
        if seed_value is not None:
            data["seed"] = str(seed_value)
        response = requests.post(
            f"{settings['api_url']}/infer",
            data=data,
            files=files,
            timeout=float(settings["request_timeout"]),
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


def synthesize_preview_audio(
    target_text: str, voice_settings: Dict[str, Any] | None
) -> tuple[int, np.ndarray]:
    sample_rate, chunks = synthesize_speech_with_settings(target_text, voice_settings)
    audio_arrays = [chunk.astype(np.float32) for chunk in chunks]
    if not audio_arrays:
        return sample_rate, np.zeros((1, 0), dtype=np.float32)
    concatenated = np.concatenate(audio_arrays, axis=-1)
    return sample_rate, concatenated


def _fetch_available_models(api_url: str) -> List[str]:
    try:
        response = requests.get(f"{api_url}/models", timeout=10)
        response.raise_for_status()
    except requests.RequestException:
        return [DEFAULT_STT_SETTINGS["model_name"]]

    try:
        payload = response.json()
    except ValueError:
        return [DEFAULT_STT_SETTINGS["model_name"]]

    models = payload.get("available_models")
    if not isinstance(models, list) or not models:
        return [DEFAULT_STT_SETTINGS["model_name"]]

    return [str(model) for model in models]


def transcribe_uploaded_audio(
    audio_path: str | None,
    model_name: str,
    language: str,
    task: str,
    beam_size: int,
    stt_state: Dict[str, Any] | None,
) -> tuple[str, Dict[str, Any] | None, str]:
    if not audio_path:
        return "", None, "❌ Please record audio before submitting."

    settings = _merge_stt_settings(stt_state)
    form_data: Dict[str, str] = {
        "model_name": model_name,
        "task": task,
        "beam_size": str(int(beam_size)),
    }

    language = language.strip()
    if language:
        form_data["language"] = language

    try:
        with open(audio_path, "rb") as audio_file:
            files = {"file": (os.path.basename(audio_path), audio_file, "audio/wav")}
            response = requests.post(
                f"{settings['api_url']}/transcribe",
                data=form_data,
                files=files,
                timeout=float(settings["timeout"]),
            )
    except FileNotFoundError:
        return "", None, "❌ Recorded audio file could not be found."
    except requests.RequestException as exc:
        return "", None, f"❌ Failed to reach the API: {exc}."

    if response.status_code != 200:
        try:
            error_payload = response.json()
        except ValueError:
            error_payload = {"detail": response.text}
        return "", None, f"❌ API returned an error: {error_payload}."

    try:
        payload = response.json()
    except ValueError:
        return "", None, "❌ Received an unexpected response from the API."

    transcription = payload.get("transcription", "")
    status_message = "✅ Transcription completed successfully."
    metadata = {
        key: payload.get(key)
        for key in [
            "model",
            "language",
            "language_probability",
            "duration",
            "segments",
        ]
        if key in payload
    }

    return transcription, metadata, status_message


def update_voice_settings_and_preview(
    reference_audio: str | None,
    reference_text: str,
    target_text: str,
    seed: Any,
    remove_silence: bool,
    state: Dict[str, Any] | None,
) -> tuple[tuple[int, np.ndarray], Dict[str, Any]]:
    settings = _merge_voice_settings(state)
    if reference_audio:
        settings["reference_audio"] = reference_audio
    if reference_text.strip():
        settings["reference_text"] = reference_text.strip()
    else:
        raise gr.Error("Reference text cannot be empty.")
    settings["remove_silence"] = bool(remove_silence)
    settings["seed"] = _parse_seed_value(seed)

    target_text = target_text.strip()
    if not target_text:
        raise gr.Error("Please enter text to synthesize.")

    sample_rate, audio = synthesize_preview_audio(target_text, settings)
    return (sample_rate, audio), settings


def update_stt_settings_and_transcribe(
    audio_path: str | None,
    model_name: str,
    language: str,
    task: str,
    beam_size: int,
    state: Dict[str, Any] | None,
) -> tuple[str, Dict[str, Any] | None, str, Dict[str, Any]]:
    settings = _merge_stt_settings(state)
    settings.update(
        {
            "model_name": model_name,
            "language": language,
            "task": task,
            "beam_size": int(beam_size),
        }
    )
    transcription, metadata, status = transcribe_uploaded_audio(
        audio_path,
        model_name,
        language,
        task,
        int(beam_size),
        settings,
    )
    return transcription, metadata, status, settings


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

def _normalize_chat_history(
    history: object | None,
) -> List[dict[str, str]]:
    """Coerce the incoming chat history into a list of ``{"role", "content"}`` dicts."""

    if history is None:
        return []

    if isinstance(history, str):
        # FastRTC/Gradio can sometimes pass serialized JSON instead of python objects.
        try:
            history = json.loads(history)
        except json.JSONDecodeError:
            return []

    if isinstance(history, dict):
        # Some browser clients may send a dict containing the messages list.
        history = history.get("messages") or history.get("data") or []

    normalized: List[dict[str, str]] = []
    if isinstance(history, list):
        for entry in history:
            if isinstance(entry, dict):
                role = entry.get("role")
                content = entry.get("content")
                if isinstance(content, dict):
                    # Gradio 4 can wrap message content with a ``{"text": {"value": ...}}`` structure.
                    content = content.get("text") or content.get("value")
                    if isinstance(content, dict):
                        content = content.get("value")
                if isinstance(content, list):
                    # Join list-based content into a plain string for the LLM.
                    content = " ".join(str(part) for part in content)
                if role and isinstance(content, str):
                    normalized.append({"role": role, "content": content})
            elif isinstance(entry, (tuple, list)) and len(entry) == 2:
                # Legacy chatbot format of ``[(user, assistant), ...]``.
                user, assistant = entry
                if user is not None:
                    normalized.append({"role": "user", "content": str(user)})
                if assistant is not None:
                    normalized.append({"role": "assistant", "content": str(assistant)})
    return normalized


def response(
    audio: tuple[int, NDArray[np.int16 | np.float32]] | None,
    chatbot: List[dict[str, str]] | str | dict | None = None,
    voice_settings: Dict[str, Any] | None = None,
    stt_settings: Dict[str, Any] | None = None,
    event: object | None = None,
):
    """Handle audio from the user, returning streamed audio + chatbot updates."""

    if event is not None:
        print("ReplyOnPause event:", event)

    if audio is None:
        # FastRTC can invoke the handler with no audio payload during setup/teardown.
        return

    chatbot = _normalize_chat_history(chatbot)
    messages: List[dict[str, str]] = list(chatbot)

    start = time.time()
    user_text = transcribe_audio_with_settings(audio, stt_settings)
    print("Transcription latency:", time.time() - start)
    print("User said:", user_text)

    user_message = {"role": "user", "content": user_text}
    chatbot.append(user_message)
    messages.append(user_message)
    yield AdditionalOutputs(chatbot)

    assistant_text = generate_response(messages)
    chatbot.append({"role": "assistant", "content": assistant_text})

    sample_rate, audio_chunks = synthesize_speech_with_settings(
        assistant_text, voice_settings
    )
    for chunk in audio_chunks:
        yield (sample_rate, chunk.astype(np.float32))
    yield AdditionalOutputs(chatbot)


chatbot_component = gr.Chatbot(type="messages")
voice_settings_state = gr.State(dict(DEFAULT_VOICE_SETTINGS))
stt_settings_state = gr.State(dict(DEFAULT_STT_SETTINGS))

stream = Stream(
    modality="audio",
    mode="send-receive",
    handler=ReplyOnPause(response, input_sample_rate=16000),
    additional_outputs_handler=lambda audio, updated_history: updated_history,
    additional_inputs=[chatbot_component, voice_settings_state, stt_settings_state],
    additional_outputs=[chatbot_component],
    rtc_configuration=get_twilio_turn_credentials() if get_space() else None,
    concurrency_limit=5 if get_space() else None,
    time_limit=90 if get_space() else None,
    ui_args={"title": "Realtime Audio Chat (Whisper STT + F5-TTS + OpenAI-compatible LLM)"},
)


with stream.ui:
    with gr.Accordion("F5-TTS Voice Cloning", open=False):
        gr.Markdown(
            """
            # F5-TTS Interactive Demo

            Upload a reference audio file and its transcript, then provide new text
            to synthesize using the same voice. These settings are also used for the realtime chat assistant.
            """
        )
        with gr.Row():
            voice_reference_audio = gr.Audio(
                label="Reference Audio",
                sources=["upload"],
                type="filepath",
                value=DEFAULT_VOICE_SETTINGS["reference_audio"] or None,
                interactive=True,
            )
            voice_seed_input = gr.Number(
                label="Seed",
                value=DEFAULT_VOICE_SETTINGS["seed"],
            )
        voice_reference_text = gr.Textbox(
            label="Reference Text",
            lines=3,
            value=DEFAULT_VOICE_SETTINGS["reference_text"],
        )
        voice_target_text = gr.Textbox(
            label="Target Text",
            lines=3,
            placeholder="Enter a sample phrase to preview the cloned voice",
        )
        voice_remove_silence = gr.Checkbox(
            label="Remove Silence",
            value=bool(DEFAULT_VOICE_SETTINGS["remove_silence"]),
        )
        voice_output_audio = gr.Audio(label="Generated Audio", type="numpy")
        voice_submit = gr.Button("Synthesize")

        voice_submit.click(
            update_voice_settings_and_preview,
            inputs=[
                voice_reference_audio,
                voice_reference_text,
                voice_target_text,
                voice_seed_input,
                voice_remove_silence,
                voice_settings_state,
            ],
            outputs=[voice_output_audio, voice_settings_state],
        )

    with gr.Accordion("Faster Whisper STT Tester", open=False):
        gr.Markdown(
            """
            # Faster Whisper Local Transcription

            Record audio from your microphone and submit it to the local Faster Whisper API
            for transcription. Configure optional parameters in the panel below. The realtime chat
            will use these parameters for speech recognition.
            """
        )

        models = _fetch_available_models(DEFAULT_STT_SETTINGS["api_url"])
        default_model = (
            DEFAULT_STT_SETTINGS["model_name"]
            if DEFAULT_STT_SETTINGS["model_name"] in models
            else models[0]
        )

        with gr.Row():
            stt_audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Microphone Recording",
                show_download_button=True,
            )
            with gr.Column():
                stt_model_dropdown = gr.Dropdown(
                    choices=models,
                    value=default_model,
                    label="Model",
                )
                stt_language_box = gr.Textbox(
                    label="Language (optional)",
                    value=DEFAULT_STT_SETTINGS["language"],
                    placeholder="Leave blank to auto-detect",
                )
                stt_task_radio = gr.Radio(
                    choices=["transcribe", "translate"],
                    value=DEFAULT_STT_SETTINGS["task"],
                    label="Task",
                )
                stt_beam_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=int(DEFAULT_STT_SETTINGS["beam_size"]),
                    label="Beam Size",
                )
                stt_submit = gr.Button("Transcribe")

        stt_transcription_output = gr.Textbox(label="Transcription", lines=8)
        stt_metadata_output = gr.JSON(label="Response Metadata")
        stt_status_output = gr.Markdown()

        stt_submit.click(
            update_stt_settings_and_transcribe,
            inputs=[
                stt_audio_input,
                stt_model_dropdown,
                stt_language_box,
                stt_task_radio,
                stt_beam_slider,
                stt_settings_state,
            ],
            outputs=[
                stt_transcription_output,
                stt_metadata_output,
                stt_status_output,
                stt_settings_state,
            ],
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
