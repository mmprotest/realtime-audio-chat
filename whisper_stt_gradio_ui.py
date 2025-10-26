"""Simple Gradio UI for recording audio and transcribing via the local Faster Whisper API."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import requests
from requests import Response

DEFAULT_API_URL = "http://localhost:8000"
API_URL = os.getenv("FASTER_WHISPER_API_URL", DEFAULT_API_URL).rstrip("/")


def _fetch_available_models() -> List[str]:
    """Retrieve the list of models exposed by the local API."""
    try:
        response = requests.get(f"{API_URL}/models", timeout=10)
        response.raise_for_status()
    except requests.RequestException:
        return ["medium"]

    try:
        payload = response.json()
    except ValueError:
        return ["medium"]

    models = payload.get("available_models")
    if not isinstance(models, list) or not models:
        return ["medium"]

    return [str(model) for model in models]


def _format_error(message: str) -> Tuple[str, Optional[Dict[str, Any]], str]:
    return "", None, f"❌ {message}"


def transcribe_audio(
    audio_path: Optional[str],
    model_name: str,
    language: str,
    task: str,
    beam_size: int,
) -> Tuple[str, Optional[Dict[str, Any]], str]:
    """Send the recorded audio to the Faster Whisper API for transcription."""
    if not audio_path:
        return _format_error("Please record audio before submitting.")

    form_data: Dict[str, Any] = {
        "model_name": model_name,
        "task": task,
        "beam_size": str(beam_size),
    }

    language = language.strip()
    if language:
        form_data["language"] = language

    try:
        with open(audio_path, "rb") as audio_file:
            files = {"file": (os.path.basename(audio_path), audio_file, "audio/wav")}
            response: Response = requests.post(
                f"{API_URL}/transcribe",
                data=form_data,
                files=files,
                timeout=300,
            )
    except FileNotFoundError:
        return _format_error("Recorded audio file could not be found.")
    except requests.RequestException as exc:
        return _format_error(f"Failed to reach the API: {exc}.")

    if response.status_code != 200:
        try:
            error_payload = response.json()
        except ValueError:
            error_payload = {"detail": response.text}
        return _format_error(f"API returned an error: {error_payload}.")

    try:
        payload = response.json()
    except ValueError:
        return _format_error("Received an unexpected response from the API.")

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


def build_interface() -> gr.Blocks:
    models = _fetch_available_models()
    default_model = models[0] if models else "medium"

    with gr.Blocks(title="Faster Whisper Gradio Client") as demo:
        gr.Markdown(
            """
            # Faster Whisper Local Transcription

            Record audio from your microphone and submit it to the local Faster Whisper API
            for transcription. Configure optional parameters in the panel below.
            """
        )

        with gr.Row():
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Microphone Recording",
                show_download_button=True,
            )

            with gr.Column():
                model_dropdown = gr.Dropdown(
                    choices=models,
                    value=default_model,
                    label="Model",
                )
                language_box = gr.Textbox(
                    label="Language (optional)",
                    placeholder="Leave blank to auto-detect",
                )
                task_radio = gr.Radio(
                    choices=["transcribe", "translate"],
                    value="transcribe",
                    label="Task",
                )
                beam_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=5,
                    label="Beam Size",
                )
                submit_button = gr.Button("Transcribe")

        transcription_output = gr.Textbox(label="Transcription", lines=8)
        metadata_output = gr.JSON(label="Response Metadata")
        status_output = gr.Markdown()

        submit_button.click(
            transcribe_audio,
            inputs=[audio_input, model_dropdown, language_box, task_radio, beam_slider],
            outputs=[transcription_output, metadata_output, status_output],
        )

    return demo


def main() -> None:
    """Entry point for launching the Gradio UI."""
    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()
