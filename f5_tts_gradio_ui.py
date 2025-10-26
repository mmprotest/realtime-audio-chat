"""Gradio interface that delegates synthesis to the local FastAPI server.

This UI acts as a lightweight client for ``serve_api.py``. Rather than loading
models directly, it forwards requests to the running REST API so users can
validate the server end-to-end and listen to the responses inside the browser.
"""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import requests
import soundfile as sf

def synthesize(
    reference_audio: Optional[str],
    reference_text: str,
    target_text: str,
    seed: Optional[float],
    remove_silence: bool,
    api_url: str,
    request_timeout: float,
) -> Tuple[int, np.ndarray]:
    """Send inference request to the REST API and return audio for playback."""

    if not reference_audio:
        raise gr.Error("Please upload a reference audio sample (wav format recommended).")
    if not reference_text.strip():
        raise gr.Error("Reference text cannot be empty.")
    if not target_text.strip():
        raise gr.Error("Please enter text to synthesize.")

    seed_value: Optional[int]
    if seed is None or seed == "" or (isinstance(seed, float) and np.isnan(seed)):
        seed_value = None
    else:
        try:
            seed_value = int(seed)
        except (TypeError, ValueError) as exc:
            raise gr.Error("Seed must be an integer value.") from exc

    infer_endpoint = api_url.rstrip("/") + "/infer"

    try:
        with Path(reference_audio).open("rb") as ref_file:
            files = {
                "reference_audio": (
                    Path(reference_audio).name or "reference.wav",
                    ref_file,
                    "audio/wav",
                )
            }
            data = {
                "reference_text": reference_text,
                "target_text": target_text,
                "remove_silence": str(bool(remove_silence)).lower(),
            }
            if seed_value is not None:
                data["seed"] = str(seed_value)

            response = requests.post(
                infer_endpoint,
                data=data,
                files=files,
                timeout=request_timeout,
            )
    except OSError as exc:
        raise gr.Error(f"Failed to read reference audio: {exc}") from exc
    except requests.RequestException as exc:
        raise gr.Error(f"API request failed: {exc}") from exc

    if response.status_code != 200:
        try:
            error_detail = response.json()
        except ValueError:
            error_detail = response.text
        raise gr.Error(f"API returned {response.status_code}: {error_detail}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise gr.Error("Failed to decode API response JSON.") from exc

    encoded_audio = payload.get("audio_base64")
    sample_rate = payload.get("sample_rate")
    if not encoded_audio or sample_rate is None:
        raise gr.Error("API response missing audio data.")

    try:
        audio_bytes = base64.b64decode(encoded_audio)
        wav, returned_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except (base64.binascii.Error, RuntimeError) as exc:
        raise gr.Error("Failed to decode audio from API response.") from exc

    if returned_sr != sample_rate:
        sample_rate = returned_sr

    audio = np.asarray(wav, dtype=np.float32)
    return sample_rate, audio


def build_interface(args: argparse.Namespace) -> gr.Blocks:
    """Create the Gradio Blocks UI for text-to-speech synthesis."""

    with gr.Blocks(title="F5-TTS Gradio Demo") as demo:
        gr.Markdown(
            """
            # F5-TTS Interactive Demo

            Upload a reference audio file and its transcript, then provide new text
            to synthesize using the same voice.
            """
        )

        with gr.Row():
            reference_audio = gr.Audio(
                label="Reference Audio",
                sources=["upload"],
                type="filepath",
                interactive=True,
            )
            seed_input = gr.Number(label="Seed", value=None)
        reference_text = gr.Textbox(label="Reference Text", lines=3)
        target_text = gr.Textbox(label="Target Text", lines=3)
        remove_silence = gr.Checkbox(label="Remove Silence", value=False)

        output_audio = gr.Audio(label="Generated Audio", type="numpy")
        submit = gr.Button("Synthesize")

        submit.click(
            synthesize,
            inputs=[
                reference_audio,
                reference_text,
                target_text,
                seed_input,
                remove_silence,
                gr.State(args.api_url),
                gr.State(args.request_timeout),
            ],
            outputs=[output_audio],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Gradio UI for F5-TTS inference")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="Base URL of the running F5-TTS API")
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=300.0,
        help="Timeout (in seconds) for API requests",
    )
    parser.add_argument("--share", action="store_true", help="Share the Gradio app publicly")
    parser.add_argument("--server-name", default="127.0.0.1", help="Host address for the Gradio server")
    parser.add_argument("--server-port", type=int, default=7860, help="Port number for the Gradio server")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_interface(args)
    demo.launch(share=args.share, server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
