"""Gradio side panel for voice upload and persona settings."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict

import gradio as gr
import soundfile as sf

from ..persona import PERSONA_QUESTIONS

LOGGER = logging.getLogger(__name__)


VoiceSetter = Callable[[str], str]
PersonaSaver = Callable[[Dict[str, str]], str]


def _read_audio_summary(path: Path) -> str:
    data, sr = sf.read(path)
    duration = len(data) / float(sr)
    channels = 1 if data.ndim == 1 else data.shape[1]
    return f"Loaded sample: {duration:.2f}s @ {sr}Hz ({channels} channel{'s' if channels != 1 else ''})"


def build_side_panel(
    set_voice_callback: VoiceSetter,
    save_persona_callback: PersonaSaver,
) -> gr.Blocks:
    """Create the Gradio Blocks side panel."""

    with gr.Blocks(title="Realtime Voice Persona") as demo:
        gr.Markdown("## Voice profile & persona")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Voice sample")
                audio_input = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    label="Upload 2–10s clean voice clip",
                )
                voice_status = gr.Markdown("Upload a sample to begin.")

                def handle_audio(path: str) -> str:
                    if not path:
                        return "No audio provided."
                    try:
                        message = set_voice_callback(path)
                        summary = _read_audio_summary(Path(path))
                        return f"{message}\n\n{summary}"
                    except Exception as exc:  # pragma: no cover - UI safety
                        LOGGER.exception("Failed to process uploaded audio")
                        return f"Error processing audio: {exc}"

                audio_input.change(handle_audio, inputs=audio_input, outputs=voice_status)
            with gr.Column():
                gr.Markdown("### Persona cues")
                persona_inputs = {}
                for question in PERSONA_QUESTIONS:
                    persona_inputs[question["key"]] = gr.Textbox(label=question["label"], lines=2)
                save_button = gr.Button("Save Persona", variant="primary")
                persona_status = gr.Markdown("Fill in any fields you like, then save.")

                def on_save(*values: str) -> str:
                    answers = {q["key"]: value for q, value in zip(PERSONA_QUESTIONS, values)}
                    try:
                        return save_persona_callback(answers)
                    except Exception as exc:  # pragma: no cover - UI safety
                        LOGGER.exception("Failed to save persona")
                        return f"Error saving persona: {exc}"

                save_button.click(
                    on_save,
                    inputs=list(persona_inputs.values()),
                    outputs=persona_status,
                )
        gr.Markdown(
            """Tip: keep persona instructions concise. Uploading a longer sample may impact latency."""
        )
    return demo


__all__ = ["build_side_panel"]
