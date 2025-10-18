"""Main entry point for realtime voice chat application."""
from __future__ import annotations

import atexit
import logging
from io import BytesIO
from typing import Generator, Iterable, List, Tuple

import gradio
import gradio as gr
import numpy as np
import soundfile as sf

try:  # pragma: no cover - runtime shim for older gradio_client releases
    import gradio_client  # type: ignore
except Exception:  # pragma: no cover - module missing entirely
    gradio_client = None  # type: ignore
else:
    if not hasattr(gradio_client, "handle_file"):
        def _passthrough_handle_file(file_like: object, *args: object, **kwargs: object) -> object:
            """Fallback for gradio_client.handle_file when running older clients."""

            return file_like

        gradio_client.handle_file = _passthrough_handle_file  # type: ignore[attr-defined]

MIN_GRADIO_VERSION = (4, 44, 0)


def _version_tuple(version_str: str) -> tuple[int, ...]:
    parts: list[int] = []
    for segment in version_str.replace("-", ".").split("."):
        digits = "".join(ch for ch in segment if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


if _version_tuple(gradio.__version__) < MIN_GRADIO_VERSION:  # pragma: no cover - runtime guard
    required = ".".join(str(x) for x in MIN_GRADIO_VERSION)
    raise RuntimeError(
        f"Gradio version {gradio.__version__} is incompatible with FastRTC; install gradio>={required}"
    )


from .config import get_settings
from .llm_client import LLMClient
from .persona import PERSONA_QUESTIONS, PersonaState, build_persona_system_prompt
from .stt_wrapper import get_stt
from .tts.f5_wrapper import F5Cloner, VoiceProfile

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def _summarize_audio(data: np.ndarray, sr: int) -> str:
    samples = data.shape[0]
    channels = 1 if data.ndim == 1 else data.shape[1]
    duration = samples / float(sr)
    return f"Loaded sample: {duration:.2f}s @ {sr}Hz ({channels} channel{'s' if channels != 1 else ''})"


def _decode_wav_chunk(payload: bytes) -> np.ndarray:
    if not payload:
        return np.array([], dtype=np.float32)
    if sf is not None:
        audio, _ = sf.read(BytesIO(payload), dtype="float32")
    else:  # pragma: no cover - fallback path when soundfile missing
        import wave

        with wave.open(BytesIO(payload), "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32767.0
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32, copy=False)


def _build_voice_profile(
    source: VoiceProfile,
    *,
    speaker_wav: np.ndarray | None = None,
    speaker_sr: int | None = None,
    style_notes: str | None = None,
) -> VoiceProfile:
    return VoiceProfile(
        speaker_wav=speaker_wav if speaker_wav is not None else source.speaker_wav,
        speaker_sr=speaker_sr if speaker_sr is not None else source.speaker_sr,
        style_notes=style_notes if style_notes is not None else source.style_notes,
    )


def main(argv: list[str] | None = None) -> None:
    settings = get_settings(argv)
    stt = get_stt()
    tts = F5Cloner(device=settings.device)
    llm = LLMClient(settings.openai_api_key, settings.openai_base_url)
    atexit.register(llm.close)

    with gr.Blocks(title="Realtime Audio Chat") as demo:
        gr.Markdown(
            """
            # Realtime audio assistant
            1. Upload a reference voice clip (or record one) for zero-shot cloning.
            2. Describe persona cues to steer the assistant's responses.
            3. Hold to record, release to send, and hear the cloned reply stream back.
            """
        )

        voice_state = gr.State(VoiceProfile())
        persona_state = gr.State(PersonaState())
        history_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=2):
                conversation = gr.Chatbot(label="Conversation", height=320)
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="Speak to the assistant",
                )
                with gr.Row():
                    submit_button = gr.Button("Send", variant="primary")
                    clear_button = gr.Button("Clear conversation")
                user_text = gr.Textbox(label="Recognized speech", interactive=False)
                assistant_text = gr.Textbox(label="Assistant reply", interactive=False)
                audio_output = gr.Audio(
                    label="Assistant audio", type="numpy", autoplay=True, streaming=True
                )

            with gr.Column():
                gr.Markdown("### Voice cloning")
                reference_audio = gr.Audio(
                    sources=["upload", "microphone"],
                    type="numpy",
                    label="Reference voice sample (2–10s)",
                )
                voice_status = gr.Markdown("Upload a clean clip to enable cloning.")

                def on_reference_audio(
                    audio: Tuple[int, np.ndarray] | None,
                    current: VoiceProfile,
                ) -> Tuple[VoiceProfile, str]:
                    if not audio:
                        return current, "No reference audio provided."
                    sr, data = audio
                    if data is None or getattr(data, "size", 0) == 0:
                        return current, "Reference audio contained no samples."
                    if data.ndim > 1:
                        data = np.mean(data, axis=1)
                    profile = tts.set_voice(data, sr)
                    profile.style_notes = current.style_notes
                    LOGGER.info("Voice sample accepted | sr=%s duration=%.2fs", sr, len(data) / float(sr))
                    summary = _summarize_audio(data, sr)
                    return profile, f"Voice sample loaded.\n\n{summary}"

                reference_audio.change(
                    on_reference_audio,
                    inputs=[reference_audio, voice_state],
                    outputs=[voice_state, voice_status],
                )

                gr.Markdown("### Persona cues")
                persona_inputs: List[gr.Textbox] = []
                for question in PERSONA_QUESTIONS:
                    persona_inputs.append(
                        gr.Textbox(label=question["label"], lines=2, placeholder="Optional")
                    )
                persona_status = gr.Markdown("Provide optional style guidance, then save.")
                save_persona_button = gr.Button("Save persona", variant="primary")

                def on_save_persona(*values: object) -> Tuple[PersonaState, VoiceProfile, str]:
                    *answers_values, persona, voice = values
                    assert isinstance(persona, PersonaState)
                    assert isinstance(voice, VoiceProfile)
                    answers = {
                        question["key"]: value or ""
                        for question, value in zip(PERSONA_QUESTIONS, answers_values)
                    }
                    persona_text = build_persona_system_prompt(answers)
                    updated_persona = PersonaState(style_text=persona_text)
                    updated_voice = _build_voice_profile(voice, style_notes=persona_text)
                    LOGGER.info("Persona updated | chars=%s", len(persona_text))
                    return updated_persona, updated_voice, "Persona saved."

                save_persona_button.click(
                    on_save_persona,
                    inputs=[*persona_inputs, persona_state, voice_state],
                    outputs=[persona_state, voice_state, persona_status],
                )

        def synthesize_stream(
            text: str,
            profile: VoiceProfile,
        ) -> Generator[Tuple[int, np.ndarray], None, None]:
            accumulated = np.array([], dtype=np.float32)
            for chunk in tts.stream_tts_sync(text, profile, settings.f5_output_sr, settings.chunk_ms):
                try:
                    decoded = _decode_wav_chunk(chunk)
                except Exception:  # pragma: no cover - runtime guard
                    LOGGER.exception("Failed to decode generated audio chunk")
                    continue
                if decoded.size == 0:
                    continue
                if accumulated.size == 0:
                    accumulated = decoded
                else:
                    accumulated = np.concatenate((accumulated, decoded))
                yield settings.f5_output_sr, accumulated

        def on_conversation(
            audio: Tuple[int, np.ndarray] | None,
            voice: VoiceProfile,
            persona: PersonaState,
            history: List[Tuple[str, str]],
        ) -> Tuple[List[Tuple[str, str]], str, str, Iterable[Tuple[int, np.ndarray]] | None, List[Tuple[str, str]]]:
            if not audio:
                return history, "", "", None, history
            sr, data = audio
            if data is None or getattr(data, "size", 0) == 0:
                return history, "", "", None, history
            transcription = ""
            try:
                transcription = stt.transcribe((sr, data))
            except Exception:  # pragma: no cover - runtime safety
                LOGGER.exception("STT failure")
                return history, "", "", None, history

            transcription = (transcription or "").strip()
            if not transcription:
                LOGGER.debug("No speech detected")
                return history, "", "", None, history

            LOGGER.info("User said: %s", transcription)
            messages = []
            if persona.style_text:
                messages.append({"role": "system", "content": persona.style_text})
            messages.append({"role": "user", "content": transcription})

            try:
                reply = llm.chat(messages, settings.openai_model)
            except Exception:
                LOGGER.exception("LLM request failed")
                return history, transcription, "LLM request failed.", None, history

            reply_text = (reply or "").strip()
            if not reply_text:
                LOGGER.warning("LLM returned empty reply")
                return history, transcription, "", None, history
            LOGGER.info("Assistant reply: %s", reply_text)

            new_history = history + [(transcription, reply_text)]
            audio_stream = synthesize_stream(reply_text, voice)
            return new_history, transcription, reply_text, audio_stream, new_history

        submit_button.click(
            on_conversation,
            inputs=[audio_input, voice_state, persona_state, history_state],
            outputs=[conversation, user_text, assistant_text, audio_output, history_state],
        )

        def clear_conversation() -> Tuple[List[Tuple[str, str]], str, str, None, List[Tuple[str, str]]]:
            return [], "", "", None, []

        clear_button.click(
            clear_conversation,
            outputs=[conversation, user_text, assistant_text, audio_output, history_state],
        )

    LOGGER.info(
        "Launching Gradio UI | host=%s port=%s share=%s",
        settings.ui_host,
        settings.ui_port,
        settings.ui_share,
    )
    demo.queue().launch(
        server_name=settings.ui_host,
        server_port=settings.ui_port,
        share=settings.ui_share,
        inbrowser=settings.ui_inbrowser,
    )


if __name__ == "__main__":  # pragma: no cover
    import sys

    main(sys.argv[1:])
