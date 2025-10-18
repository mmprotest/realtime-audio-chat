"""Main entry point for realtime voice chat application."""
from __future__ import annotations

import atexit
import logging
from dataclasses import dataclass
from io import BytesIO
from typing import List, Tuple

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


try:
    from fastrtc import ReplyOnPause, Stream
except ModuleNotFoundError as exc:  # pragma: no cover - import-time guard
    if exc.name == "numpy.typing":
        raise ModuleNotFoundError(
            "FastRTC requires numpy>=1.20 to expose numpy.typing."
            " Install or upgrade numpy with 'pip install "
            "\"numpy>=1.20\"'' before running the app."
        ) from exc
    raise

from .config import get_settings
from .llm_client import LLMClient
from .persona import PERSONA_QUESTIONS, PersonaState, build_persona_system_prompt
from .stt_wrapper import get_stt
from .tts.f5_wrapper import F5Cloner, VoiceProfile

try:  # Optional import for pause detection extras
    from fastrtc.pause_detection import SileroVadOptions, get_silero_model
except Exception:  # pragma: no cover - runtime guard when extras missing
    SileroVadOptions = None  # type: ignore[assignment]
    get_silero_model = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

if get_silero_model is None or SileroVadOptions is None:  # pragma: no cover - runtime guard
    raise RuntimeError(
        "FastRTC pause detection extras are required. Install fastrtc[vad] to enable realtime silence detection.",
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


@dataclass
class PauseAlgoOptions:
    """Algorithm tuning parameters for pause detection."""

    audio_chunk_duration: float = 0.6
    started_talking_threshold: float = 0.2
    speech_threshold: float = 0.1
    max_continuous_speech_s: float = float("inf")


@dataclass
class PauseState:
    """Runtime state for the pause detector."""

    buffer: np.ndarray | None = None
    stream: np.ndarray | None = None
    sampling_rate: int = 0
    started_talking: bool = False


class PauseDetector:
    """Lightweight wrapper around FastRTC's Silero VAD model."""

    def __init__(
        self,
        algo_options: PauseAlgoOptions | None = None,
        model_options: SileroVadOptions | None = None,
    ) -> None:
        if get_silero_model is None or SileroVadOptions is None:  # pragma: no cover - runtime guard
            raise RuntimeError(
                "FastRTC pause detection extras are required. Install fastrtc[vad] to enable silence detection.",
            )
        self._algo = algo_options or PauseAlgoOptions()
        self._model_options = model_options or SileroVadOptions()
        self._model = get_silero_model()
        self._state = PauseState()

    @staticmethod
    def _to_mono(data: np.ndarray) -> np.ndarray:
        array = np.asarray(data)
        if array.ndim > 1:
            array = np.mean(array, axis=1)
        return array.astype(np.float32, copy=False)

    @staticmethod
    def _resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return data
        try:  # pragma: no cover - optional dependency
            import librosa

            resampled = librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
            return resampled.astype(np.float32, copy=False)
        except Exception:
            # Fall back to linear interpolation when librosa is unavailable.
            ratio = target_sr / float(orig_sr)
            if ratio == 0:
                return data
            x_old = np.arange(data.size, dtype=np.float32)
            x_new = np.arange(0, data.size * ratio, 1, dtype=np.float32) / ratio
            resampled = np.interp(x_new[: int(data.size * ratio)], x_old, data)
            return resampled.astype(np.float32, copy=False)

    def reset(self) -> None:
        self._state = PauseState()

    def _current_rate(self, sr: int) -> int:
        if not self._state.sampling_rate:
            self._state.sampling_rate = sr
        return self._state.sampling_rate

    def _update_buffer(self, sr: int, data: np.ndarray) -> None:
        target_sr = self._current_rate(sr)
        if sr != target_sr:
            data = self._resample(data, sr, target_sr)
        if self._state.buffer is None:
            self._state.buffer = data
        else:
            self._state.buffer = np.concatenate((self._state.buffer, data))

    def _should_emit(self) -> bool:
        buffer = self._state.buffer
        sr = self._state.sampling_rate
        if buffer is None or buffer.size == 0 or sr == 0:
            return False
        duration = buffer.size / float(sr)
        if duration < self._algo.audio_chunk_duration:
            return False
        speech_duration, _ = self._model.vad((sr, buffer), self._model_options)
        if speech_duration > self._algo.started_talking_threshold and not self._state.started_talking:
            self._state.started_talking = True
        if self._state.started_talking:
            if self._state.stream is None:
                self._state.stream = buffer.copy()
            else:
                self._state.stream = np.concatenate((self._state.stream, buffer))
            current_duration = self._state.stream.size / float(sr)
            if current_duration >= self._algo.max_continuous_speech_s:
                self._state.buffer = None
                return True
        self._state.buffer = None
        if self._state.started_talking and speech_duration < self._algo.speech_threshold:
            return True
        return False

    def accept(self, audio: Tuple[int, np.ndarray]) -> Tuple[bool, Tuple[int, np.ndarray] | None]:
        sr, data = audio
        mono = self._to_mono(data)
        if mono.size == 0:
            return False, None
        self._update_buffer(sr, mono)
        if not self._should_emit():
            return False, None
        if self._state.stream is None:
            payload = np.array([], dtype=np.float32)
        else:
            payload = self._state.stream.astype(np.float32, copy=False)
        sample_rate = self._state.sampling_rate
        self.reset()
        return True, (sample_rate, payload)

    def flush(self) -> Tuple[int, np.ndarray] | None:
        if self._state.stream is not None and self._state.stream.size > 0:
            payload = self._state.stream.astype(np.float32, copy=False)
            sr = self._state.sampling_rate
            self.reset()
            return sr, payload
        if (
            self._state.started_talking
            and self._state.buffer is not None
            and self._state.buffer.size > 0
        ):
            payload = self._state.buffer.astype(np.float32, copy=False)
            sr = self._state.sampling_rate
            self.reset()
            return sr, payload
        self.reset()
        return None

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
            3. Press record to speak — silence detection will transcribe and respond automatically.
            """
        )

        voice_state = gr.State(VoiceProfile())
        persona_state = gr.State(PersonaState())
        history_state = gr.State([])
        pause_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=2):
                conversation = gr.Chatbot(label="Conversation", height=320, type="messages")
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="Realtime microphone",
                    streaming=True,
                    show_download_button=False,
                )
                detection_status = gr.Markdown(
                    "Click the microphone to start speaking. We'll listen for pauses to respond."
                )
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

        def synthesize_audio(
            text: str,
            profile: VoiceProfile,
        ) -> Tuple[int, np.ndarray] | None:
            chunks: List[np.ndarray] = []
            for chunk in tts.stream_tts_sync(text, profile, settings.f5_output_sr, settings.chunk_ms):
                try:
                    decoded = _decode_wav_chunk(chunk)
                except Exception:  # pragma: no cover - runtime guard
                    LOGGER.exception("Failed to decode generated audio chunk")
                    continue
                if decoded.size == 0:
                    continue
                chunks.append(decoded)
            if not chunks:
                return None
            combined = np.concatenate(chunks).astype(np.float32, copy=False)
            return settings.f5_output_sr, combined

        def process_turn(
            audio: Tuple[int, np.ndarray] | None,
            voice: VoiceProfile,
            persona: PersonaState,
            history: List[Tuple[str, str]],
        ) -> Tuple[List[Tuple[str, str]], str, str, Tuple[int, np.ndarray] | None, List[Tuple[str, str]]]:
            if not audio:
                return history, "", "", None, history
            sr, data = audio
            if sr <= 0:
                return history, "", "", None, history
            array = np.asarray(data)
            if array.size == 0:
                return history, "", "", None, history
            if array.ndim > 1:
                array = np.mean(array, axis=1)
            array = array.astype(np.float32, copy=False)
            try:
                transcription = stt.transcribe((sr, array))
            except Exception:  # pragma: no cover - runtime safety
                LOGGER.exception("STT failure")
                return history, "", "Speech recognition failed.", None, history

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
            audio_result = synthesize_audio(reply_text, voice)
            return new_history, transcription, reply_text, audio_result, new_history

        def on_audio_stream(
            audio: Tuple[int, np.ndarray] | None,
            voice: VoiceProfile,
            persona: PersonaState,
            history: List[Tuple[str, str]],
            detector: PauseDetector | None,
        ):
            if detector is None:
                detector = PauseDetector()
            if audio is None:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    history,
                    detector,
                )
            emitted, payload = detector.accept(audio)
            if not emitted or payload is None:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    history,
                    detector,
                )
            history_out, transcript, reply_text, audio_stream, updated_history = process_turn(
                payload,
                voice,
                persona,
                history,
            )
            return (
                history_out,
                transcript,
                reply_text,
                audio_stream,
                updated_history,
                detector,
            )

        def on_recording_stop(
            voice: VoiceProfile,
            persona: PersonaState,
            history: List[Tuple[str, str]],
            detector: PauseDetector | None,
        ):
            if detector is None:
                detector = PauseDetector()
            payload = detector.flush()
            if not payload:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    history,
                    detector,
                )
            history_out, transcript, reply_text, audio_stream, updated_history = process_turn(
                payload,
                voice,
                persona,
                history,
            )
            return (
                history_out,
                transcript,
                reply_text,
                audio_stream,
                updated_history,
                detector,
            )

        def clear_conversation(
            detector: PauseDetector | None,
        ) -> Tuple[List[Tuple[str, str]], str, str, None, List[Tuple[str, str]], PauseDetector]:
            if detector is None:
                detector = PauseDetector()
            detector.reset()
            return [], "", "", None, [], detector

        audio_input.stream(
            on_audio_stream,
            inputs=[audio_input, voice_state, persona_state, history_state, pause_state],
            outputs=[conversation, user_text, assistant_text, audio_output, history_state, pause_state],
        )

        audio_input.stop_recording(
            on_recording_stop,
            inputs=[voice_state, persona_state, history_state, pause_state],
            outputs=[conversation, user_text, assistant_text, audio_output, history_state, pause_state],
        )

        clear_button.click(
            clear_conversation,
            inputs=[pause_state],
            outputs=[conversation, user_text, assistant_text, audio_output, history_state, pause_state],
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
