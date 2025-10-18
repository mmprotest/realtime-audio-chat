"""Main entry point for realtime voice chat application."""
from __future__ import annotations

import logging
import threading
from typing import Dict, Generator

import soundfile as sf

from fastrtc import ReplyOnPause, Stream

from .config import get_settings
from .llm_client import LLMClient
from .persona import PersonaState, build_persona_system_prompt
from .stt_wrapper import get_stt
from .tts.f5_wrapper import F5Cloner, VoiceProfile
from .ui.side_panel import build_side_panel

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def main(argv: list[str] | None = None) -> None:
    settings = get_settings(argv)
    stt = get_stt()
    tts = F5Cloner(device=settings.device)
    llm = LLMClient(settings.openai_api_key, settings.openai_base_url)

    persona_state = PersonaState()
    current_voice = VoiceProfile()

    def set_voice_from_path(path: str) -> str:
        data, sr = sf.read(path)
        profile = tts.set_voice(data, sr)
        current_voice.speaker_wav = profile.speaker_wav
        current_voice.speaker_sr = profile.speaker_sr
        LOGGER.info("Voice sample accepted | sr=%s", sr)
        return "Voice sample loaded."

    def save_persona(answers: Dict[str, str]) -> str:
        persona_text = build_persona_system_prompt(answers)
        persona_state.style_text = persona_text
        current_voice.style_notes = persona_text
        LOGGER.info("Persona updated | chars=%s", len(persona_text))
        return "Persona saved. Responses will follow your cues."

    def launch_panel() -> None:
        panel = build_side_panel(set_voice_from_path, save_persona)
        LOGGER.info("Launching side panel on %s:%s", settings.panel_host, settings.panel_port)
        panel.queue().launch(
            server_name=settings.panel_host,
            server_port=settings.panel_port,
            inbrowser=False,
            share=False,
        )

    def echo(audio) -> Generator[bytes, None, None]:
        try:
            text = stt.transcribe(audio)
        except Exception as exc:  # pragma: no cover - runtime safety
            LOGGER.exception("STT failure")
            return
            yield  # pragma: no cover

        if not text:
            LOGGER.debug("No speech detected")
            return

        LOGGER.info("User said: %s", text)
        messages = []
        if persona_state.style_text:
            messages.append({"role": "system", "content": persona_state.style_text})
        messages.append({"role": "user", "content": text})

        try:
            reply = llm.chat(messages, settings.openai_model)
        except Exception:
            LOGGER.exception("LLM request failed")
            return

        reply_text = (reply or "").strip()
        if not reply_text:
            LOGGER.warning("LLM returned empty reply")
            return
        LOGGER.info("Assistant reply: %s", reply_text)
        if current_voice.speaker_wav is None:
            LOGGER.warning("No voice sample provided; using default TTS")
        try:
            for chunk in tts.stream_tts_sync(reply_text, current_voice, settings.f5_output_sr, settings.chunk_ms):
                yield chunk
        except Exception:
            LOGGER.exception("TTS synthesis failed")
            return

    panel_thread = threading.Thread(target=launch_panel, daemon=True)
    panel_thread.start()

    LOGGER.info("Starting FastRTC stream | modality=%s mode=%s", settings.fastrtc_modality, settings.fastrtc_mode)
    stream = Stream(ReplyOnPause(echo), modality=settings.fastrtc_modality, mode=settings.fastrtc_mode)
    stream.ui.launch()


if __name__ == "__main__":  # pragma: no cover
    import sys

    main(sys.argv[1:])
