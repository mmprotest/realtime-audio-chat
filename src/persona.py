"""Persona management utilities for shaping the assistant."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


PERSONA_QUESTIONS: List[dict[str, str]] = [
    {"key": "tone", "label": "Conversational tone"},
    {"key": "formality", "label": "Formality level"},
    {"key": "pacing", "label": "Pacing"},
    {"key": "vocabulary", "label": "Vocabulary"},
    {"key": "temperament", "label": "Temperament"},
    {"key": "filler", "label": "Filler/Discourse"},
    {"key": "directness", "label": "Directness"},
    {"key": "cultural_cues", "label": "Cultural cues"},
]


@dataclass(slots=True)
class PersonaState:
    """Stores persona derived text for reuse in TTS and LLM prompts."""

    style_text: str = ""


def build_persona_system_prompt(answers: Dict[str, str]) -> str:
    """Create a concise system prompt from persona answers."""

    lines: List[str] = ["You are a realtime voice assistant. Mimic the user’s linguistic style:"]
    for question in PERSONA_QUESTIONS:
        key = question["key"]
        value = (answers.get(key) or "").strip()
        if value:
            lines.append(f"- {question['label']}: {value}")
    lines.append("Keep answers brief unless asked.")
    return "\n".join(lines)


__all__ = ["PERSONA_QUESTIONS", "PersonaState", "build_persona_system_prompt"]
