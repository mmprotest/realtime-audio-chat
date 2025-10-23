"""Utilities for normalizing and exporting chat history entries."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

ChatMessageDict = dict[str, str]
ChatEntry = ChatMessageDict | Sequence[str] | str
ChatHistory = list[ChatEntry]


def normalize_chat_history(chatbot: Optional[ChatHistory | ChatEntry]) -> ChatHistory:
    """Ensure chatbot history is always represented as a mutable list."""

    if chatbot is None:
        return []
    if isinstance(chatbot, list):
        return chatbot
    if isinstance(chatbot, dict):
        return [chatbot]
    if isinstance(chatbot, str):
        return [{"role": "user", "content": chatbot}]
    if isinstance(chatbot, Sequence):
        return list(chatbot)
    return []


def chatbot_to_messages(chatbot: ChatHistory) -> list[ChatMessageDict]:
    """Convert Gradio Chatbot history entries into OpenAI-style messages."""

    messages: list[ChatMessageDict] = []
    for entry in chatbot:
        if isinstance(entry, dict):
            role = entry.get("role")
            content = entry.get("content")
            if role and content is not None:
                messages.append({"role": role, "content": str(content)})
        elif isinstance(entry, str):
            messages.append({"role": "user", "content": entry})
        elif isinstance(entry, Sequence):
            if len(entry) >= 1 and entry[0]:
                messages.append({"role": "user", "content": str(entry[0])})
            if len(entry) >= 2 and entry[1]:
                messages.append({"role": "assistant", "content": str(entry[1])})
    return messages


__all__ = [
    "ChatEntry",
    "ChatHistory",
    "ChatMessageDict",
    "chatbot_to_messages",
    "normalize_chat_history",
]
