"""OpenAI-compatible chat client wrapper."""
from __future__ import annotations

import logging
from typing import List

from openai import OpenAI

LOGGER = logging.getLogger(__name__)


class LLMClient:
    """Wrapper around the OpenAI SDK for chat completions."""

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)
        LOGGER.info("Initialized OpenAI client base_url=%s", base_url or "default")

    def chat(
        self,
        messages: List[dict],
        model: str,
        *,
        max_tokens: int = 200,
        temperature: float = 0.7,
    ) -> str:
        """Send a chat completion request and return the assistant text."""

        if not messages:
            raise ValueError("messages must not be empty")

        LOGGER.debug("Sending chat completion | model=%s messages=%s", model, messages)
        completion = self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choice = completion.choices[0]
        content = choice.message.content if choice.message else None
        return content or ""


__all__ = ["LLMClient"]
