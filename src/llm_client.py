"""OpenAI-compatible chat client wrapper."""
from __future__ import annotations

import logging
from typing import List

try:  # pragma: no cover - dependency import is trivial when available
    from openai import OpenAI as _OpenAI
except Exception:  # pragma: no cover - exercised in compatibility tests
    _OpenAI = None


class _MissingOpenAI:
    def __init__(self, *_, **__):
        raise RuntimeError(
            "openai is not installed. Install the optional dependencies to enable LLM features."
        )


OpenAI = _OpenAI or _MissingOpenAI

LOGGER = logging.getLogger(__name__)


class LLMClient:
    """Wrapper around the OpenAI SDK for chat completions."""

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._http_client = None
        try:
            self._client = OpenAI(**client_kwargs)
        except TypeError as exc:
            if "unexpected keyword argument 'proxies'" not in str(exc):
                raise
            LOGGER.warning(
                "OpenAI client failed to initialize due to proxy incompatibility; retrying without env proxies."
            )
            retry_kwargs = dict(client_kwargs)
            self._http_client = self._build_http_client_without_proxies()
            retry_kwargs["http_client"] = self._http_client
            self._client = OpenAI(**retry_kwargs)

        LOGGER.info("Initialized OpenAI client base_url=%s", base_url or "default")

    @staticmethod
    def _build_http_client_without_proxies():
        """Return an httpx client that ignores environment proxy settings."""

        try:
            import httpx
        except Exception as exc:  # pragma: no cover - import should succeed when openai installed
            raise RuntimeError("httpx is required to create an OpenAI client without proxies") from exc
        return httpx.Client(trust_env=False)

    def close(self) -> None:
        """Close any underlying HTTP client resources."""

        if self._http_client is not None:
            try:
                self._http_client.close()
            finally:
                self._http_client = None

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
