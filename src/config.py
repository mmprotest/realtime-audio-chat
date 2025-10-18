"""Application configuration helpers with minimal dependencies."""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from functools import lru_cache

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - used when python-dotenv missing
    def load_dotenv(*_, **__):
        logging.getLogger(__name__).debug(
            "python-dotenv not installed; skipping .env loading",
            exc_info=True,
        )
        return False


LOGGER = logging.getLogger(__name__)
load_dotenv()


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration sourced from environment variables."""

    openai_api_key: str
    openai_base_url: str | None
    openai_model: str
    device: str
    f5_output_sr: int
    chunk_ms: int
    ui_host: str
    ui_port: int
    ui_share: bool
    ui_inbrowser: bool


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - conservative fallback
        LOGGER.debug("CUDA availability check failed", exc_info=True)
        return False


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    return value


def _require_env(name: str) -> str:
    value = _get_env(name)
    if value is None:
        raise RuntimeError(f"Environment variable {name} is required")
    return value


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    LOGGER.warning("Unrecognized boolean value '%s'; using default=%s", value, default)
    return default


def _load_settings(argv: tuple[str, ...]) -> Settings:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference even if CUDA is available")
    known_args, _ = parser.parse_known_args(argv)

    if known_args.cpu:
        os.environ["DEVICE"] = "cpu"

    default_device = "cuda" if _cuda_available() else "cpu"

    settings = Settings(
        openai_api_key=_require_env("OPENAI_API_KEY"),
        openai_base_url=_get_env("OPENAI_BASE_URL"),
        openai_model=_get_env("OPENAI_MODEL", "gpt-4o-mini"),
        device=_get_env("DEVICE", default_device) or default_device,
        f5_output_sr=int(_get_env("F5_OUTPUT_SR", "24000") or 24000),
        chunk_ms=int(_get_env("CHUNK_MS", "200") or 200),
        ui_host=_get_env("UI_HOST", "0.0.0.0") or "0.0.0.0",
        ui_port=int(_get_env("UI_PORT", "7860") or 7860),
        ui_share=_to_bool(_get_env("UI_SHARE"), False),
        ui_inbrowser=_to_bool(_get_env("UI_OPEN_BROWSER"), False),
    )

    LOGGER.info(
        "Loaded settings | device=%s model=%s base_url=%s chunk_ms=%s ui_host=%s ui_port=%s",
        settings.device,
        settings.openai_model,
        settings.openai_base_url or "default",
        settings.chunk_ms,
        settings.ui_host,
        settings.ui_port,
    )
    return settings


@lru_cache()
def _cached_settings(argv: tuple[str, ...]) -> Settings:
    return _load_settings(argv)


def get_settings(argv: list[str] | None = None) -> Settings:
    """Load settings once, supporting a ``--cpu`` CLI override."""

    argv_tuple = tuple(argv or [])
    return _cached_settings(argv_tuple)


get_settings.cache_clear = _cached_settings.cache_clear  # type: ignore[attr-defined]


__all__ = ["Settings", "get_settings"]
