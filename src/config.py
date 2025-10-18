"""Application configuration using Pydantic settings."""
from __future__ import annotations

import argparse
import logging
import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    device: str = Field(default_factory=lambda: "cuda" if _cuda_available() else "cpu", alias="DEVICE")
    f5_output_sr: int = Field(default=24000, alias="F5_OUTPUT_SR")
    chunk_ms: int = Field(default=200, alias="CHUNK_MS")
    fastrtc_mode: str = Field(default="send-receive", alias="FASTRTC_MODE")
    fastrtc_modality: str = Field(default="audio", alias="FASTRTC_MODALITY")
    panel_host: str = Field(default="0.0.0.0", alias="PANEL_HOST")
    panel_port: int = Field(default=7862, alias="PANEL_PORT")

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        frozen=True,
    )


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - conservative fallback
        logging.getLogger(__name__).debug("CUDA availability check failed", exc_info=True)
        return False


def _load_settings(argv: tuple[str, ...]) -> Settings:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference even if CUDA is available")
    known_args, _ = parser.parse_known_args(argv)

    if known_args.cpu:
        os.environ["DEVICE"] = "cpu"

    settings = Settings()  # type: ignore[arg-type]
    logging.getLogger(__name__).info(
        "Loaded settings | device=%s model=%s base_url=%s chunk_ms=%s",
        settings.device,
        settings.openai_model,
        settings.openai_base_url or "default",
        settings.chunk_ms,
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
