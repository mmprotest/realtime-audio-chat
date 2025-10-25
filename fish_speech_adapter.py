"""FastRTC-compatible adapter for the Fish Audio OpenAudio S1 Mini model."""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Generator, Optional

import numpy as np
from numpy.typing import NDArray

try:
    import torch
except ImportError as exc:  # pragma: no cover - torch is required for inference
    raise ImportError("fish-speech adapter requires PyTorch to be installed") from exc

try:
    import resampy
except ImportError:  # pragma: no cover - resampy is an optional dependency
    resampy = None

try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError as exc:  # pragma: no cover - required for automatic checkpoint download
    raise ImportError(
        "fish-speech adapter requires `huggingface_hub` to fetch checkpoints"
    ) from exc

def _ensure_fish_speech_project_root() -> None:
    """Create the indicator file expected by pyrootutils inside the package."""

    spec = importlib.util.find_spec("fish_speech")
    if not spec or not spec.submodule_search_locations:
        return

    package_root = Path(next(iter(spec.submodule_search_locations)))
    indicator = package_root / ".project-root"
    if indicator.exists():
        return

    try:
        indicator.touch(exist_ok=True)
    except OSError:
        # If the environment is read-only we silently ignore the failure. The
        # upcoming import will still surface the original error message, which
        # is more helpful than masking it with our own exception.
        return


logger = logging.getLogger(__name__)


_ensure_fish_speech_project_root()

try:
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.models.dac.inference import load_model as load_decoder_model
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest
except ImportError as exc:  # pragma: no cover - fish-speech must be available at runtime
    raise ImportError(
        "fish-speech adapter requires the `fish-speech` package to be installed"
    ) from exc


DEFAULT_CHECKPOINT_REPO = "fishaudio/openaudio-s1-mini"
DEFAULT_CHECKPOINT_DIR = Path("checkpoints/openaudio-s1-mini")
DEFAULT_DECODER_CONFIG = "modded_dac_vq"


@dataclass
class TTSOptions:
    """Placeholder for future configurable options."""

    kwargs: dict[str, Any] | None = None


class FishSpeechTTSModel:
    """Wrap the Fish Audio OpenAudio S1 Mini model for use with FastRTC."""

    def __init__(
        self,
        ref_wav: str,
        ref_text: Optional[str] = None,
        *,
        checkpoint_dir: str | os.PathLike[str] | None = None,
        download: bool = True,
        device: Optional[str] = None,
        precision: Optional[str] = None,
        compile: bool | None = None,
        inference_kwargs: Optional[dict[str, Any]] = None,
        target_sample_rate: int | None = None,
        hf_token: Optional[str] = None,
        allow_unsupported_cuda: bool | None = None,
    ) -> None:
        reference_path = Path(ref_wav)
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_wav}")

        self._ref_audio_bytes = reference_path.read_bytes()
        self._ref_text = (ref_text or "").strip()
        self._checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir is not None else DEFAULT_CHECKPOINT_DIR
        )
        self._download = download
        self._allow_unsupported_cuda = self._resolve_allow_unsupported_cuda(
            allow_unsupported_cuda
        )
        self._device = self._resolve_device(device)
        self._precision = self._resolve_precision(precision)
        self._compile = bool(compile) if compile is not None else False
        self._target_sample_rate = target_sample_rate
        self._hf_token = self._resolve_hf_token()

        self._default_inference_kwargs = {
            "chunk_length": 200,
            "max_new_tokens": 1024,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "temperature": 0.9,
            "seed": None,
            "use_memory_cache": "off",
            "normalize": True,
            "format": "wav",
            "streaming": False,
        }
        if inference_kwargs:
            self._default_inference_kwargs.update(inference_kwargs)

        self._engine: TTSInferenceEngine | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @staticmethod
    def _parse_truthy(value: str | None) -> bool:
        if value is None:
            return False
        value = value.strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _resolve_allow_unsupported_cuda(
        self, hint: bool | None
    ) -> bool:
        if hint is not None:
            return bool(hint)

        env_value = os.getenv("FISH_SPEECH_ALLOW_UNSUPPORTED_CUDA")
        if env_value is not None:
            return self._parse_truthy(env_value)

        return False

    def _is_cuda_compatible(self) -> bool:
        """Return ``True`` when the active CUDA device is supported by PyTorch."""

        if not torch.cuda.is_available():
            return False

        try:
            major, minor = torch.cuda.get_device_capability(0)
        except Exception:  # pragma: no cover - device query failure
            return False

        device_name: str
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:  # pragma: no cover - fallback when the driver is unavailable
            device_name = "CUDA device"

        arch = f"sm_{major}{minor}"

        get_arch_list = getattr(torch.cuda, "get_arch_list", None)
        supported_arches: set[str] | None = None
        if callable(get_arch_list):
            try:
                supported_arches = {entry.lower() for entry in get_arch_list()}
            except Exception:  # pragma: no cover - guard against backend introspection failures
                supported_arches = None

        if supported_arches is not None and arch.lower() not in supported_arches:
            supported = ", ".join(sorted(supported_arches)) or "<unknown>"
            message = (
                "Detected CUDA device %s with compute capability %s which is "
                "not part of this PyTorch build (supported: %s)."
            )
            if self._allow_unsupported_cuda:
                logger.warning(message, device_name, arch, supported)
            else:
                logger.warning(
                    message + " Falling back to CPU. Set "
                    "FISH_SPEECH_ALLOW_UNSUPPORTED_CUDA=1 to bypass this check.",
                    device_name,
                    arch,
                    supported,
                )
                return False

        try:
            torch.zeros(1, device="cuda")
        except RuntimeError as exc:  # pragma: no cover - CUDA runtime failure
            message = str(exc)
            normalized = message.lower()
            if "no kernel image" in normalized or "not compatible" in normalized:
                if self._allow_unsupported_cuda:
                    logger.warning(
                        "%s reported a compatibility issue: %s. Continuing "
                        "because unsupported CUDA has been explicitly allowed.",
                        device_name,
                        message.strip(),
                    )
                    return True
                logger.warning(
                    "Falling back to CPU because %s is incompatible with this "
                    "PyTorch build: %s. Set FISH_SPEECH_ALLOW_UNSUPPORTED_CUDA=1 "
                    "to attempt running on this device anyway.",
                    device_name,
                    message.strip(),
                )
                return False
            raise

        return True

    def _resolve_device(self, device_hint: Optional[str]) -> str:
        if device_hint:
            if device_hint.startswith("cuda") and not self._is_cuda_compatible():
                if self._allow_unsupported_cuda:
                    logger.warning(
                        "Attempting to use CUDA device %s despite compatibility "
                        "warnings.",
                        device_hint,
                    )
                    return device_hint
                logger.warning(
                    "Requested CUDA device is not supported; using CPU instead."
                )
                return "cpu"
            return device_hint

        if self._is_cuda_compatible():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        if hasattr(torch, "xpu") and torch.xpu.is_available():  # type: ignore[attr-defined]
            return "xpu"
        return "cpu"

    def _resolve_precision(self, precision_hint: Optional[str]) -> torch.dtype:
        if precision_hint:
            precision_hint = precision_hint.lower()
            if precision_hint in {"fp16", "float16", "half"}:
                return torch.float16
            if precision_hint in {"bf16", "bfloat16"}:
                return torch.bfloat16
            if precision_hint in {"fp32", "float32"}:
                return torch.float32
            raise ValueError(f"Unsupported precision hint: {precision_hint}")

        if self._device == "cuda":
            return torch.bfloat16
        if self._device in {"mps", "xpu"}:
            return torch.float16
        return torch.float32

    @staticmethod
    def _resolve_hf_token() -> Optional[str]:
        """Resolve the Hugging Face token from environment variables."""

        # Accept a handful of common environment variable names so users can
        # configure credentials without modifying code.
        env_vars = (
            "FISH_SPEECH_HF_TOKEN",
            "HF_TOKEN",
            "HUGGINGFACE_TOKEN",
            "HUGGINGFACE_HUB_TOKEN",
            "HUGGINGFACEHUB_API_TOKEN",
        )
        for env_var in env_vars:
            token = os.getenv(env_var)
            if token and token.strip():
                return token.strip()

        return None

    def _ensure_engine(self) -> TTSInferenceEngine:
        if self._engine is not None:
            return self._engine

        with self._load_lock:
            if self._engine is not None:
                return self._engine

            os.environ.setdefault("EINX_FILTER_TRACEBACK", "false")

            checkpoint_dir = self._checkpoint_dir
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            if self._download:
                try:
                    snapshot_download(
                        repo_id=DEFAULT_CHECKPOINT_REPO,
                        local_dir=str(checkpoint_dir),
                        local_dir_use_symlinks=False,
                        token=self._hf_token,
                    )
                except HfHubHTTPError as exc:  # pragma: no cover - network failure path
                    status_code = getattr(exc.response, "status_code", None)
                    if status_code in {401, 403}:
                        raise RuntimeError(
                            "Authentication is required to download Fish Audio checkpoints. "
                            "Set the FISH_SPEECH_HF_TOKEN environment variable (or one of "
                            "HF_TOKEN, HUGGINGFACE_TOKEN, HUGGINGFACE_HUB_TOKEN, "
                            "HUGGINGFACEHUB_API_TOKEN) with a valid Hugging Face access token."
                        ) from exc
                    raise

            try:
                llama_queue = launch_thread_safe_queue(
                    checkpoint_path=checkpoint_dir,
                    device=self._device,
                    precision=self._precision,
                    compile=self._compile,
                )
            except RuntimeError as exc:
                error_message = str(exc)
                if (
                    self._device.startswith("cuda")
                    and "no kernel image" in error_message.lower()
                    and not self._allow_unsupported_cuda
                ):
                    raise RuntimeError(
                        "Failed to initialize Fish-Speech on CUDA because the "
                        "installed PyTorch build does not include kernels for the "
                        "detected GPU architecture. Reinstall PyTorch with CUDA "
                        "support for this GPU (see https://pytorch.org/get-started/locally/) "
                        "or set FISH_SPEECH_ALLOW_UNSUPPORTED_CUDA=1 to force GPU "
                        "initialization at your own risk."
                    ) from exc
                raise

            decoder_checkpoint = checkpoint_dir / "codec.pth"
            decoder_model = load_decoder_model(
                config_name=DEFAULT_DECODER_CONFIG,
                checkpoint_path=decoder_checkpoint,
                device=self._device,
            )

            self._engine = TTSInferenceEngine(
                llama_queue=llama_queue,
                decoder_model=decoder_model,
                compile=self._compile,
                precision=self._precision,
            )

        return self._engine

    def _build_request_kwargs(self, options: Optional[TTSOptions]) -> dict[str, Any]:
        merged = dict(self._default_inference_kwargs)
        if options and options.kwargs:
            merged.update(options.kwargs)

        request_kwargs: dict[str, Any] = {}
        for key in [
            "chunk_length",
            "max_new_tokens",
            "top_p",
            "repetition_penalty",
            "temperature",
            "seed",
            "use_memory_cache",
            "normalize",
            "format",
            "streaming",
        ]:
            if key in merged and merged[key] is not None:
                request_kwargs[key] = merged[key]

        # Ensure chunk length remains within the range accepted by the schema.
        chunk_length = request_kwargs.get("chunk_length")
        if chunk_length is not None:
            chunk_length = int(chunk_length)
            request_kwargs["chunk_length"] = min(300, max(100, chunk_length))

        if request_kwargs.get("format") not in {"wav", "pcm", "mp3"}:
            request_kwargs["format"] = "wav"

        use_memory_cache = request_kwargs.get("use_memory_cache")
        if use_memory_cache not in {"on", "off"}:
            request_kwargs["use_memory_cache"] = "off"

        request_kwargs["streaming"] = False
        request_kwargs.setdefault("normalize", True)
        request_kwargs.setdefault("max_new_tokens", 1024)

        return request_kwargs

    def _prepare_references(self) -> list[ServeReferenceAudio]:
        if not self._ref_audio_bytes:
            return []
        return [
            ServeReferenceAudio(audio=self._ref_audio_bytes, text=self._ref_text or "")
        ]

    def _run_inference(
        self, text: str, options: Optional[TTSOptions] = None
    ) -> tuple[int, NDArray[np.float32]]:
        engine = self._ensure_engine()
        request_kwargs = self._build_request_kwargs(options)
        references = self._prepare_references()

        req = ServeTTSRequest(
            text=text,
            references=references,
            reference_id=None,
            **request_kwargs,
        )

        with self._inference_lock:
            for result in engine.inference(req):
                if result.code == "error" and result.error is not None:
                    raise result.error
                if result.code == "final" and result.audio is not None:
                    sample_rate, audio = result.audio
                    return self._post_process_audio(sample_rate, audio)

        raise RuntimeError("Fish-Speech returned no audio for the provided text")

    def _post_process_audio(
        self, sample_rate: int, audio: np.ndarray
    ) -> tuple[int, NDArray[np.float32]]:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[None, :]
        elif audio.ndim > 2:
            audio = audio.reshape(audio.shape[0], -1)

        target_rate = self._target_sample_rate
        if target_rate and target_rate != sample_rate:
            if resampy is None:
                raise RuntimeError(
                    "Resampling requested but `resampy` is not installed."
                )
            audio = resampy.resample(audio, sample_rate, target_rate, axis=-1).astype(
                np.float32
            )
            sample_rate = target_rate

        return sample_rate, audio

    def tts(
        self, text: str, options: Optional[TTSOptions] = None
    ) -> tuple[int, NDArray[np.float32]]:
        """Generate a single utterance synchronously."""

        stripped = text.strip()
        if not stripped:
            raise ValueError("Cannot synthesize empty text")
        return self._run_inference(stripped, options=options)

    def stream_tts_sync(
        self, text: str, options: Optional[TTSOptions] = None
    ) -> Generator[tuple[int, NDArray[np.float32]], None, None]:
        """Yield audio for each sentence-sized segment synchronously."""

        for segment in _split_on_sentences(text):
            stripped = segment.strip()
            if not stripped:
                continue
            yield self._run_inference(stripped, options)

    async def stream_tts(
        self, text: str, options: Optional[TTSOptions] = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.float32]], None]:
        """Asynchronously yield audio segments using a background thread."""

        loop = asyncio.get_running_loop()
        for segment in _split_on_sentences(text):
            stripped = segment.strip()
            if not stripped:
                continue
            result = await loop.run_in_executor(
                None, self._run_inference, stripped, options
            )
            yield result


def _split_on_sentences(text: str) -> list[str]:
    pattern = re.compile(r"(?<=[.!?]\s)|(?<=\n)")
    parts = pattern.split(text)
    if not parts:
        return [text]
    return [part for part in parts if part]
