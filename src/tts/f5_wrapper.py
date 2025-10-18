"""F5-TTS wrapper for voice cloning and streaming."""
from __future__ import annotations

import contextlib
import inspect
import logging
import os
import tempfile
from dataclasses import dataclass
from io import BytesIO
from itertools import product
from typing import Generator, Optional

import numpy as np

try:  # pragma: no cover - dependency import is trivial when present
    import soundfile as sf
except Exception:  # pragma: no cover - exercised when soundfile missing
    sf = None

LOGGER = logging.getLogger(__name__)

_REFERENCE_PARAM_NAMES = {
    "ref_file",
    "ref_path",
    "ref_audio_path",
    "ref_wav_path",
    "reference_path",
    "ref_audio_file",
    "prompt_wav",
    "prompt_audio",
    "ref_audio",
    "reference_audio",
    "ref_waveform",
}


@dataclass(slots=True)
class VoiceProfile:
    """Stores reference audio and style instructions."""

    speaker_wav: Optional[np.ndarray] = None
    speaker_sr: Optional[int] = None
    style_notes: str = ""
    reference_text: str = ""


class F5Cloner:
    """Lazy-loading helper for F5-TTS voice cloning."""

    def __init__(self, *, device: str = "cpu") -> None:
        self.device = device
        self._pipe = None
        self._default_profile: VoiceProfile | None = None
        self._default_voice_attempted = False
        LOGGER.info("F5Cloner initialized | device=%s", device)

    def _load_pipe(self):  # pragma: no cover - small wrapper
        if self._pipe is None:
            F5TTS_cls = _resolve_f5_api()
            self._pipe = F5TTS_cls(device=self.device)
            LOGGER.info("Loaded F5-TTS pipeline")
        return self._pipe

    def ensure_pipeline(self) -> None:
        """Ensure the underlying F5 pipeline is instantiated."""

        self._load_pipe()

    def _resolve_profile(self, profile: VoiceProfile) -> VoiceProfile:
        """Fill in missing reference audio using bundled defaults when available."""

        if profile.speaker_wav is not None and profile.speaker_sr is not None:
            return profile

        default = self._get_default_profile()
        if default is None:
            if not self._default_voice_attempted:
                LOGGER.warning(
                    "No reference audio provided and no bundled default voice is available; synthesis may be skipped."
                )
                self._default_voice_attempted = True
            return profile

        merged = VoiceProfile(
            speaker_wav=default.speaker_wav,
            speaker_sr=default.speaker_sr,
            style_notes=profile.style_notes,
            reference_text=profile.reference_text or default.reference_text,
        )
        if not self._default_voice_attempted:
            LOGGER.info("Using bundled default voice sample for synthesis")
            self._default_voice_attempted = True
        return merged

    def _get_default_profile(self) -> VoiceProfile | None:
        if self._default_profile is not None:
            return self._default_profile

        clip = _load_packaged_reference()
        if clip is None:
            return None

        wav, sr, transcript = clip
        self._default_profile = VoiceProfile(
            speaker_wav=wav,
            speaker_sr=sr,
            reference_text=transcript,
        )
        return self._default_profile

    def set_voice(self, wav: np.ndarray, sr: int) -> VoiceProfile:
        """Prepare and store a normalized mono reference voice."""

        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)
        wav = wav.astype(np.float32, copy=False)
        peak = np.max(np.abs(wav))
        if peak > 0:
            wav = wav * (0.95 / peak)
        profile = VoiceProfile(speaker_wav=wav, speaker_sr=sr)
        LOGGER.info("Voice profile updated | sr=%s duration=%.2fs", sr, len(wav) / float(sr))
        return profile

    def stream_tts_sync(
        self,
        text: str,
        profile: VoiceProfile,
        out_sr: int,
        chunk_ms: int,
    ) -> Generator[bytes, None, None]:
        """Synthesize speech and yield WAV-encoded chunks."""

        if not text:
            return
            yield  # pragma: no cover - generator formality

        pipe = self._load_pipe()
        profile = self._resolve_profile(profile)
        LOGGER.debug(
            "Synthesizing text | length=%s ref_sr=%s chunk_ms=%s", len(text), profile.speaker_sr, chunk_ms
        )
        audio_np = _synthesize(pipe, text, profile, out_sr)
        if audio_np.size == 0:
            return

        chunk_samples = max(int(out_sr * (chunk_ms / 1000.0)), 1)
        for start in range(0, audio_np.size, chunk_samples):
            end = min(start + chunk_samples, audio_np.size)
            chunk = audio_np[start:end]
            yield _encode_wav(chunk, out_sr)


def _encode_wav(chunk: np.ndarray, sr: int) -> bytes:
    """Encode the provided samples into a mono 16-bit WAV payload."""

    buffer = BytesIO()
    if sf is not None:
        sf.write(buffer, chunk, sr, format="WAV", subtype="PCM_16")
        return buffer.getvalue()

    import struct
    import wave

    samples = [float(x) for x in chunk]
    # Clamp to [-1, 1] before scaling to 16-bit PCM
    pcm = [
        max(-32768, min(32767, int(round(sample * 32767.0))))
        for sample in (_clamp(sample, -1.0, 1.0) for sample in samples)
    ]

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        wav_file.writeframes(b"".join(struct.pack("<h", value) for value in pcm))

    return buffer.getvalue()


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return minimum if value < minimum else maximum if value > maximum else value


def _synthesize(pipe: object, text: str, profile: VoiceProfile, out_sr: int) -> np.ndarray:
    """Invoke the available F5 API and return mono audio samples."""

    if hasattr(pipe, "tts"):
        audio = pipe.tts(
            text=text,
            ref_audio=profile.speaker_wav,
            ref_sr=profile.speaker_sr,
            sr=out_sr,
            prompt_style=profile.style_notes or None,
        )
        return np.array(audio, dtype=np.float32).reshape(-1)

    if hasattr(pipe, "infer"):
        return _infer_v1(pipe, text, profile, out_sr)

    if callable(pipe):  # pragma: no cover - extremely defensive
        audio = pipe(text)
        return np.array(audio, dtype=np.float32).reshape(-1)

    raise AttributeError("F5-TTS pipeline exposes no supported synthesis method")


class _SilentProgress:
    """Minimal tqdm-compatible shim to silence console output."""

    def tqdm(self, iterable, *args, **kwargs):  # pragma: no cover - trivial container
        return iterable


def _infer_v1(pipe: object, text: str, profile: VoiceProfile, out_sr: int) -> np.ndarray:
    """Compatibility bridge for packages exposing an ``infer`` entry point."""

    ref_text = getattr(profile, "reference_text", "") or ""
    requires_reference = _infer_requires_reference(getattr(pipe, "infer"))
    candidates: list[tuple[str | None, str]] = []

    if profile.speaker_wav is not None and profile.speaker_sr is not None:
        try:
            wav = profile.speaker_wav.astype(np.float32, copy=False)
            sr = int(profile.speaker_sr)
            path = _write_temp_wav(wav, sr)
        except ValueError:
            LOGGER.warning("Reference audio contained no samples after preprocessing")
        else:
            candidates.append((path, ref_text))

    if not requires_reference:
        candidates.append((None, ref_text))

    if not candidates:
        if requires_reference:
            LOGGER.error(
                "F5 inference requires reference audio, but no usable clip was provided"
            )
        return np.array([], dtype=np.float32)

    errors: list[Exception] = []
    for ref_path, ref_prompt in candidates:
        try:
            result = _call_infer(pipe, text, ref_path, ref_prompt, profile)
        except Exception as exc:  # pragma: no cover - runtime guard
            attempt_no = len(errors) + 1
            errors.append(exc)
            LOGGER.error(
                "F5 inference attempt %s failed with %s: %s",
                attempt_no,
                exc.__class__.__name__,
                exc,
                exc_info=exc,
            )
            result = None
        finally:
            if ref_path:
                try:
                    os.remove(ref_path)
                except OSError:  # pragma: no cover - cleanup best effort
                    LOGGER.debug("Temporary reference file already cleaned up", exc_info=True)

        if result is None:
            continue

        audio, native_sr = _normalize_infer_result(pipe, result, out_sr)
        if audio.size == 0:
            continue
        if native_sr != out_sr:
            audio = _resample_linear(audio, native_sr, out_sr)
        return audio.astype(np.float32, copy=False)

    if errors:
        LOGGER.warning(
            "F5 inference failed after %s attempt(s); returning silence", len(errors)
        )
    return np.array([], dtype=np.float32)


def _infer_requires_reference(infer: object) -> bool:
    try:
        signature = inspect.signature(infer)
    except (TypeError, ValueError):
        return False

    for name, param in signature.parameters.items():
        if name == "self":
            continue
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ) and param.default is inspect._empty and name in _REFERENCE_PARAM_NAMES:
            return True
    return False


def _call_infer(
    pipe: object,
    text: str,
    ref_path: str | None,
    ref_text: str,
    profile: VoiceProfile,
) -> object | None:
    infer = getattr(pipe, "infer")
    try:
        signature = inspect.signature(infer)
    except (TypeError, ValueError):  # pragma: no cover - dynamic/built-in callables
        signature = None

    params = signature.parameters if signature else {}
    required_params: set[str] = set()
    if signature:
        for name, param in params.items():
            if name == "self":
                continue
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ) and param.default is inspect._empty:
                required_params.add(name)
    accepts_kwargs = (
        any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
        if signature
        else True
    )

    def _accepts(name: str) -> bool:
        return bool(name) and (accepts_kwargs or name in params)

    def _variant_map(value: object, names: list[str]) -> list[dict[str, object]]:
        variants: list[dict[str, object]] = []
        for candidate in names:
            if _accepts(candidate):
                variants.append({candidate: value})
        return variants

    text_variants = _variant_map(
        text,
        ["gen_text", "text", "prompt", "prompt_text", "tts_text", "target_text"],
    )
    if not text_variants:
        fallback_key = "gen_text" if _accepts("gen_text") else "text"
        text_variants = [{fallback_key: text}]

    clean_ref = (ref_text or "").strip()
    prompt_variants = (
        _variant_map(clean_ref, ["ref_text", "prompt_ref", "reference_text", "text_ref"])
        if clean_ref
        else [{}]
    ) or [{}]

    style_text = getattr(profile, "style_notes", "") or ""
    style_variants = (
        _variant_map(style_text, ["prompt_style", "style", "style_text", "persona"])
        if style_text
        else [{}]
    ) or [{}]

    path_variants = [{}]
    if ref_path:
        str_path = str(ref_path)
        path_variants = _variant_map(
            str_path,
            [
                "ref_file",
                "ref_path",
                "ref_audio_path",
                "ref_wav_path",
                "reference_path",
                "ref_audio_file",
                "prompt_wav",
                "ref_speaker_path",
            ],
        ) or [{}]

    wav = getattr(profile, "speaker_wav", None)
    sr = int(getattr(profile, "speaker_sr", 0) or 0)
    audio_variants: list[dict[str, object]] = [{}]
    sr_variants: list[dict[str, object]] = [{}]
    dict_variants: list[dict[str, object]] = [{}]
    if wav is not None and np.size(wav) > 0:
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        audio_variants = _variant_map(
            wav,
            [
                "ref_audio",
                "ref_wav",
                "speaker_audio",
                "voice_sample",
                "voice_samples",
                "ref_speaker",
                "ref_waveform",
            ],
        ) or [{}]
        for variant in audio_variants:
            if "voice_samples" in variant and not isinstance(variant["voice_samples"], (list, tuple)):
                variant["voice_samples"] = [variant["voice_samples"]]
        if sr > 0:
            sr_variants = _variant_map(
                sr,
                ["ref_sr", "sr", "speaker_sr", "sample_rate", "ref_sample_rate"],
            ) or [{}]
        payload = {"waveform": wav}
        if sr > 0:
            payload["sample_rate"] = sr
        dict_candidates = _variant_map(payload, ["prompt_speech", "ref_audio_dict", "reference_audio"])
        if dict_candidates:
            dict_variants = dict_candidates

    base_kwargs: dict[str, object] = {}
    if _accepts("show_info"):
        base_kwargs["show_info"] = lambda *args, **kwargs: None
    if _accepts("progress"):
        base_kwargs["progress"] = _SilentProgress()

    combos: list[dict[str, object]] = []
    for pieces in product(
        text_variants,
        prompt_variants,
        style_variants,
        path_variants,
        audio_variants,
        sr_variants,
        dict_variants,
    ):
        merged = dict(base_kwargs)
        for fragment in pieces:
            merged.update({k: v for k, v in fragment.items() if v is not None})
        if required_params and not (set(merged) >= required_params or accepts_kwargs):
            missing_required = required_params - set(merged)
            if missing_required and not accepts_kwargs:
                continue
        combos.append(merged)

    if (
        not combos
        and required_params
        and not accepts_kwargs
        and ref_path is None
        and required_params & _REFERENCE_PARAM_NAMES
    ):
        raise TypeError(
            "infer requires reference audio arguments that could not be populated"
        )

    unique: list[dict[str, object]] = []
    seen: set[tuple[tuple[str, object], ...]] = set()
    for kwargs in combos:
        normalized: list[tuple[str, object]] = []
        for key, value in sorted(kwargs.items(), key=lambda item: item[0]):
            try:
                hash(value)
            except TypeError:
                normalized.append((key, id(value)))
            else:
                normalized.append((key, value))
        normalized_key = tuple(normalized)
        if normalized_key not in seen:
            seen.add(normalized_key)
            unique.append(kwargs)

    last_type_error: TypeError | None = None
    for kwargs in unique:
        try:
            return infer(**kwargs)
        except TypeError as exc:
            last_type_error = exc
            continue

    # Fallback to positional signatures for legacy builds.
    with contextlib.suppress(TypeError):
        return infer(text)
    if ref_path is not None:
        with contextlib.suppress(TypeError):
            return infer(ref_path, ref_text, text)
        with contextlib.suppress(TypeError):
            return infer(ref_path, text)

    if last_type_error is not None:
        raise last_type_error
    raise TypeError("Unable to call infer with any supported signature")


def _normalize_infer_result(pipe: object, result: object, out_sr: int) -> tuple[np.ndarray, int]:
    if isinstance(result, dict):
        audio_value: object | None = None
        for key in (
            "audio",
            "audios",
            "waveform",
            "waveforms",
            "samples",
            "audio_data",
            "tts_audio",
        ):
            if key not in result:
                continue
            audio_value = result[key]
            if key in {"audios", "waveforms"} and isinstance(audio_value, (list, tuple)):
                audio_value = audio_value[0] if audio_value else None
            break

        if audio_value is None:
            raise ValueError("Infer result dictionary did not contain audio samples")

        sr_value: object | None = None
        for key in ("sample_rate", "sampling_rate", "sr", "audio_sr", "sampleRate"):
            if key in result:
                sr_value = result[key]
                break

        audio = np.array(audio_value, dtype=np.float32).reshape(-1)
        native_sr = int(sr_value) if sr_value else getattr(pipe, "target_sample_rate", out_sr)
        return audio, native_sr

    if not isinstance(result, tuple):
        audio = np.array(result, dtype=np.float32).reshape(-1)
        native_sr = getattr(pipe, "target_sample_rate", out_sr)
    else:
        if len(result) == 3:
            audio, native_sr, _ = result
        elif len(result) >= 2:
            audio, native_sr = result[:2]
        else:  # pragma: no cover - defensive guard
            audio, native_sr = result[0], out_sr
        audio = np.array(audio, dtype=np.float32).reshape(-1)

    native_sr = int(native_sr) if native_sr else out_sr
    return audio, native_sr


def _write_temp_wav(wav: np.ndarray, sr: int) -> str:
    """Persist the reference clip to disk for APIs that require a filepath."""

    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim != 1:
        wav = wav.reshape(-1)
    if wav.size == 0:
        raise ValueError("Reference audio is empty")

    # NamedTemporaryFile(delete=False) for cross-platform compatibility.
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    path = tmp.name
    tmp.close()

    if sf is not None:
        sf.write(path, wav, sr, format="WAV", subtype="PCM_16")
    else:  # pragma: no cover - fallback
        import wave

        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sr)
            samples = (_clamp(sample, -1.0, 1.0) for sample in wav)
            pcm = [int(round(value * 32767.0)) for value in samples]
            wav_file.writeframes(b"".join(int(sample).to_bytes(2, "little", signed=True) for sample in pcm))

    return path


def _load_packaged_reference() -> tuple[np.ndarray, int, str] | None:
    """Load a bundled example clip to use as a fallback reference voice."""

    try:
        import importlib.resources as resources
    except Exception:  # pragma: no cover - fallback for very old Python
        try:
            import importlib_resources as resources  # type: ignore
        except Exception:
            return None

    try:
        import tomllib  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - fallback for <=3.10
        try:  # pragma: no cover - dependency-light fallback
            import tomli as tomllib  # type: ignore
        except Exception:
            tomllib = None  # type: ignore

    package = "f5_tts.infer.examples"
    try:
        root = resources.files(package)
    except Exception:
        return None

    # Prefer reading the example manifests (``*.toml``) because they contain
    # canonical reference transcripts.  This keeps us from hitting Whisper
    # transcription on user machines where the ASR dependencies may be
    # unavailable or slow.
    if tomllib is not None:
        for manifest in _iter_traversable_matching(root, ".toml"):
            try:
                data = manifest.read_bytes()
            except Exception:
                continue
            try:
                config = tomllib.loads(data.decode("utf-8"))
            except Exception:
                continue

            ref_audio_path = config.get("ref_audio")
            ref_text = (config.get("ref_text") or "").strip()
            if not ref_audio_path:
                continue

            try:
                wav_bytes = root.joinpath(ref_audio_path).read_bytes()
            except Exception:
                continue

            try:
                wav, sr = _read_wav_bytes(wav_bytes)
            except Exception:
                continue
            if wav.size == 0:
                continue

            return wav, sr, ref_text

    # Fall back to scanning for standalone ``.wav`` files paired with ``.txt``
    # transcripts when manifests are unavailable.
    for candidate in _iter_traversable_wavs(root):
        try:
            wav_bytes = candidate.read_bytes()
        except Exception:
            continue

        transcript = ""
        with contextlib.suppress(Exception):
            text_resource = candidate.with_suffix(".txt")
            transcript = (text_resource.read_text(encoding="utf-8") or "").strip()

        try:
            wav, sr = _read_wav_bytes(wav_bytes)
        except Exception:
            continue
        if wav.size == 0:
            continue
        return wav, sr, transcript

    return None


def _read_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    buffer = BytesIO(data)
    if sf is not None:
        audio, sr = sf.read(buffer, dtype="float32")
    else:  # pragma: no cover - fallback path
        import wave

        with wave.open(buffer, "rb") as wav_file:
            sr = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32767.0

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32, copy=False), int(sr)


def _resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Lightweight linear resampler to avoid heavy dependencies."""

    if orig_sr == target_sr or audio.size == 0:
        return audio.astype(np.float32, copy=False)

    duration = audio.size / float(orig_sr)
    target_size = max(int(round(duration * target_sr)), 1)
    # Use linspace to map samples over the same duration.
    x_old = np.linspace(0.0, duration, num=audio.size, endpoint=False, dtype=np.float32)
    x_new = np.linspace(0.0, duration, num=target_size, endpoint=False, dtype=np.float32)
    resampled = np.interp(x_new, x_old, audio.astype(np.float32, copy=False))
    return resampled.astype(np.float32, copy=False)


__all__ = ["VoiceProfile", "F5Cloner"]


def _resolve_f5_api():  # pragma: no cover - exercised only with dependency installed
    """Locate the F5-TTS entry point across published package layouts."""

    try:
        import f5_tts  # type: ignore
    except ImportError as exc:
        raise RuntimeError("f5-tts is not installed. Please install dependencies.") from exc

    candidate_attrs = ("F5TTS", "F5TTSInference", "F5TTSPipeline")
    for attr in candidate_attrs:
        cls = getattr(f5_tts, attr, None)
        if cls is not None:
            return cls

    from importlib import import_module

    candidate_modules = (
        "f5_tts.api",
        "f5_tts.inference",
    )
    for module_name in candidate_modules:
        try:
            module = import_module(module_name)
        except ImportError:
            continue
        for attr in candidate_attrs:
            cls = getattr(module, attr, None)
            if cls is not None:
                return cls

    raise RuntimeError(
        "Installed f5-tts package does not expose an F5TTS-compatible class. "
        "Please upgrade/downgrade f5-tts or install extras that include the inference API."
    )


def _iter_traversable_wavs(root):
    """Iterate over ``*.wav`` files beneath an importlib.resources Traversable tree."""

    yield from _iter_traversable_matching(root, ".wav")


def _iter_traversable_matching(root, suffix: str):
    """Iterate over Traversable entries that end with ``suffix``."""

    stack = [root]
    while stack:
        current = stack.pop()
        try:
            children = list(current.iterdir())
        except Exception:
            continue

        for child in children:
            try:
                if child.is_dir():
                    stack.append(child)
                    continue
            except Exception:
                pass

            name = getattr(child, "name", "")
            if isinstance(name, str) and name.lower().endswith(suffix):
                yield child
