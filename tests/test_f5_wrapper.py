import numpy as np

from src.tts.f5_wrapper import F5Cloner, VoiceProfile


class DummyPipe:
    def tts(self, text, ref_audio, ref_sr, sr, prompt_style=None):  # noqa: D401
        duration_s = 0.6
        t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
        return 0.2 * np.sin(2 * np.pi * 220 * t)


class DictInferPipe:
    def infer(self, gen_text=None, ref_audio=None, ref_sr=None, sr=None, **kwargs):  # noqa: D401
        duration_s = 0.4
        rate = sr or 24000
        t = np.linspace(0, duration_s, int(rate * duration_s), endpoint=False)
        audio = 0.1 * np.sin(2 * np.pi * 330 * t)
        return {"audio": audio, "sample_rate": rate}


def test_set_voice_downmixes_and_normalizes():
    cloner = F5Cloner(device="cpu")
    stereo = np.stack([np.linspace(-1, 1, 1000), np.linspace(-0.5, 0.5, 1000)], axis=1)
    profile = cloner.set_voice(stereo, 16000)
    assert profile.speaker_wav.ndim == 1
    assert profile.speaker_sr == 16000
    peak = np.max(np.abs(profile.speaker_wav))
    assert 0 < peak <= 0.95 + 1e-3


def test_stream_tts_sync_emits_wav_chunks(monkeypatch):
    cloner = F5Cloner(device="cpu")
    cloner._pipe = DummyPipe()
    profile = VoiceProfile(speaker_wav=None, speaker_sr=None, style_notes="Warm")
    chunks = list(cloner.stream_tts_sync("Hello there", profile, out_sr=24000, chunk_ms=200))
    assert len(chunks) > 1
    assert all(chunk.startswith(b"RIFF") for chunk in chunks)


def test_stream_tts_sync_handles_dict_infer_result(monkeypatch):
    cloner = F5Cloner(device="cpu")
    cloner._pipe = DictInferPipe()
    profile = VoiceProfile(speaker_wav=np.zeros(1000), speaker_sr=24000, reference_text="Sample")
    chunks = list(cloner.stream_tts_sync("Testing dict output", profile, out_sr=24000, chunk_ms=200))
    assert len(chunks) > 0
    assert all(chunk.startswith(b"RIFF") for chunk in chunks)
