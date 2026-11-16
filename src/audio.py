"""Audio I/O: load any wav as mono float32 at a target sample rate."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 16000


def load_audio(path: str | Path, sr: int | None = TARGET_SR) -> tuple[np.ndarray, int]:
    """Return (mono float32 waveform in [-1, 1], sample rate). Resamples if ``sr`` is given."""
    wav, native_sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr is not None and native_sr != sr:
        import librosa

        wav = librosa.resample(wav, orig_sr=native_sr, target_sr=sr, res_type="soxr_hq").astype(np.float32)
        return wav, sr
    return wav, native_sr


def save_audio(path: str | Path, wav: np.ndarray, sr: int = TARGET_SR) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(wav, -1.0, 1.0), sr, subtype="PCM_16")


def pcm16_sha256(wav: np.ndarray) -> str:
    """Hash of the waveform quantised to 16-bit PCM (exact-duplicate detection)."""
    import hashlib

    q = np.round(np.clip(wav, -1.0, 1.0) * 32767).astype("<i2")
    return hashlib.sha256(q.tobytes()).hexdigest()
