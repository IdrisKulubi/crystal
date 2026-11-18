"""Phase 5 — training-only augmentations and context features for fine-tuning (E30–E34).

Applied ONLY to training batches, after splitting (CLAUDE.md rule 1). Validation and test inputs are
never augmented. All random draws use the run's seeded numpy Generator.

* Noise (``train.augment.noise_snr_db: [lo, hi]``, ``noise_prob``): with probability ``noise_prob`` a training
  clip gets a random ESC-50 background clip (49 non-baby classes, CC BY-NC) added at an SNR drawn uniformly
  from [lo, hi] dB. The noise is band-limited to 4 kHz like the data (Donate-a-Cry is 8 kHz: wide-band noise
  would make train and test differ in a way unrelated to the task), tiled/cropped to the clip, and added
  only over the clip's real length (padding stays silent, as at test time). Household noise (HANDOFF) is
  not available yet, so the pool is ESC-50 only.
* SpecAugment (``train.augment.specaugment: true`` or ``{freq_mask: F, time_mask: T, n_masks: m}``): on the
  AST fbank input [T=1024, F=128], m frequency masks of width U[0, F] and m time masks of width U[0, T],
  set to 0 (= the AST normalisation mean). Defaults F=24, T=96, m=1 (AST paper ESC-50 recipe). AST only.
* Mixup (``train.augment.mixup_alpha: a``): lambda ~ Beta(a, a) per batch, inputs (and context vectors)
  mixed with a permutation of the batch; loss = lambda * L(y) + (1 - lambda) * L(y_perm).
* Context (``model.context_features: [age_bracket, gender]``): one-hot age bracket (0-4, from the
  Donate-a-Cry filename) and gender (f/m), concatenated to the pooled embedding before the classifier.
  Unknown values give all zeros.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.audio import TARGET_SR, load_audio
from src.config import REPO_ROOT

ESC = REPO_ROOT / "data" / "raw" / "ESC-50"
CONTEXT_LEVELS = {"age_bracket": [0, 1, 2, 3, 4], "gender": ["f", "m"]}
SPEC_DEFAULTS = {"freq_mask": 24, "time_mask": 96, "n_masks": 1}


# --------------------------------------------------------------------------- noise
def noise_bank(max_clips: int | None = None, rng_seed: int = 0) -> list[np.ndarray]:
    """Band-limited (4 kHz) 16 kHz ESC-50 background clips, excluding ``crying_baby``. float16 to save RAM."""
    from src.detection import band_limit

    meta_path = ESC / "meta" / "esc50.csv"
    if not meta_path.is_file():
        raise FileNotFoundError(f"{meta_path} missing: git clone https://github.com/karolpiczak/ESC-50 data/raw/ESC-50")
    meta = pd.read_csv(meta_path)
    meta = meta[meta["category"] != "crying_baby"].sort_values("filename").reset_index(drop=True)
    if max_clips is not None and max_clips < len(meta):
        meta = meta.iloc[np.random.default_rng(rng_seed).choice(len(meta), max_clips, replace=False)]
    bank = []
    for fn in meta["filename"]:
        cached = REPO_ROOT / "data" / "processed" / "esc50" / fn
        wav, _ = load_audio(cached if cached.is_file() else ESC / "audio" / fn, TARGET_SR)
        wav = band_limit(np.asarray(wav, np.float32))
        if np.sqrt(np.mean(wav ** 2)) > 1e-4:
            bank.append(wav.astype(np.float16))
    return bank


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def add_noise(wav: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """wav + noise scaled to ``snr_db`` (RMS ratio over the clip). Noise is randomly rolled and tiled."""
    wav = np.asarray(wav, np.float32)
    n = len(wav)
    noise = np.asarray(noise, np.float32)
    noise = np.roll(noise, int(rng.integers(0, len(noise))))
    noise = np.resize(noise, n)
    s, r = _rms(wav), _rms(noise)
    if s < 1e-6 or r < 1e-6:
        return wav
    scale = s / (r * 10 ** (snr_db / 20.0))
    return (wav + scale * noise).astype(np.float32)


def noisy_batch(wavs: list[np.ndarray], bank: list[np.ndarray], snr: tuple[float, float], prob: float,
                rng: np.random.Generator) -> list[np.ndarray]:
    out = []
    for w in wavs:
        if bank and rng.random() < prob:
            w = add_noise(w, bank[int(rng.integers(0, len(bank)))], float(rng.uniform(snr[0], snr[1])), rng)
        out.append(w)
    return out


# --------------------------------------------------------------------------- SpecAugment / mixup
def spec_cfg(value) -> dict | None:
    if not value:
        return None
    return {**SPEC_DEFAULTS, **(value if isinstance(value, dict) else {})}


def specaugment(x: np.ndarray, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """x [B, T, F] (AST fbank). Returns a masked copy."""
    x = x.copy()
    _, t, f = x.shape
    for i in range(len(x)):
        for _ in range(int(cfg["n_masks"])):
            w = int(rng.integers(0, int(cfg["freq_mask"]) + 1))
            if w:
                f0 = int(rng.integers(0, f - w + 1))
                x[i, :, f0:f0 + w] = 0.0
            w = int(rng.integers(0, int(cfg["time_mask"]) + 1))
            if w:
                t0 = int(rng.integers(0, t - w + 1))
                x[i, t0:t0 + w, :] = 0.0
    return x


def mixup_params(n: int, alpha: float, rng: np.random.Generator) -> tuple[float, np.ndarray]:
    return float(rng.beta(alpha, alpha)), rng.permutation(n)


# --------------------------------------------------------------------------- context
def context_matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """[n, sum(levels)] one-hot context; unknown values -> zeros."""
    cols = []
    for feat in features:
        if feat not in CONTEXT_LEVELS:
            raise ValueError(f"unknown context feature {feat!r}; known: {sorted(CONTEXT_LEVELS)}")
        levels = CONTEXT_LEVELS[feat]
        vals = df[feat].tolist() if feat in df else [None] * len(df)
        m = np.zeros((len(df), len(levels)), np.float32)
        for i, v in enumerate(vals):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            v = int(v) if feat == "age_bracket" else str(v)
            if v in levels:
                m[i, levels.index(v)] = 1.0
        cols.append(m)
    return np.concatenate(cols, axis=1) if cols else np.zeros((len(df), 0), np.float32)
