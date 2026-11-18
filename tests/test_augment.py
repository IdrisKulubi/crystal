import numpy as np
import pandas as pd
import torch

from src.augment import add_noise, context_matrix, noisy_batch, spec_cfg, specaugment


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def test_add_noise_hits_target_snr():
    rng = np.random.default_rng(0)
    wav = np.sin(np.linspace(0, 300, 16000)).astype(np.float32)
    noise = rng.normal(size=5000).astype(np.float32)
    for snr in (0.0, 10.0, 20.0):
        out = add_noise(wav, noise, snr, np.random.default_rng(1))
        got = 20 * np.log10(_rms(wav) / _rms(out - wav))
        assert abs(got - snr) < 1e-3
        assert out.shape == wav.shape and out.dtype == np.float32


def test_silent_clip_untouched_and_prob_zero_is_identity():
    rng = np.random.default_rng(0)
    z = np.zeros(1000, np.float32)
    assert np.array_equal(add_noise(z, np.ones(10, np.float32), 5.0, rng), z)
    wavs = [rng.normal(size=800).astype(np.float32) for _ in range(3)]
    out = noisy_batch(wavs, [np.ones(100, np.float16)], (0, 20), 0.0, rng)
    assert all(np.array_equal(a, b) for a, b in zip(wavs, out))


def test_specaugment_masks_within_limits_and_copies():
    x = np.ones((4, 1024, 128), np.float32)
    cfg = spec_cfg({"freq_mask": 24, "time_mask": 96})
    out = specaugment(x, cfg, np.random.default_rng(0))
    assert np.all(x == 1)                              # input not modified
    for i in range(4):
        full_t = (out[i] == 0).all(axis=1).sum()       # fully masked time frames
        full_f = (out[i] == 0).all(axis=0).sum()       # fully masked frequency bins
        assert full_t <= 96 and full_f <= 24
    assert spec_cfg(False) is None and spec_cfg(True)["n_masks"] == 1


def test_context_matrix_one_hot_and_unknown():
    df = pd.DataFrame({"age_bracket": [0, 4, np.nan, 2], "gender": ["m", "f", "m", None]})
    m = context_matrix(df, ["age_bracket", "gender"])
    assert m.shape == (4, 7)
    assert m[0].tolist() == [1, 0, 0, 0, 0, 0, 1]
    assert m[1].tolist() == [0, 0, 0, 0, 1, 1, 0]
    assert m[2, :5].sum() == 0 and m[3, 5:].sum() == 0
    assert context_matrix(df, []).shape == (4, 0)


def test_finetune_model_context_head():
    from src.models.finetune import AttentionPool  # noqa: F401  (import check)
    lin = torch.nn.Linear(10 + 7, 5)
    z = torch.zeros(2, 10)
    ctx = torch.ones(2, 7)
    assert lin(torch.cat([z, ctx], 1)).shape == (2, 5)
