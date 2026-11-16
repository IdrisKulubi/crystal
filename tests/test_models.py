import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.calibrate import calibrate, fit_temperature  # noqa: E402
from src.models.deepinfant_repro import (  # noqa: E402
    N_SAMPLES, DeepInfantRepro, TorchDeepInfantMel, deepinfant_augment, deepinfant_mel,
)
from src.results import metrics as M  # noqa: E402


def test_torch_mel_matches_librosa():
    rng = np.random.default_rng(0)
    t = np.arange(N_SAMPLES) / 16000
    wav = (0.3 * np.sin(2 * np.pi * 450 * t) * np.abs(np.sin(2 * np.pi * 0.7 * t))
           + 0.01 * rng.normal(size=N_SAMPLES)).astype(np.float32)
    ref = deepinfant_mel(wav)
    got = TorchDeepInfantMel()(torch.from_numpy(wav[None]))[0].numpy()
    assert got.shape == ref.shape == (80, 438)
    assert np.max(np.abs(got - ref)) < 0.05  # dB


def test_model_forward_shape_and_size():
    m = DeepInfantRepro(5).eval()
    with torch.no_grad():
        out = m(torch.randn(2, 1, 80, 438))
    assert out.shape == (2, 5)
    assert 19_000_000 < sum(p.numel() for p in m.parameters()) < 21_000_000


def test_augment_keeps_length():
    rng = np.random.default_rng(0)
    w = np.ones(N_SAMPLES, np.float32)
    for _ in range(20):
        assert len(deepinfant_augment(w, rng)) == N_SAMPLES


def test_temperature_scaling_reduces_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 400)
    logits = rng.normal(size=(400, 3))
    logits[np.arange(400), y] += 1.0
    logits *= 5  # overconfident
    t = fit_temperature(logits, y)
    assert t > 1.5
    assert M.nll(y, M.softmax(logits, t)) < M.nll(y, M.softmax(logits, 1.0))
    cal = calibrate(logits, y)
    assert cal["ece_val_after"] < cal["ece_val_before"]
    assert set(cal["thresholds"]) == {"0.9", "0.7", "0.5"}
    assert cal["thresholds"]["0.5"] >= cal["thresholds"]["0.9"]
