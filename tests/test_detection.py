import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.detection import FRAME, SmallCNN, best_f1_threshold, frames_of  # noqa: E402


def test_frames_of_padding_and_gate():
    loud, quiet = np.ones(FRAME, np.float32), np.full(FRAME, 1e-3, np.float32)
    x, keep = frames_of(np.concatenate([loud, quiet, loud[: FRAME // 2]]), is_cry=True)
    assert x.shape == (3, FRAME) and list(keep) == [True, False, True]      # -60 dB frame gated out
    assert np.all(x[2, FRAME // 2:] == 0)
    x, keep = frames_of(np.concatenate([loud, quiet, loud[: FRAME // 4]]), is_cry=False)
    assert x.shape == (2, FRAME) and keep.all()                              # short remainder dropped; non-cry kept
    x, keep = frames_of(np.ones(100, np.float32), is_cry=False)
    assert x.shape == (1, FRAME)                                              # always >= 1 frame


def test_best_f1_threshold():
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.6])
    thr = best_f1_threshold(y, s)
    assert thr == pytest.approx(0.6)                                          # all positives, one FP -> F1 6/7
    y2, s2 = np.array([0, 1, 1]), np.array([0.5, 0.5, 0.9])                  # ties are not split
    assert best_f1_threshold(y2, s2) in (0.5, 0.9)


def test_small_cnn_shape():
    m = SmallCNN()
    assert m(torch.randn(3, 64, 101)).shape == (3,)
    assert sum(p.numel() for p in m.parameters()) < 200_000
