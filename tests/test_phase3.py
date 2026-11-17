import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.datasets import FoldData  # noqa: E402
from src.models import probe  # noqa: E402
from src.models.finetune import AttentionPool, class_weights  # noqa: E402

LABELS = ["a", "b", "c"]


def _fold(n=60, seed=0):
    rng = np.random.default_rng(seed)
    parts = {}
    for role, m in (("train", n), ("val", n // 3), ("test", n // 3)):
        y = rng.integers(0, 3, m)
        parts[role] = pd.DataFrame({"clip_id": [f"{role}{i}" for i in range(m)], "path": "x",
                                    "label_idx": y, "label": [LABELS[i] for i in y]})
    return FoldData(labels=LABELS, **parts)


def _fake_embeddings(fd, informative_layer=2, n_layers=4, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    table = {}
    for part in (fd.train, fd.val, fd.test):
        for cid, y in zip(part["clip_id"], part["label_idx"]):
            e = rng.normal(size=(n_layers, dim)).astype(np.float32)
            e[informative_layer, y] += 4.0       # only one layer carries the label
            table[cid] = e
    return lambda key, ids, paths, device=None: (np.stack([table[i] for i in ids]), {"hf_id": "fake", "revision": "r"})


class _FakeEnc:
    n_params = 100


@pytest.fixture
def patched(monkeypatch):
    fd = _fold()
    monkeypatch.setattr(probe, "embeddings_for", _fake_embeddings(fd))
    monkeypatch.setattr(probe, "get_encoder", lambda key, device: _FakeEnc())
    return fd


def test_probe_last_layer_and_best_val(patched):
    cfg = {"model": {"encoder": "fake", "layer": "last", "grid_C": [0.1, 1.0]}}
    out = probe.probe_fit_predict(patched, cfg, 0)
    assert out["test_logits"].shape == (len(patched.test), 3)
    assert out["chosen"]["layer"] == 3 and set(out["chosen"]["val_macro_f1_by_layer"]) == {"0", "1", "2", "3"}
    assert out["params_trainable"] == 3 * 8 + 3 and out["params_total"] == 100 + 27
    best = probe.probe_fit_predict(patched, {"model": {**cfg["model"], "layer": "best_val"}}, 0)
    assert best["chosen"]["layer"] == 2                      # the informative layer, found on val
    acc = (best["test_logits"].argmax(1) == patched.test["label_idx"].to_numpy()).mean()
    assert acc > 0.9


def test_probe_is_deterministic(patched):
    cfg = {"model": {"encoder": "fake", "layer": "last", "grid_C": [1.0]}}
    a = probe.probe_fit_predict(patched, cfg, 0)["test_logits"]
    b = probe.probe_fit_predict(patched, cfg, 1)["test_logits"]
    np.testing.assert_allclose(a, b)


def test_class_weights_balanced():
    w = class_weights(np.array([0, 0, 0, 1]), 3).numpy()
    assert w[0] == pytest.approx(4 / 9) and w[1] == pytest.approx(4 / 3) and w[2] == 0.0


def test_attention_pool_is_convex_combination():
    pool = AttentionPool(8)
    h = torch.randn(2, 5, 8)
    z = pool(h)
    assert z.shape == (2, 8)
    assert torch.all(z <= h.max(1).values + 1e-5) and torch.all(z >= h.min(1).values - 1e-5)


def test_resolve_parent(tmp_path):
    from src.results.run import append_registry
    from src.train import resolve_parent

    append_registry({"run_id": "E07__ast_probe__20261117-000000__aaaaaaa", "experiment_id": "E07",
                     "status": "completed", "created_utc": "2026-11-17T00:00:00Z"}, tmp_path)
    append_registry({"run_id": "E07__ast_probe__20261118-000000__bbbbbbb", "experiment_id": "E07",
                     "status": "failed", "created_utc": "2026-11-18T00:00:00Z"}, tmp_path)
    assert resolve_parent("E07", tmp_path) == "E07__ast_probe__20261117-000000__aaaaaaa"
    assert resolve_parent("E99", tmp_path) == "E99"
    assert resolve_parent("E05__x__20261117-000000__ccccccc", tmp_path) == "E05__x__20261117-000000__ccccccc"
    assert resolve_parent(None, tmp_path) is None
    json.dumps(resolve_parent("E07", tmp_path))
