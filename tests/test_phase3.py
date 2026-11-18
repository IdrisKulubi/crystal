import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.datasets import FoldData  # noqa: E402
from src.models import probe  # noqa: E402
from src.models.finetune import AttentionPool, FocalLoss, class_weights, epoch_order  # noqa: E402

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
    return lambda key, ids, paths, device=None, checkpoint=None: (np.stack([table[i] for i in ids]), {"hf_id": "fake", "revision": "r"})


class _FakeEnc:
    n_params = 100


@pytest.fixture
def patched(monkeypatch):
    fd = _fold()
    monkeypatch.setattr(probe, "embeddings_for", _fake_embeddings(fd))
    monkeypatch.setattr(probe, "get_encoder", lambda key, device, checkpoint=None: _FakeEnc())
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


def test_focal_gamma0_equals_weighted_ce():
    logits, y = torch.randn(16, 3), torch.randint(0, 3, (16,))
    w = torch.tensor([0.5, 1.0, 2.0])
    ref = torch.nn.functional.cross_entropy(logits, y, weight=w)
    assert torch.allclose(FocalLoss(0.0, w)(logits, y), ref, atol=1e-6)
    assert FocalLoss(2.0, w)(logits, y) < ref                      # down-weights easy examples


def test_balanced_sampler_equalises_classes():
    y = np.array([0] * 90 + [1] * 9 + [2] * 1)
    rng = np.random.default_rng(0)
    draws = np.concatenate([epoch_order(y, 3, "balanced", rng) for _ in range(50)])
    frac = np.bincount(y[draws], minlength=3) / len(draws)
    assert np.all(np.abs(frac - 1 / 3) < 0.03)
    perm = epoch_order(y, 3, "shuffle", np.random.default_rng(0))
    assert sorted(perm) == list(range(len(y)))
    with pytest.raises(ValueError):
        epoch_order(y, 3, "nope", rng)


@pytest.mark.parametrize("train", [
    {"loss": "focal", "sampler": "shuffle", "early_stopping_metric": "val_macro_f1"},
    {"loss": "ce", "sampler": "balanced", "early_stopping_metric": "val_macro_f1"},
    {"loss": "weighted_ce", "sampler": "shuffle", "early_stopping_metric": "val_loss_weighted"},
    {"loss": "weighted_ce", "sampler": "shuffle", "early_stopping_metric": "val_macro_f1",
     "augment": {"mixup_alpha": 0.4, "specaugment": {"freq_mask": 1, "time_mask": 1}}},
    {"loss": "focal", "sampler": "shuffle", "early_stopping_metric": "val_macro_f1",
     "augment": {"noise_snr_db": [0, 20], "noise_prob": 1.0}, "_context": ["age_bracket", "gender"]},
])
def test_finetune_loop_options(monkeypatch, train):
    import transformers
    from src.models import finetune

    class Tiny(torch.nn.Module):
        def __init__(self, *a, **k):
            super().__init__()
            self.lin, self.revision = torch.nn.Linear(16 + k.get("context_dim", 0), 3), "r"

        def forward(self, x, ctx=None):
            z = x.reshape(len(x), -1)
            return self.lin(z if ctx is None else torch.cat([z, ctx], 1))

    fd = _fold(30)
    monkeypatch.setattr(finetune, "FineTuneModel", Tiny)
    monkeypatch.setattr(finetune, "load_waveforms", lambda paths: [np.zeros(4, np.float32)] * len(paths))
    monkeypatch.setattr(finetune, "model_inputs", lambda fe, kind, w, n: np.random.default_rng(0).normal(size=(len(w), 4, 4)).astype(np.float32))
    import src.augment
    monkeypatch.setattr(src.augment, "noise_bank", lambda: [np.ones(4, np.float16)])
    train = dict(train)
    context = train.pop("_context", [])
    monkeypatch.setattr(transformers.AutoFeatureExtractor, "from_pretrained", staticmethod(lambda *a, **k: None))
    cfg = {"model": {"backbone": "ast", "head": "mean_pool", "peft": None, "context_features": context},
           "data": {"clip_seconds": 1.0},
           "train": {"lr": 1e-2, "weight_decay": 0.0, "batch_size": 8, "grad_accum": 1, "epochs": 3,
                     "warmup_ratio": 0.0, "patience": 5, "focal_gamma": 2.0, **train}}
    out = finetune.finetune_fit_predict(fd, cfg, 0)
    ch = out["chosen"]
    assert out["test_logits"].shape == (len(fd.test), 3) and len(ch["selection_trace"]) == 3
    assert ch["best_epoch"] == 1 + int(np.argmax(ch["selection_trace"]))
    assert ch["loss"] == train["loss"] and ch["sampler"] == train["sampler"]


def test_dapt_resolve_checkpoint_and_helpers(tmp_path):
    from src.dapt import crop_batch, normalise, resolve_checkpoint
    from src.results.run import write_json

    assert resolve_checkpoint(None) is None and resolve_checkpoint("some/dir") == "some/dir"
    dd, cd = tmp_path / "dapt", tmp_path / "ckpt"
    for rid, st, t in [("D01__a__20261118-000000__aaaaaaa", "completed", "2026-11-18T01:00:00Z"),
                       ("D01__a__20261119-000000__bbbbbbb", "failed", "2026-11-19T01:00:00Z"),
                       ("D01__a__20261117-000000__ccccccc", "completed", "2026-11-17T01:00:00Z")]:
        (dd / rid).mkdir(parents=True)
        write_json(dd / rid / "status.json", {"status": st, "finished_utc": t})
        (cd / rid).mkdir(parents=True)
    assert resolve_checkpoint("dapt:D01", dd, cd).endswith("D01__a__20261118-000000__aaaaaaa")
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint("dapt:D02", dd, cd)

    rng = np.random.default_rng(0)
    x, lens = crop_batch([np.ones(10, np.float32), np.arange(3, dtype=np.float32)], 5, rng)
    assert x.shape == (2, 5) and list(lens) == [5, 3] and np.all(x[1, 3:] == 0)
    z = normalise(x, lens)
    assert abs(z[1, :3].mean()) < 1e-6 and np.all(z[1, 3:] == 0)


def test_pack_chunks_keeps_order_and_limit():
    from src.build_dapt_corpus import pack_chunks

    segs = [np.zeros(n) for n in (3, 4, 2, 9, 1)]
    assert pack_chunks(segs, 8) == [[0, 1], [2], [3], [4]]
