"""E05–E09 — linear probes on frozen pretrained encoders.

Embeddings: per-clip, per-layer vectors from ``src.models.encoders`` (frozen model, no training, so
they depend on the clip alone and can be cached across folds without leakage). Cache:
``data/embeddings/<encoder>.npz`` (git-ignored: derived from audio).

Probe (HANDOFF Phase 3): StandardScaler (fit on train) + multinomial LogisticRegression with
class_weight="balanced"; C chosen on validation macro-F1. Logits = decision_function (softmax
logits), calibrated afterwards by temperature scaling on validation like every model.
``model.layer``: "last" (paper rows) or "best_val" (layer chosen on validation macro-F1; note only).
The validation macro-F1 of every layer is always recorded in ``chosen.val_macro_f1_by_layer``.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from src.config import REPO_ROOT
from src.datasets import FoldData, load_waveforms

EMB_DIR = REPO_ROOT / "data" / "embeddings"
_ENCODERS: dict = {}


def get_encoder(key: str, device: str):
    from src.models.encoders import Encoder

    if (key, device) not in _ENCODERS:
        _ENCODERS[(key, device)] = Encoder(key, device)
    return _ENCODERS[(key, device)]


def embeddings_for(key: str, clip_ids, paths, device: str | None = None) -> tuple[np.ndarray, dict]:
    """[n_clips, n_layers, dim] for the given clips, computing and caching any that are missing."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    path = EMB_DIR / f"{key}.npz"
    cache, info = {}, {}
    if path.exists():
        z = np.load(path, allow_pickle=False)
        cache = dict(zip(z["ids"].tolist(), z["emb"]))
        info = {"hf_id": str(z["hf_id"]), "revision": str(z["revision"])}
    todo = [(c, p) for c, p in zip(clip_ids, paths) if c not in cache]
    if todo:
        enc = get_encoder(key, device)
        if info and info["revision"] != str(enc.revision):
            raise RuntimeError(f"{path} was built with revision {info['revision']}, model is {enc.revision}; "
                               "delete the cache to rebuild")
        t0 = time.time()
        for i, (c, p) in enumerate(todo):
            cache[c] = enc.layer_embeddings(load_waveforms([p])[0])
            if (i + 1) % 100 == 0:
                print(f"  [{key}] embedded {i + 1}/{len(todo)} clips ({time.time() - t0:.0f}s)", flush=True)
        info = {"hf_id": enc.hf_id, "revision": str(enc.revision)}
        ids = sorted(cache)
        np.savez(path, ids=np.array(ids, dtype=str), emb=np.stack([cache[i] for i in ids]),
                 hf_id=np.array(info["hf_id"]), revision=np.array(info["revision"]))
    return np.stack([cache[c] for c in clip_ids]), info


def _fit_lr(x, y, c, seed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence warnings on tiny folds are logged via val score
        return LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=seed).fit(x, y)


def _macro_f1(y, pred, k):
    return float(f1_score(y, pred, labels=range(k), average="macro", zero_division=0))


def _logits(clf, scaler, x, k):
    s = np.full((len(x), k), -10.0)
    s[:, clf.classes_] = clf.decision_function(scaler.transform(x)) if len(clf.classes_) > 2 else \
        np.stack([-clf.decision_function(scaler.transform(x)), clf.decision_function(scaler.transform(x))], 1)
    return s


def probe_fit_predict(fd: FoldData, cfg: dict, seed: int) -> dict:
    mcfg = cfg["model"]
    key, layer_mode = mcfg["encoder"], mcfg.get("layer", "last")
    grid_c = [float(c) for c in mcfg.get("grid_C", [1e-3, 1e-2, 1e-1, 1, 10])]
    k = len(fd.labels)
    emb = {}
    for role, part in (("train", fd.train), ("val", fd.val), ("test", fd.test)):
        emb[role], info = embeddings_for(key, part["clip_id"].tolist(), part["path"].tolist())
    ytr, yva = fd.train["label_idx"].to_numpy(), fd.val["label_idx"].to_numpy()
    n_layers = emb["train"].shape[1]

    def search(layer):
        best, log = None, []
        for c in grid_c:
            scaler = StandardScaler().fit(emb["train"][:, layer])
            clf = _fit_lr(scaler.transform(emb["train"][:, layer]), ytr, c, seed)
            f1 = _macro_f1(yva, clf.predict(scaler.transform(emb["val"][:, layer])), k)
            log.append({"layer": layer, "C": c, "val_macro_f1": f1})
            if best is None or f1 > best[0]:
                best = (f1, c, scaler, clf)
        return best, log

    by_layer, logs = {}, []
    for layer in range(n_layers):
        best, log = search(layer)
        by_layer[layer] = best
        logs += log
    if layer_mode == "last":
        layer = n_layers - 1
    elif layer_mode == "best_val":
        layer = max(by_layer, key=lambda i: (by_layer[i][0], i))
    else:
        raise ValueError(f"model.layer must be 'last' or 'best_val', got {layer_mode!r}")
    f1, c, scaler, clf = by_layer[layer]

    def predict_one(wav):
        enc = get_encoder(key, "cpu")
        return _logits(clf, scaler, enc.layer_embeddings(wav)[None, layer], k)

    probe_params = int(clf.coef_.size + clf.intercept_.size)
    enc_params = get_encoder(key, "cpu").n_params
    train_log = [{"epoch": i, "val_macro_f1": r["val_macro_f1"]} for i, r in enumerate(logs)]  # grid order
    return {
        "val_logits": _logits(clf, scaler, emb["val"][:, layer], k),
        "test_logits": _logits(clf, scaler, emb["test"][:, layer], k),
        "params_total": enc_params + probe_params, "params_trainable": probe_params,
        "train_log": train_log, "predict_one": predict_one,
        "chosen": {"encoder": key, "hf_id": info.get("hf_id"), "revision": info.get("revision"),
                   "layer": int(layer), "n_layers": int(n_layers), "C": c, "val_macro_f1": f1,
                   "val_macro_f1_by_layer": {str(i): round(b[0], 4) for i, b in by_layer.items()},
                   "grid": logs},
    }
