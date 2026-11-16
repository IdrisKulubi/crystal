"""Non-neural baselines: E00 majority class, E01 MFCC + SVM.

Every model exposes ``fit_predict(fd: FoldData, cfg, seed) -> dict`` returning
``{"val_logits", "test_logits", "params_total", "params_trainable", "train_log": [rows],
   "predict_one": callable(waveform) -> logits}`` so src/train.py can treat all models alike.
"""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.datasets import FoldData, load_waveforms, pad_or_crop
from src.audio import TARGET_SR


# --------------------------------------------------------------------------- E00
def majority_fit_predict(fd: FoldData, cfg: dict, seed: int) -> dict:
    """Always predicts the training prior (argmax = majority class, i.e. 'hungry').
    Logits are log class frequencies in the training split (+1 smoothing)."""
    k = len(fd.labels)
    counts = np.bincount(fd.train["label_idx"], minlength=k) + 1.0
    logit = np.log(counts / counts.sum())

    def predict_one(_wav):
        return logit[None, :]

    return {
        "val_logits": np.tile(logit, (len(fd.val), 1)),
        "test_logits": np.tile(logit, (len(fd.test), 1)),
        "params_total": k, "params_trainable": k, "train_log": [],
        "predict_one": predict_one, "chosen": {"prior": counts.tolist()},
    }


# --------------------------------------------------------------------------- E01
def mfcc_features(wav: np.ndarray, sr: int = TARGET_SR, n_mfcc: int = 40, seconds: float | None = None) -> np.ndarray:
    """MFCC(40) + delta: mean and std over time -> 160-dim vector."""
    import librosa

    if seconds:
        wav = pad_or_crop(wav, int(seconds * sr))
    m = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc, n_fft=1024, hop_length=256)
    d = librosa.feature.delta(m)
    return np.concatenate([m.mean(1), m.std(1), d.mean(1), d.std(1)]).astype(np.float32)


def svm_fit_predict(fd: FoldData, cfg: dict, seed: int) -> dict:
    mcfg = cfg.get("model", {})
    seconds = cfg["data"].get("clip_seconds")
    feats = {role: np.stack([mfcc_features(w, seconds=None) for w in load_waveforms(part["path"])])
             for role, part in (("train", fd.train), ("val", fd.val), ("test", fd.test))}
    scaler = StandardScaler().fit(feats["train"])
    X = {k: scaler.transform(v) for k, v in feats.items()}
    ytr, yva = fd.train["label_idx"].to_numpy(), fd.val["label_idx"].to_numpy()
    grid_c = mcfg.get("grid_C", [0.1, 1, 10, 100])
    grid_g = mcfg.get("grid_gamma", ["scale", 1e-3, 1e-2, 1e-1])
    log, best = [], None
    for c, g in itertools.product(grid_c, grid_g):
        clf = SVC(C=c, gamma=g, kernel="rbf", class_weight="balanced", probability=True, random_state=seed)
        clf.fit(X["train"], ytr)
        f1 = f1_score(yva, clf.predict(X["val"]), labels=range(len(fd.labels)), average="macro", zero_division=0)
        log.append({"epoch": len(log), "val_macro_f1": f1})
        if best is None or f1 > best[0]:
            best = (f1, c, g, clf)
    _, c, g, clf = best

    def logits(x):
        # predict_proba is ordered by clf.classes_; classes absent from training get ~0 prob
        p = np.full((len(x), len(fd.labels)), 1e-6)
        p[:, clf.classes_] = np.clip(clf.predict_proba(x), 1e-6, 1.0)
        p /= p.sum(1, keepdims=True)
        return np.log(p)

    def predict_one(wav):
        return logits(scaler.transform(mfcc_features(wav)[None, :]))

    n_params = int(clf.support_vectors_.size + clf.dual_coef_.size + clf.intercept_.size)
    return {
        "val_logits": logits(X["val"]), "test_logits": logits(X["test"]),
        "params_total": n_params, "params_trainable": n_params, "train_log": log,
        "predict_one": predict_one, "chosen": {"C": c, "gamma": g, "val_macro_f1": best[0]},
    }
