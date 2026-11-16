"""ALL metric definitions for the project (RESULTS_SPEC §5). Unit-tested in tests/test_metrics.py.

Conventions
-----------
* ``y`` is an int array of label indices in ``[0, n_classes)``.
* ``probs`` is an ``[n, n_classes]`` array of probabilities (rows sum to 1).
* Accuracy-like values are fractions in [0, 1]; report.py formats them as percentages.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

EPS = 1e-12


# --------------------------------------------------------------------------- basics
def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _check(y: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] != y.shape[0]:
        raise ValueError(f"shape mismatch: y {y.shape}, probs {probs.shape}")
    return y, probs


# --------------------------------------------------------------------------- calibration
def ece(y: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error: 15 equal-width confidence bins, |acc - conf| weighted by
    bin fraction. Bin i covers [i/B, (i+1)/B); confidence 1.0 goes in the last bin."""
    y, probs = _check(y, probs)
    if len(y) == 0:
        return float("nan")
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(np.float64)
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def reliability_bins(y: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> dict:
    """Per-bin accuracy / confidence / count, for the reliability diagram (fig3)."""
    y, probs = _check(y, probs)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(np.float64)
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    acc, cnf, cnt = [], [], []
    for b in range(n_bins):
        m = idx == b
        cnt.append(int(m.sum()))
        acc.append(float(correct[m].mean()) if m.any() else None)
        cnf.append(float(conf[m].mean()) if m.any() else None)
    return {"edges": np.linspace(0, 1, n_bins + 1).tolist(), "accuracy": acc,
            "confidence": cnf, "count": cnt}


def nll(y: np.ndarray, probs: np.ndarray) -> float:
    """Mean negative log-likelihood of the true class."""
    y, probs = _check(y, probs)
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], EPS, 1.0))))


def brier(y: np.ndarray, probs: np.ndarray) -> float:
    """Multi-class Brier score: mean over samples of sum over classes (p_k - onehot_k)^2."""
    y, probs = _check(y, probs)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


# --------------------------------------------------------------------------- selective prediction
def risk_coverage_curve(confidence: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by confidence (desc, stable); keep the top k for k = 1..n.
    Returns (coverage k/n, selective risk = 1 - accuracy on kept)."""
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    n = len(confidence)
    if n == 0:
        return np.array([]), np.array([])
    order = np.argsort(-confidence, kind="stable")
    cum_correct = np.cumsum(correct[order])
    k = np.arange(1, n + 1)
    return k / n, 1.0 - cum_correct / k


def aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve, trapezoid rule over all n points."""
    cov, risk = risk_coverage_curve(confidence, correct)
    if len(cov) == 0:
        return float("nan")
    if len(cov) == 1:
        return float(risk[0])
    return float(np.sum((cov[1:] - cov[:-1]) * (risk[1:] + risk[:-1]) / 2.0))


def threshold_for_coverage(val_confidence: np.ndarray, coverage: float) -> float:
    """Confidence threshold tau such that keeping ``conf >= tau`` on the VALIDATION set keeps
    ceil(coverage * n) clips (ties may keep slightly more). coverage=1.0 -> keep everything."""
    conf = np.sort(np.asarray(val_confidence, dtype=np.float64))[::-1]
    if len(conf) == 0:
        raise ValueError("empty validation confidences")
    if coverage >= 1.0:
        return float(conf[-1])
    k = max(1, math.ceil(coverage * len(conf)))
    return float(conf[k - 1])


def selective_accuracy(confidence: np.ndarray, correct: np.ndarray, tau: float) -> tuple[float, float]:
    """(accuracy on clips with conf >= tau, realised coverage). Accuracy is NaN if none kept.
    RESULTS_SPEC calls this 'precision_at_coverage'."""
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    keep = confidence >= tau
    cov = float(keep.mean()) if len(keep) else float("nan")
    acc = float(correct[keep].mean()) if keep.any() else float("nan")
    return acc, cov


# --------------------------------------------------------------------------- classification
def top_k_accuracy(y: np.ndarray, probs: np.ndarray, k: int = 2) -> float:
    y, probs = _check(y, probs)
    k = min(k, probs.shape[1])
    topk = np.argsort(-probs, axis=1, kind="stable")[:, :k]
    return float(np.mean([y[i] in topk[i] for i in range(len(y))]))


def classification_metrics(
    y: np.ndarray,
    probs: np.ndarray,
    labels: Sequence[str],
    probs_uncal: np.ndarray | None = None,
    thresholds: dict[str, float] | None = None,
    coverages: Sequence[float] = (0.9, 0.7, 0.5),
    n_bins: int = 15,
) -> dict:
    """Everything in RESULTS_SPEC §5 that can be computed from predictions.

    ``probs``       calibrated probabilities (or raw if not calibrated yet).
    ``probs_uncal`` raw probabilities (T=1); defaults to ``probs``.
    ``thresholds``  {"0.9": tau, ...} chosen on validation. If missing, precision_at_coverage
                    for those coverages is None (never chosen on test).
    """
    y, probs = _check(y, probs)
    probs_uncal = probs if probs_uncal is None else np.asarray(probs_uncal, dtype=np.float64)
    n_classes = len(labels)
    if probs.shape[1] != n_classes:
        raise ValueError(f"probs has {probs.shape[1]} columns but {n_classes} labels")
    class_ids = list(range(n_classes))
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (pred == y).astype(np.float64)

    p, r, f, s = precision_recall_fscore_support(y, pred, labels=class_ids, zero_division=0)
    per_class = {
        lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i, lab in enumerate(labels)
    }

    pac: dict[str, float | None] = {"1.0": float(correct.mean()) if len(y) else float("nan")}
    realised: dict[str, float | None] = {}
    for c in coverages:
        key = str(c)
        tau = (thresholds or {}).get(key)
        if tau is None:
            pac[key], realised[key] = None, None
        else:
            pac[key], realised[key] = selective_accuracy(conf, correct, tau)

    return {
        "n_test": int(len(y)),
        "n_classes": n_classes,
        "labels": list(labels),
        "accuracy": float(correct.mean()),
        "macro_f1": float(f1_score(y, pred, labels=class_ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=class_ids, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y, pred, labels=class_ids).tolist(),
        "nll": nll(y, probs),
        "brier": brier(y, probs),
        "ece_15_uncal": ece(y, probs_uncal, n_bins),
        "ece_15_cal": ece(y, probs, n_bins),
        "aurc": aurc(conf, correct),
        "precision_at_coverage": pac,
        "realised_coverage": realised,
        "top2_accuracy": top_k_accuracy(y, probs, 2),
    }


# --------------------------------------------------------------------------- detection
def detection_metrics(y: np.ndarray, scores: np.ndarray, threshold: float = 0.5,
                      false_alarms: int | None = None, hours: float | None = None) -> dict:
    """Binary cry (1) vs non-cry (0). ``scores`` = P(cry).
    false_alarms_per_hour is computed from a long held-out household recording
    (``false_alarms`` events over ``hours``), not from the clip test set."""
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    pred = (scores >= threshold).astype(int)
    recall = float(recall_score(y, pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y, scores))
    except ValueError:  # only one class present
        auc = float("nan")
    return {
        "n_test": int(len(y)),
        "threshold": float(threshold),
        "accuracy": float((pred == y).mean()),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": recall,
        "miss_rate": 1.0 - recall,
        "roc_auc": auc,
        "false_alarms_per_hour": false_alarms_per_hour(false_alarms, hours),
    }


def false_alarms_per_hour(n_false_alarms: int | None, hours: float | None) -> float | None:
    if n_false_alarms is None or not hours:
        return None
    return float(n_false_alarms) / float(hours)
