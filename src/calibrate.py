"""Temperature scaling and selective-prediction thresholds — fitted on VALIDATION only.

Used inside src/train.py for every run; ``python -m src.calibrate --run <run_id>`` recomputes
calibration.json / predictions.csv / metrics.json for an existing run from its saved logits
(e.g. after changing the calibration method) without retraining.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from src.results import metrics as M


def fit_temperature(val_logits: np.ndarray, val_y: np.ndarray, bounds=(0.05, 20.0)) -> float:
    """T > 0 minimising validation NLL of softmax(logits / T) (Guo et al., 2017)."""
    val_logits = np.asarray(val_logits, dtype=np.float64)
    val_y = np.asarray(val_y).astype(int)
    if len(val_y) == 0:
        return 1.0

    def nll(log_t: float) -> float:
        return M.nll(val_y, M.softmax(val_logits, float(np.exp(log_t))))

    res = minimize_scalar(nll, bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded",
                          options={"xatol": 1e-4})
    t = float(np.exp(res.x))
    # never accept a temperature that makes validation NLL worse than T=1
    return t if nll(np.log(t)) <= nll(0.0) else 1.0


def calibrate(val_logits: np.ndarray, val_y: np.ndarray, coverages=(0.9, 0.7, 0.5), n_bins: int = 15) -> dict:
    """Return calibration.json content (RESULTS_SPEC §6)."""
    t = fit_temperature(val_logits, val_y)
    p_before = M.softmax(val_logits, 1.0)
    p_after = M.softmax(val_logits, t)
    conf = p_after.max(axis=1)
    return {
        "method": "temperature_scaling",
        "temperature": t,
        "fit_on": "val",
        "n_val": int(len(val_y)),
        "ece_val_before": M.ece(val_y, p_before, n_bins),
        "ece_val_after": M.ece(val_y, p_after, n_bins),
        "nll_val_before": M.nll(val_y, p_before),
        "nll_val_after": M.nll(val_y, p_after),
        "thresholds": {str(c): M.threshold_for_coverage(conf, c) for c in coverages},
    }
