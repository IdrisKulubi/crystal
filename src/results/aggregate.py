"""Aggregate per-fold/seed outputs of one run into summary.json (RESULTS_SPEC §7).

* ``metrics``: for every scalar metric (nested keys flattened with dots, e.g.
  ``per_class.hungry.recall`` or ``precision_at_coverage.0.9``) ->
  ``{"mean", "std", "n", "values"}`` over all fold x seed combinations.
  std is the sample standard deviation (ddof=1); 0.0 when n == 1.
* ``calibration``: the same for scalars in calibration.json (temperature, thresholds...).
* ``pooled``: metrics on the concatenated out-of-fold predictions of seed 0
  (used for the confusion-matrix figure and significance tests).
* ``per_class_recall_mean_std``: convenience copy for figure A.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from src.config import yaml_load

from src.results import metrics as M
from src.results.run import read_json, write_json

FOLD_DIR_RE = re.compile(r"^fold_(\d+)_seed_(\d+)$")


def flatten(d: dict, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, bool) or v is None:
            continue
        elif isinstance(v, (int, float, np.integer, np.floating)):
            fv = float(v)
            if math.isfinite(fv):
                out[key] = fv
    return out


def mean_std(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "n": int(len(a)), "values": [float(v) for v in a]}


def _collect(dirs: list[Path], filename: str) -> dict[str, dict]:
    vals: dict[str, list[float]] = {}
    for d in dirs:
        p = d / filename
        if p.is_file():
            for k, v in flatten(read_json(p)).items():
                vals.setdefault(k, []).append(v)
    return {k: mean_std(v) for k, v in sorted(vals.items())}


def pooled_metrics(pred_files: list[Path], coverages=(0.9, 0.7, 0.5)) -> dict | None:
    """Metrics on concatenated out-of-fold predictions (calibrated p_* columns)."""
    frames = [pd.read_csv(p) for p in pred_files if p.is_file()]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    prob_cols = [c for c in df.columns if c.startswith("p_")]
    labels = [c[2:] for c in prob_cols]
    if not labels or "y_true" not in df:
        return None
    index = {lab: i for i, lab in enumerate(labels)}
    y = df["y_true"].map(index).to_numpy()
    probs = df[prob_cols].to_numpy(dtype=np.float64)
    out = M.classification_metrics(y, probs, labels, coverages=coverages)
    out["n_folds"] = int(df["fold"].nunique()) if "fold" in df else None
    return out


def aggregate_run(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    with open(run_dir / "config.yaml", encoding="utf-8") as f:
        cfg = yaml_load(f) or {}
    folds = list(cfg.get("eval", {}).get("folds", []))
    seeds = list(cfg.get("eval", {}).get("seeds", []))

    found: list[tuple[int, int, Path]] = []
    for d in sorted(run_dir.iterdir()):
        m = FOLD_DIR_RE.match(d.name)
        if d.is_dir() and m and (d / "metrics.json").is_file():
            found.append((int(m.group(1)), int(m.group(2)), d))
    have = {(f, s) for f, s, _ in found}
    expected = [(f, s) for f in folds for s in seeds]
    missing = [f"fold_{f}_seed_{s}" for f, s in expected if (f, s) not in have]

    dirs = [d for _, _, d in found]
    metrics = _collect(dirs, "metrics.json")
    calibration = _collect(dirs, "calibration.json")

    seed0 = min(seeds) if seeds else 0
    pooled = pooled_metrics([d / "predictions.csv" for f, s, d in found if s == seed0])

    per_class_recall = {
        k.split(".")[1]: v for k, v in metrics.items()
        if k.startswith("per_class.") and k.endswith(".recall")
    }
    summary = {
        "run_id": run_dir.name,
        "n_expected": len(expected),
        "n_found": len(found),
        "missing": missing,
        "metrics": metrics,
        "calibration": calibration,
        "pooled_seed": seed0,
        "pooled": pooled,
        "per_class_recall_mean_std": per_class_recall,
    }
    write_json(run_dir / "summary.json", summary)
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="(Re)build summary.json for a run folder")
    ap.add_argument("run_dir")
    s = aggregate_run(ap.parse_args().run_dir)
    print(f"{s['run_id']}: {s['n_found']}/{s['n_expected']} fold x seed results aggregated")
