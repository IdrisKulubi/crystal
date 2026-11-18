"""Paired significance test between two reason-task runs on the SAME grouped folds.

    python -m src.significance --a <run_id> --b <run_id> [--resamples 10000]

Test: paired cluster bootstrap over groups (babies / app installs). Both runs' out-of-fold test
predictions are joined on (clip_id, seed); each resample draws groups with replacement, and the
statistic is macro-F1(a) - macro-F1(b) on the pooled out-of-fold predictions, averaged over seeds.
Resampling whole groups keeps clips of one baby together (the unit of independence, as in the splits).

* delta_macro_f1  observed delta (full data, mean over seeds)
* ci95            2.5 / 97.5 percentiles of the bootstrap deltas
* p_value         two-sided: 2 * min(P(delta* <= 0), P(delta* >= 0)), floor 1/(B+1)

Writes results/significance/<a>__vs__<b>.json (report.py reads it).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_project
from src.results.run import DEFAULT_RESULTS_DIR, iso, utcnow, write_json


def load_oof(run_dir: Path) -> pd.DataFrame:
    files = sorted(run_dir.glob("fold_*_seed_*/predictions.csv"))
    if not files:
        raise FileNotFoundError(f"no predictions.csv under {run_dir}")
    df = pd.concat([pd.read_csv(p, usecols=["clip_id", "group", "seed", "y_true", "y_pred"]) for p in files],
                   ignore_index=True)
    if df.duplicated(["clip_id", "seed"]).any():
        raise ValueError(f"{run_dir.name}: a clip appears twice for one seed (not out-of-fold)")
    return df


def _macro_f1_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    """Macro-F1 over the class axis (last); zero_division=0 like sklearn."""
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return f1.mean(axis=-1)


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, resamples: int = 10000, seed: int = 0) -> dict:
    m = a.merge(b, on=["clip_id", "seed"], suffixes=("_a", "_b"))
    if len(m) != len(a) or len(m) != len(b):
        raise ValueError(f"runs do not cover the same (clip, seed) pairs: {len(a)} vs {len(b)}, joined {len(m)}")
    if (m["y_true_a"] != m["y_true_b"]).any() or (m["group_a"] != m["group_b"]).any():
        raise ValueError("labels or groups differ between runs (different folds/label set?)")
    labels = sorted(set(m["y_true_a"]))
    lab = {l: i for i, l in enumerate(labels)}
    seeds = sorted(m["seed"].unique())
    groups = sorted(m["group_a"].unique())
    gidx = {g: i for i, g in enumerate(groups)}
    G, S, K = len(groups), len(seeds), len(labels)
    sidx = {s: i for i, s in enumerate(seeds)}

    # per (model, group, seed, class) counts of tp / fp / fn -> any resample is a weighted sum over groups
    counts = np.zeros((2, G, S, 3, K))
    y = m["y_true_a"].map(lab).to_numpy()
    g = m["group_a"].map(gidx).to_numpy()
    s = m["seed"].map(sidx).to_numpy()
    for mi, col in enumerate(("y_pred_a", "y_pred_b")):
        p = m[col].map(lambda v: lab.get(v, -1)).to_numpy()
        hit = p == y
        np.add.at(counts[mi], (g[hit], s[hit], 0, y[hit]), 1)          # tp
        miss = ~hit
        pm = miss & (p >= 0)
        np.add.at(counts[mi], (g[pm], s[pm], 1, p[pm]), 1)             # fp for predicted class
        np.add.at(counts[mi], (g[miss], s[miss], 2, y[miss]), 1)       # fn for true class

    def stat(weights: np.ndarray) -> np.ndarray:
        # weights (R, G) -> delta (R,)
        tot = np.einsum("rg,mgsck->rmsck", weights, counts)
        f1 = _macro_f1_counts(tot[..., 0, :], tot[..., 1, :], tot[..., 2, :])  # (R, 2, S)
        f1 = f1.mean(axis=-1)
        return f1[:, 0] - f1[:, 1], f1

    obs_delta, obs_f1 = stat(np.ones((1, G)))
    rng = np.random.default_rng(seed)
    deltas = []
    for start in range(0, resamples, 500):
        r = min(500, resamples - start)
        draw = rng.integers(0, G, size=(r, G))
        w = np.zeros((r, G))
        np.add.at(w, (np.repeat(np.arange(r), G), draw.ravel()), 1)
        deltas.append(stat(w)[0])
    d = np.concatenate(deltas)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    p = float(min(1.0, max(p, 1.0 / (resamples + 1))))
    return {"delta_macro_f1": float(obs_delta[0]),
            "macro_f1_a_pooled": float(obs_f1[0, 0]), "macro_f1_b_pooled": float(obs_f1[0, 1]),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_value": p, "n_clips": int(m["clip_id"].nunique()), "n_groups": G, "seeds": [int(x) for x in seeds],
            "labels": labels, "resamples": int(resamples), "rng_seed": int(seed),
            "method": "paired cluster bootstrap over groups; pooled OOF macro-F1 averaged over seeds"}


def compare(run_a: str, run_b: str, results_dir: Path = DEFAULT_RESULTS_DIR, resamples: int | None = None) -> Path:
    results_dir = Path(results_dir)
    resamples = resamples or load_project()["evaluation"].get("bootstrap_resamples", 10000)
    res = paired_bootstrap(load_oof(results_dir / "runs" / run_a), load_oof(results_dir / "runs" / run_b), resamples)
    out = results_dir / "significance" / f"{run_a}__vs__{run_b}.json"
    write_json(out, {"run_a": run_a, "run_b": run_b, **res, "created_utc": iso(utcnow())})
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--resamples", type=int, default=None)
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    args = ap.parse_args(argv)
    out = compare(args.a, args.b, Path(args.results_dir), args.resamples)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
