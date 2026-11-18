"""Phase 6 figures — risk–coverage curves (fig2) and reliability before/after temperature scaling (fig3).

Called by ``python -m src.report``. Both use the saved out-of-fold TEST outputs of every fold x seed:
* "after"  = calibrated probabilities in predictions.csv (temperature fitted on validation only)
* "before" = softmax of test_logits.npy (T = 1)
Which T3 rows are plotted is set in configs/project.yaml -> figures (the first available row of
fig3_rows is used, so the figure shows "Ours" once that run exists). The run ids actually used are
written to results/figures/figure_sources.json and listed in RESULTS.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.results import metrics as M


def load_outputs(run_dir: Path) -> dict:
    """Pooled OOF test outputs over all folds x seeds: y, calibrated probs, uncalibrated probs."""
    ys, cal, raw = [], [], []
    labels = None
    for d in sorted(run_dir.glob("fold_*_seed_*")):
        p = d / "predictions.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        cols = [c for c in df.columns if c.startswith("p_")]
        labs = [c[2:] for c in cols]
        if labels is None:
            labels = labs
        elif labs != labels:
            raise ValueError(f"{d}: label columns differ")
        index = {l: i for i, l in enumerate(labels)}
        ys.append(df["y_true"].map(index).to_numpy())
        cal.append(df[cols].to_numpy(dtype=np.float64))
        lg = d / "test_logits.npy"
        raw.append(M.softmax(np.load(lg), 1.0) if lg.is_file() else None)
    if not ys:
        return {}
    have_raw = all(r is not None and len(r) == len(y) for r, y in zip(raw, ys))
    return {"labels": labels, "y": np.concatenate(ys), "p_cal": np.concatenate(cal),
            "p_raw": np.concatenate(raw) if have_raw else None}


def plot_risk_coverage(curves: list[tuple[str, dict]], out_stem: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for name, o in curves:
        conf = o["p_cal"].max(axis=1)
        correct = (o["p_cal"].argmax(axis=1) == o["y"]).astype(float)
        cov, risk = M.risk_coverage_curve(conf, correct)
        ax.plot(cov, risk, lw=1.4, label=f"{name} (AURC {M.aurc(conf, correct):.3f})")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk (1 − accuracy)")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)


def plot_reliability(name: str, o: dict, out_stem: Path, n_bins: int = 15) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), sharey=True)
    for ax, (title, p) in zip(axes, (("Before (T = 1)", o["p_raw"]), ("After temperature scaling", o["p_cal"]))):
        rb = M.reliability_bins(o["y"], p, n_bins)
        edges = np.array(rb["edges"])
        centers = (edges[:-1] + edges[1:]) / 2
        acc = np.array([a if a is not None else np.nan for a in rb["accuracy"]])
        ax.bar(centers, np.nan_to_num(acc), width=1 / n_bins, edgecolor="k", lw=0.4, alpha=0.8, label="Accuracy")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
        ax.set_title(f"{title}\nECE {M.ece(o['y'], p, n_bins):.3f}", fontsize=8)
        ax.set_xlabel("Confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=6.5, loc="upper left")
    fig.suptitle(name, fontsize=8)
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)


def make_figures(results_dir: Path, chosen: dict, fig_cfg: dict, n_bins: int = 15) -> dict:
    """chosen: {(table,row): registry record}. Returns {figure: [run_ids]} for figure_sources.json."""
    fig_dir = Path(results_dir) / "figures"
    sources: dict[str, list[str]] = {}
    cache: dict[str, dict] = {}

    def outputs(row: str) -> tuple[str, dict] | None:
        rec = chosen.get(("T3", row))
        if not rec or rec.get("task", "reason") != "reason":
            return None
        rid = rec["run_id"]
        if rid not in cache:
            cache[rid] = load_outputs(Path(results_dir) / "runs" / rid)
        return (rid, cache[rid]) if cache[rid] else None

    curves = []
    for row in fig_cfg.get("fig2_rows", []):
        got = outputs(row)
        if got:
            curves.append((row, got[1]))
            sources.setdefault("fig2_risk_coverage", []).append(got[0])
    if curves:
        plot_risk_coverage(curves, fig_dir / "fig2_risk_coverage")
    for row in fig_cfg.get("fig3_rows", []):
        got = outputs(row)
        if got and got[1]["p_raw"] is not None:
            plot_reliability(row, got[1], fig_dir / "fig3_reliability_before_after", n_bins)
            sources["fig3_reliability_before_after"] = [got[0]]
            break
    return sources
