"""Dataset statistics for paper Table 1 / Section 3 (RESULTS_SPEC §9) + Figure 5.

    python -m src.dataset_stats

Reads data/metadata.csv, data/metadata_folds.csv, data/dedup_report.csv.
Writes results/dataset_stats.json and results/figures/fig5_class_distribution.{png,pdf}.
Contains counts only — no audio, no file names of individual clips.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import REPO_ROOT, load_project
from src.results.run import write_json

# Okabe-Ito colour-blind-safe palette
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]


def describe(df: pd.DataFrame, label_col: str, labels: list[str] | None = None) -> dict:
    counts = df[label_col].value_counts()
    if labels is not None:
        counts = counts.reindex(labels, fill_value=0)
    per_group = df.groupby("group").size()
    d = df["duration_s"]
    return {
        "n_clips": int(len(df)),
        "n_groups": int(df["group"].nunique()),
        "total_minutes": round(float(d.sum() / 60), 2),
        "duration_s": {"mean": round(float(d.mean()), 3), "std": round(float(d.std(ddof=1)), 3) if len(d) > 1 else 0.0,
                       "min": round(float(d.min()), 3), "max": round(float(d.max()), 3)},
        "class_counts": {k: int(v) for k, v in counts.items()},
        "class_percent": {k: round(100 * float(v) / max(1, len(df)), 1) for k, v in counts.items()},
        "groups_per_class": {k: int(df.loc[df[label_col] == k, "group"].nunique()) for k in counts.index},
        "clips_per_group": {"min": int(per_group.min()), "median": float(per_group.median()),
                            "max": int(per_group.max())},
    }


def overlap_stats(meta: pd.DataFrame, pairs: pd.DataFrame, eval_ds: str) -> dict:
    """How much of each non-eval dataset duplicates the eval dataset (and itself)."""
    out = {}
    eval_clusters = set(meta.loc[meta["dataset"] == eval_ds, "dup_cluster"])
    for ds, g in meta.groupby("dataset"):
        if ds == eval_ds:
            continue
        within = pairs[(pairs["dataset_a"] == ds) & (pairs["dataset_b"] == ds)]
        out[ds] = {
            "n_clips": int(len(g)),
            "n_clips_duplicating_eval_set": int(g["dup_cluster"].isin(eval_clusters).sum()),
            "n_generated_copies": int(g["is_generated"].sum()),
            "n_within_dataset_duplicate_pairs": {k: int(v) for k, v in within["kind"].value_counts().items()},
            "n_within_dataset_duplicates_removed": int(g["is_duplicate"].sum()),
            "orig_sample_rates": {str(k): int(v) for k, v in g["orig_sr"].value_counts().items()},
        }
    return out


def plot_class_distribution(stats: dict, labels: list[str], out_stem) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = [stats["class_counts"].get(lab, 0) for lab in labels]
    groups = [stats["groups_per_class"].get(lab, 0) for lab in labels]
    fig, ax = plt.subplots(figsize=(3.35, 2.3))  # ~8.5 cm single column
    x = np.arange(len(labels))
    ax.bar(x - 0.2, counts, 0.4, label="clips", color=PALETTE[0])
    ax.bar(x + 0.2, groups, 0.4, label="app installs (groups)", color=PALETTE[1])
    for xi, c in zip(x, counts):
        ax.text(xi - 0.2, c, str(c), ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x, [lab.replace("_", " ") for lab in labels], fontsize=7, rotation=20)
    ax.set_ylabel("count", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=6, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=300)
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args(argv)
    project = load_project()
    meta = pd.read_csv(REPO_ROOT / "data" / "metadata.csv")
    folds = pd.read_csv(REPO_ROOT / "data" / "metadata_folds.csv")
    pairs = pd.read_csv(REPO_ROOT / "data" / "dedup_report.csv")
    split_ls = project["split_label"]

    stats: dict = {"label_set_for_splits": split_ls, "datasets": {}, "label_sets": {}, "per_fold_test_counts": {}}
    names = []
    for ds in project["eval_datasets"]:
        f = folds[folds["dataset"] == ds]
        spec = project["datasets"][ds]
        names.append(spec.get("display_name", ds))
        raw = meta[meta["dataset"] == ds]
        s = describe(f, f"label_{split_ls}", project["label_sets"][split_ls])
        s.update({
            "display_name": spec.get("display_name", ds),
            "licence": spec.get("licence"),
            "n_clips_raw": int(len(raw)),
            "n_duplicates_removed": int(raw["is_duplicate"].sum()),
            "n_groups_merged_by_dedup": int(raw["group_raw"].nunique() - raw["group"].nunique()),
            "orig_sample_rates": {str(k): int(v) for k, v in raw["orig_sr"].value_counts().items()},
            "age_bracket_counts": {str(k): int(v) for k, v in f["age_bracket"].value_counts().sort_index().items()},
            "gender_counts": {str(k): int(v) for k, v in f["gender"].value_counts().items()},
            "platform_counts": {str(k): int(v) for k, v in f["platform"].value_counts().items()},
            "label_orig_counts": {str(k): int(v) for k, v in raw["label_orig"].value_counts().items()},
        })
        stats["datasets"][ds] = s
        stats["per_fold_test_counts"][ds] = pd.crosstab(f["fold"], f[f"label_{split_ls}"]).to_dict(orient="index")
        for ls, labels in project["label_sets"].items():
            col = f"label_{ls}"
            if col in f:
                stats["label_sets"].setdefault(ls, {})[ds] = {
                    "labels": labels,
                    "mapping": project["label_maps"][ls],
                    "class_counts": {k: int(v) for k, v in f[col].value_counts().reindex(labels, fill_value=0).items()},
                    "min_class_count": int(f[col].value_counts().reindex(labels, fill_value=0).min()),
                }
        plot_class_distribution(s, project["label_sets"][split_ls],
                                REPO_ROOT / "results" / "figures" / "fig5_class_distribution")
    stats["names_for_abstract"] = " and ".join(names)
    stats["overlap"] = overlap_stats(meta, pairs, project["eval_datasets"][0])
    stats["n_duplicate_pairs_all"] = {k: int(v) for k, v in pairs["kind"].value_counts().items()}
    write_json(REPO_ROOT / "results" / "dataset_stats.json", stats)
    for ds, s in stats["datasets"].items():
        print(f"{ds}: {s['n_clips']} clips, {s['n_groups']} groups, {s['total_minutes']} min; {s['class_counts']}")
    print("wrote results/dataset_stats.json and results/figures/fig5_class_distribution.{png,pdf}")
    return stats


if __name__ == "__main__":
    main()
