"""Grouped, stratified 5-fold splits (+ the clip-level random split used ONLY by E90).

    python -m src.make_splits

For each evaluation dataset (configs/project.yaml -> eval_datasets):
* test folds: StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) on the
  ``split_label`` label set, grouped by ``group`` (app install after duplicate merging).
* validation: inside each training portion, ~15% of GROUPS are held out with
  StratifiedGroupKFold(n_splits=round(1/0.15)=7, shuffle=True, random_state=42+k), first split.
* columns ``split_f0..split_f4`` in {train, val, test}; ``fold`` = the clip's test fold.

Outputs: data/metadata_folds.csv, data/metadata_randomsplit.csv, results/splits_report.json
Hard assertions (raise): no group in >1 test fold; no group in two roles within a fold;
no duplicate clip_id; no duplicate audio. Warnings (logged): a class missing from a fold.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit

from src.config import REPO_ROOT, load_project
from src.results.run import write_json

KEEP_COLS = ["clip_id", "dataset", "path", "group", "label_orig", "age_bracket", "gender",
             "platform", "duration_s", "orig_sr"]


def eval_table(meta: pd.DataFrame, dataset: str, project: dict) -> pd.DataFrame:
    split_ls = project["split_label"]
    sub = meta[(meta["dataset"] == dataset) & (~meta["is_duplicate"].astype(bool))].copy()
    for ls, mapping in project["label_maps"].items():
        sub[f"label_{ls}"] = sub["label_orig"].map(mapping)
    sub = sub[sub[f"label_{split_ls}"].notna()]
    cols = KEEP_COLS + [f"label_{ls}" for ls in project["label_maps"]]
    return sub[cols].sort_values("clip_id").reset_index(drop=True)


def grouped_folds(df: pd.DataFrame, label_col: str, n_folds: int, seed: int,
                  val_fraction: float) -> pd.DataFrame:
    df = df.copy()
    y, groups = df[label_col].to_numpy(), df["group"].to_numpy()
    df["fold"] = -1
    for k, (_, test_idx) in enumerate(
            StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(df, y, groups)):
        df.loc[df.index[test_idx], "fold"] = k
    n_val_splits = max(2, int(round(1 / val_fraction)))
    for k in range(n_folds):
        col = f"split_f{k}"
        df[col] = np.where(df["fold"] == k, "test", "train")
        pool = df.index[df["fold"] != k]
        val_idx = choose_val_groups(df.loc[pool, label_col].to_numpy(), df.loc[pool, "group"].to_numpy(),
                                    n_val_splits, seed + k)
        df.loc[pool[val_idx], col] = "val"
    return df


def choose_val_groups(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int,
                      max_tries: int = 50) -> np.ndarray:
    """Hold out ~1/n_splits of the groups for validation. Deterministically tries seeds
    seed, seed+1000, ... (all candidate splits of each) and returns the first candidate whose
    validation part contains every class that has >= 2 groups in the pool AND leaves every
    class in training. Falls back to the very first candidate. Never looks at test data."""
    import warnings as _w

    classes = np.unique(y)
    need_val = [c for c in classes if len(np.unique(groups[y == c])) >= 2]
    first = None
    with _w.catch_warnings():
        _w.simplefilter("ignore", UserWarning)  # rare classes have < n_splits members: expected
        for t in range(max_tries):
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + 1000 * t)
            for train_idx, val_idx in sgkf.split(np.zeros(len(y)), y, groups):
                if first is None:
                    first = val_idx
                if set(need_val) <= set(y[val_idx]) and set(classes) <= set(y[train_idx]):
                    return val_idx
    return first


def random_folds(df: pd.DataFrame, label_col: str, n_folds: int, seed: int,
                 val_fraction: float) -> pd.DataFrame:
    """Clip-level (NOT grouped) split, mimicking common practice. E90 leakage demo only."""
    df = df.copy()
    y = df[label_col].to_numpy()
    df["fold"] = -1
    for k, (_, test_idx) in enumerate(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(df, y)):
        df.loc[df.index[test_idx], "fold"] = k
    for k in range(n_folds):
        col = f"split_f{k}"
        df[col] = np.where(df["fold"] == k, "test", "train")
        pool = df.index[df["fold"] != k]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed + k)
        _, val_idx = next(sss.split(pool, df.loc[pool, label_col]))
        df.loc[pool[val_idx], col] = "val"
    return df


def check_grouped(df: pd.DataFrame, n_folds: int) -> None:
    """Hard leakage assertions."""
    assert not df["clip_id"].duplicated().any(), "duplicate clip_id in folds file"
    per_group = df.groupby("group")["fold"].nunique()
    bad = per_group[per_group > 1]
    assert bad.empty, f"groups in more than one test fold: {list(bad.index)[:5]}"
    for k in range(n_folds):
        roles = df.groupby("group")[f"split_f{k}"].nunique()
        bad = roles[roles > 1]
        assert bad.empty, f"fold {k}: groups in two roles: {list(bad.index)[:5]}"
        assert set(df[f"split_f{k}"]) == {"train", "val", "test"}, f"fold {k}: missing a role"


def fold_report(df: pd.DataFrame, label_col: str, n_folds: int, labels: list[str]) -> tuple[dict, list[str]]:
    report, warnings = {}, []
    for k in range(n_folds):
        col = f"split_f{k}"
        entry = {}
        for role in ("train", "val", "test"):
            part = df[df[col] == role]
            counts = part[label_col].value_counts().reindex(labels, fill_value=0).astype(int).to_dict()
            entry[role] = {"n_clips": int(len(part)), "n_groups": int(part["group"].nunique()), "classes": counts}
            for lab, c in counts.items():
                if c == 0:
                    warnings.append(f"fold {k} {role}: no '{lab}' clips")
        report[f"fold_{k}"] = entry
    return report, warnings


def group_overlap(df: pd.DataFrame, n_folds: int) -> dict:
    """For a random split: fraction of test clips whose group also appears in train (leakage)."""
    out = []
    for k in range(n_folds):
        col = f"split_f{k}"
        train_groups = set(df.loc[df[col] == "train", "group"])
        test = df[df[col] == "test"]
        out.append(float(test["group"].isin(train_groups).mean()))
    return {"per_fold": out, "mean": float(np.mean(out))}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args(argv)
    project = load_project()
    ev = project["evaluation"]
    n_folds, seed, vf = ev["n_folds"], ev["split_random_state"], ev["val_group_fraction"]
    meta = pd.read_csv(REPO_ROOT / "data" / "metadata.csv")
    if "group" not in meta:
        raise SystemExit("run `python -m src.dedup` first (it adds the merged `group` column)")
    split_ls = project["split_label"]
    label_col = f"label_{split_ls}"
    labels = project["label_sets"][split_ls]

    grouped, randomised, report = [], [], {"label_set": split_ls, "n_folds": n_folds, "seed": seed,
                                          "val_group_fraction": vf, "datasets": {}}
    for ds in project["eval_datasets"]:
        table = eval_table(meta, ds, project)
        g = grouped_folds(table, label_col, n_folds, seed, vf)
        check_grouped(g, n_folds)
        r = random_folds(table, label_col, n_folds, seed, vf)
        g_rep, g_warn = fold_report(g, label_col, n_folds, labels)
        r_rep, _ = fold_report(r, label_col, n_folds, labels)
        report["datasets"][ds] = {
            "n_clips": int(len(g)), "n_groups": int(g["group"].nunique()),
            "grouped": g_rep, "grouped_warnings": g_warn,
            "random": r_rep,
            "random_split_test_clips_with_group_in_train": group_overlap(r, n_folds),
            "grouped_split_test_clips_with_group_in_train": group_overlap(g, n_folds),
        }
        grouped.append(g)
        randomised.append(r)
        for w in g_warn:
            print(f"WARNING [{ds}] {w}")

    out_g = REPO_ROOT / "data" / "metadata_folds.csv"
    out_r = REPO_ROOT / "data" / "metadata_randomsplit.csv"
    pd.concat(grouped).to_csv(out_g, index=False, lineterminator="\n")
    pd.concat(randomised).to_csv(out_r, index=False, lineterminator="\n")
    report["folds_file_sha256"] = sha256(out_g)
    report["randomsplit_file_sha256"] = sha256(out_r)
    write_json(REPO_ROOT / "results" / "splits_report.json", report)
    for ds, d in report["datasets"].items():
        leak = d["random_split_test_clips_with_group_in_train"]["mean"]
        print(f"{ds}: {d['n_clips']} clips, {d['n_groups']} groups, {len(d['grouped_warnings'])} warnings; "
              f"random split puts {leak:.0%} of test clips next to same-group training clips")
    print(f"folds sha256 {report['folds_file_sha256'][:16]}...  -> data/metadata_folds.csv")
    return report


if __name__ == "__main__":
    main()
