"""Phase 7 — build the cry-DETECTION table (cry vs non-cry) and its grouped folds.

    python -m src.prepare_detection [--regenerate]

Needs: ``python -m src.prepare_data && python -m src.dedup && python -m src.make_splits`` (Donate-a-Cry),
``python -m src.build_dapt_corpus`` (CryCeleb chunks) and ESC-50 cloned to ``data/raw/ESC-50``.

Sources (label set ``detect_v1`` = [non_cry, cry]):
* cry      donateacry  every clip of the reason benchmark (data/metadata_folds.csv); group = app install
* cry      cryceleb    ONE chunk per baby (the first chunk, data/dapt/manifest.csv); group = baby id.
                       (CryCeleb is CC BY-NC-ND: research use only, nothing derived is released.)
* cry      esc50       ESC-50 class ``crying_baby`` (40 clips); group = freesound source file
* non_cry  esc50       the other 49 ESC-50 classes (1,960 clips: animals, nature, human non-speech such as
                       laughing/coughing/sneezing, domestic and urban sounds); group = freesound source file
ESC-50 is CC BY-NC 3.0. It is resampled to 16 kHz into data/processed/esc50/ (git-ignored).

Folds: 5 x StratifiedGroupKFold on ``strat`` = source x label (so every test fold has all four parts),
validation groups chosen inside the training folds exactly as for the reason task. The assignment is
committed in ``splits/detect_v1.csv`` (clip ids only) and re-applied unless ``--regenerate``.

Outputs: data/detection_folds.csv, results/detection_splits_report.json
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.audio import TARGET_SR, load_audio, save_audio
from src.config import REPO_ROOT, load_project
from src.make_splits import (SPLITS_DIR, check_grouped, grouped_folds, load_assignment, save_assignment, sha256,
                             split_cols)
from src.results.run import write_json

ESC = REPO_ROOT / "data" / "raw" / "ESC-50"
OUT = REPO_ROOT / "data" / "detection_folds.csv"
ASSIGN = SPLITS_DIR / "detect_v1.csv"
LABEL_COL = "label_detect_v1"


def esc50_table() -> pd.DataFrame:
    meta = pd.read_csv(ESC / "meta" / "esc50.csv")
    rows = []
    for r in meta.itertuples():
        out = REPO_ROOT / "data" / "processed" / "esc50" / r.filename
        if not out.is_file():
            wav, _ = load_audio(ESC / "audio" / r.filename, TARGET_SR)
            save_audio(out, wav, TARGET_SR)
        rows.append({"clip_id": f"esc50__{r.filename[:-4]}", "dataset": "esc50",
                     "path": out.relative_to(REPO_ROOT).as_posix(), "group": f"esc:{r.src_file}",
                     "category": r.category, LABEL_COL: "cry" if r.category == "crying_baby" else "non_cry",
                     "duration_s": 5.0})
    return pd.DataFrame(rows)


def build_table() -> pd.DataFrame:
    dac = pd.read_csv(REPO_ROOT / "data" / "metadata_folds.csv")
    dac = pd.DataFrame({"clip_id": dac["clip_id"], "dataset": "donateacry", "path": dac["path"],
                        "group": dac["group"], "category": "cry", LABEL_COL: "cry", "duration_s": dac["duration_s"]})
    man = pd.read_csv(REPO_ROOT / "data" / "dapt" / "manifest.csv", dtype={"baby_id": str})
    first = man.sort_values("chunk_id").groupby("baby_id", as_index=False).first()
    cc = pd.DataFrame({"clip_id": first["chunk_id"], "dataset": "cryceleb", "path": first["path"],
                       "group": "cc:" + first["baby_id"], "category": "cry", LABEL_COL: "cry",
                       "duration_s": (first["n_samples"] / TARGET_SR).round(4)})
    df = pd.concat([dac, cc, esc50_table()], ignore_index=True)
    df["strat"] = df["dataset"] + ":" + df[LABEL_COL]
    return df.sort_values("clip_id").reset_index(drop=True)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true")
    args = ap.parse_args(argv)
    ev = load_project()["evaluation"]
    n_folds = ev["n_folds"]
    table = build_table()
    if ASSIGN.exists() and not args.regenerate:
        df = load_assignment(table, ASSIGN, "strat", n_folds)
        print(f"using committed fold assignment {ASSIGN.relative_to(REPO_ROOT).as_posix()}")
    else:
        df = grouped_folds(table, "strat", n_folds, ev["split_random_state"], ev["val_group_fraction"])
        save_assignment(df, ASSIGN, "strat", n_folds)
        print(f"computed folds -> {ASSIGN.relative_to(REPO_ROOT).as_posix()}")
    check_grouped(df, n_folds)
    df = df[list(table.columns) + split_cols(n_folds)]
    df.to_csv(OUT, index=False, lineterminator="\n")
    report = {"label_set": "detect_v1", "n_clips": int(len(df)), "n_groups": int(df["group"].nunique()),
              "hours": round(float(df["duration_s"].sum()) / 3600, 4),
              "by_part": {k: {"clips": int(len(g)), "groups": int(g["group"].nunique()),
                              "hours": round(float(g["duration_s"].sum()) / 3600, 4)}
                          for k, g in df.groupby("strat")},
              "folds": {f"fold_{k}": {role: df.loc[df[f"split_f{k}"] == role, "strat"].value_counts().to_dict()
                                      for role in ("train", "val", "test")} for k in range(n_folds)},
              "folds_file_sha256": sha256(OUT)}
    write_json(REPO_ROOT / "results" / "detection_splits_report.json", report)
    print({k: v["clips"] for k, v in report["by_part"].items()}, f"{report['hours']:.2f} h",
          "sha", report["folds_file_sha256"][:16])
    return report


if __name__ == "__main__":
    main()
