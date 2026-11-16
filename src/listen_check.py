"""Export a random sample of clips per class to data/listen/ for a human listening audit.

    python -m src.listen_check [--per-class 5] [--seed 0]

Writes data/listen/<label>/<n>_<clip>.wav and data/listen/listen_sheet.csv with empty columns
``is_cry`` / ``label_plausible`` / ``notes`` for Idris to fill in. data/ is git-ignored.
"""
from __future__ import annotations

import argparse
import shutil

import pandas as pd

from src.config import REPO_ROOT, load_project


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    project = load_project()
    folds = pd.read_csv(REPO_ROOT / "data" / "metadata_folds.csv")
    col = f"label_{project['split_label']}"
    out = REPO_ROOT / "data" / "listen"
    if out.exists():
        shutil.rmtree(out)
    rows = []
    for label, g in folds.groupby(col):
        sample = g.sample(n=min(args.per_class, len(g)), random_state=args.seed)
        for i, r in enumerate(sample.itertuples(), 1):
            dst = out / label / f"{i:02d}_{r.clip_id}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO_ROOT / r.path, dst)
            rows.append({"file": dst.relative_to(out).as_posix(), "label": label, "group": r.group,
                         "is_cry": "", "label_plausible": "", "notes": ""})
    pd.DataFrame(rows).to_csv(out / "listen_sheet.csv", index=False)
    print(f"exported {len(rows)} clips to {out.relative_to(REPO_ROOT)}/ (fill in listen_sheet.csv)")


if __name__ == "__main__":
    main()
