"""Import run folders produced elsewhere (Kaggle/Colab) into this repo's results/.

    python -m src.import_runs path/to/crystal_results.zip     # or an unzipped folder

The archive must contain ``runs/<run_id>/...`` and ``registry_new.jsonl`` (written by the Kaggle
notebook). Run folders are copied only if they do not exist locally (never overwritten);
registry lines are appended only for run_ids not already in the local registry.
DAPT logs (``dapt/<run_id>/``, from src.dapt) are copied to results/dapt/ the same way (no registry).
Numbers are never re-typed: they arrive as the same files the trainer wrote.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from src.results.run import DEFAULT_RESULTS_DIR, append_registry, read_registry


def import_runs(src: str | Path, results_dir: str | Path = DEFAULT_RESULTS_DIR) -> list[str]:
    src = Path(src)
    results_dir = Path(results_dir)
    with tempfile.TemporaryDirectory() as tmp:
        root = src
        if src.is_file() and src.suffix == ".zip":
            with zipfile.ZipFile(src) as z:
                for name in z.namelist():  # refuse path traversal
                    if name.startswith("/") or ".." in Path(name).parts:
                        raise ValueError(f"unsafe path in archive: {name}")
                z.extractall(tmp)
            root = Path(tmp)
        cands = [p for p in root.rglob("registry_new.jsonl")]
        if len(cands) != 1:
            raise FileNotFoundError("archive must contain exactly one registry_new.jsonl")
        base = cands[0].parent
        known = {r.get("run_id") for r in read_registry(results_dir)}
        imported = []
        new_lines = [json.loads(line) for line in cands[0].read_text(encoding="utf-8").splitlines() if line.strip()]
        for rec in new_lines:
            rid = rec.get("run_id")
            src_dir = base / "runs" / rid
            dst_dir = results_dir / "runs" / rid
            if rid in known and rec.get("record_type") != "correction":
                print(f"skip (already registered): {rid}")
                continue
            if rec.get("record_type") != "correction":
                if not src_dir.is_dir():
                    raise FileNotFoundError(f"run folder missing in archive: {rid}")
                if dst_dir.exists():
                    raise FileExistsError(f"{dst_dir} exists locally but is not registered; inspect manually")
                shutil.copytree(src_dir, dst_dir)
                imported.append(rid)
            append_registry(rec, results_dir)
            known.add(rid)
        for d in sorted((base / "dapt").glob("*")) if (base / "dapt").is_dir() else []:
            dst = results_dir / "dapt" / d.name
            if dst.exists():
                print(f"skip (already present): dapt/{d.name}")
                continue
            shutil.copytree(d, dst)
            imported.append(f"dapt/{d.name}")
    for rid in imported:
        print(f"imported {rid}")
    return imported


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    a = ap.parse_args(argv)
    import_runs(a.source, a.results_dir)


if __name__ == "__main__":
    main()
