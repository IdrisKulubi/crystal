"""Generic trainer: config -> run folder per docs/RESULTS_SPEC.md.

    python -m src.train --config configs/exp/E01_mfcc_svm.yaml                 # all folds x seeds
    python -m src.train --config configs/exp/E02_deepinfant_repro.yaml --fold 0 --seeds 0
    python -m src.train --config ... --debug        # 5 clips/class, fold 0, seed 0, 1 epoch -> results/debug/
    python -m src.train --config ... --set train.lr=1e-4 --set train.epochs=30

For every fold x seed: fit on train, select on val, calibrate on val (temperature + coverage
thresholds), predict test ONCE, save logits / predictions / metrics. Failures are logged.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import numpy as np

from src.calibrate import calibrate
from src.config import REPO_ROOT, get_path, load_config, load_project
from src.datasets import FoldData, fold_data, load_table, load_waveforms
from src.results import metrics as M
from src.results.run import DEFAULT_RESULTS_DIR, Run, build_predictions


def _model_fn(model_type: str) -> Callable[[FoldData, dict, int], dict]:
    if model_type == "majority":
        from src.models.baselines import majority_fit_predict
        return majority_fit_predict
    if model_type == "mfcc_svm":
        from src.models.baselines import svm_fit_predict
        return svm_fit_predict
    if model_type == "deepinfant_repro":
        from src.models.deepinfant_repro import deepinfant_fit_predict
        return deepinfant_fit_predict
    if model_type == "linear_probe":
        from src.models.probe import probe_fit_predict
        return probe_fit_predict
    if model_type == "finetune":
        from src.models.finetune import finetune_fit_predict
        return finetune_fit_predict
    raise ValueError(f"unknown model.type {model_type!r}")


def inference_ms_cpu(predict_one: Callable, wavs: list[np.ndarray], n: int = 50, warmup: int = 5) -> float | None:
    """Median wall-clock ms per clip, batch 1, CPU, after warm-up calls (RESULTS_SPEC §5)."""
    if not wavs:
        return None
    for i in range(warmup):
        predict_one(wavs[i % len(wavs)])
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        predict_one(wavs[i % len(wavs)])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def run_fold_seed(run: Run, fd: FoldData, fold: int, seed: int, fit: Callable, time_inference: bool,
                  coverages) -> dict:
    cfg = run.config
    t0 = time.time()
    out = fit(fd, cfg, seed)
    train_seconds = time.time() - t0
    for row in out.get("train_log", []):
        run.log_epoch(fold, seed, **row)
    yva = fd.val["label_idx"].to_numpy()
    yte = fd.test["label_idx"].to_numpy()
    run.save_val(fold, seed, out["val_logits"], fd.val[["clip_id", "group", "label_idx"]])
    run.save_test(fold, seed, out["test_logits"])

    cal = calibrate(out["val_logits"], yva, coverages)
    run.save_calibration(fold, seed, cal)
    T, thr = cal["temperature"], cal["thresholds"]
    run.save_predictions(fold, seed, build_predictions(
        clip_ids=fd.test["clip_id"], groups=fd.test["group"], datasets=fd.test["dataset"], fold=fold, seed=seed,
        y_true=yte, logits=out["test_logits"], labels=fd.labels, temperature=T, thresholds=thr,
        coverages=coverages))
    m = M.classification_metrics(yte, M.softmax(out["test_logits"], T), fd.labels,
                                 probs_uncal=M.softmax(out["test_logits"]), thresholds=thr, coverages=coverages)
    m.update({
        "train_seconds": round(train_seconds, 2),
        "params_total": int(out.get("params_total", 0)),
        "params_trainable": int(out.get("params_trainable", 0)),
        "inference_ms_per_clip_cpu": (inference_ms_cpu(out["predict_one"], load_waveforms(fd.test["path"][:50]))
                                      if time_inference and out.get("predict_one") else None),
        "model_selection": out.get("chosen", {}),
    })
    run.save_metrics(fold, seed, m)
    return m


def resolve_parent(parent: str | None, results_dir: Path = DEFAULT_RESULTS_DIR) -> str | None:
    """``parent_run: E07`` (an experiment id) -> run_id of that experiment's latest completed,
    non-superseded run in the registry at launch time. Full run_ids are kept as they are."""
    import re

    from src.results.run import read_registry, resolve_registry

    if not parent or not re.fullmatch(r"E\d{2}[\w-]*", parent) or "__" in parent:
        return parent
    runs = [r for r in resolve_registry(read_registry(results_dir)).values()
            if r.get("experiment_id") == parent and r.get("status") == "completed" and not r.get("superseded_by")]
    if not runs:
        print(f"WARNING: parent experiment {parent} has no completed run yet; recording {parent!r}")
        return parent
    return max(runs, key=lambda r: r.get("created_utc", ""))["run_id"]


def main(argv: list[str] | None = None) -> str:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", default=None, help="'all' or comma list (default: config eval.folds)")
    ap.add_argument("--seeds", default=None, help="comma list (default: config eval.seeds)")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. train.lr=1e-4")
    ap.add_argument("--debug", action="store_true", help="tiny CPU smoke test -> results/debug/")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--notes", default="")
    args = ap.parse_args(argv)

    overrides = list(args.set)
    if args.debug:
        overrides += ["eval.folds=[0]", "eval.seeds=[0]", "train.epochs=1"]
    else:
        if args.fold and args.fold != "all":
            overrides.append(f"eval.folds=[{args.fold}]")
        if args.seeds:
            overrides.append(f"eval.seeds=[{args.seeds}]")
    cfg = load_config(args.config, overrides)
    if args.notes:
        cfg["notes"] = args.notes
    if args.debug:
        cfg["notes"] = (cfg.get("notes") or "") + " [debug]"
    results_dir = Path(args.results_dir) if args.results_dir else (
        REPO_ROOT / "results" / "debug" if args.debug else DEFAULT_RESULTS_DIR)
    cfg["parent_run"] = resolve_parent(cfg.get("parent_run"))

    df, labels = load_table(cfg)
    fit = _model_fn(get_path(cfg, "model.type"))
    coverages = load_project()["evaluation"]["coverages"]
    folds, seeds = cfg["eval"]["folds"], cfg["eval"]["seeds"]
    with Run.create(cfg, results_dir=results_dir) as run:
        print(f"run {run.run_id}: {len(folds)} folds x {len(seeds)} seeds, {len(df)} clips, labels={labels}")
        first = True
        for fold in folds:
            for seed in seeds:
                fd = fold_data(df, labels, fold, debug=args.debug, seed=seed)
                m = run_fold_seed(run, fd, fold, seed, fit, time_inference=first, coverages=coverages)
                first = False
                print(f"  fold {fold} seed {seed}: macro-F1 {m['macro_f1']:.3f}  acc {m['accuracy']:.3f}  "
                      f"bal-acc {m['balanced_accuracy']:.3f}  ECE {m['ece_15_cal']:.3f}  ({m['train_seconds']:.0f}s)")
    s = run.status()
    summ = (run.dir / "summary.json")
    if summ.is_file():
        from src.results.run import read_json
        mm = read_json(summ)["metrics"]
        print(f"done: macro-F1 {mm['macro_f1']['mean']:.3f} +/- {mm['macro_f1']['std']:.3f}, "
              f"acc {mm['accuracy']['mean']:.3f} +/- {mm['accuracy']['std']:.3f}  [{s['status']}]")
    return run.run_id


if __name__ == "__main__":
    main()
