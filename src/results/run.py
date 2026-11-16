"""Run folders, env.json, status.json and the append-only registry (RESULTS_SPEC §1, §3, §8).

Typical use inside a trainer::

    with Run.create(cfg) as run:
        for fold in cfg["eval"]["folds"]:
            for seed in cfg["eval"]["seeds"]:
                run.log_epoch(fold, seed, epoch=1, train_loss=..., val_macro_f1=...)
                run.save_val(fold, seed, val_logits, val_meta)
                run.save_test(fold, seed, test_logits)          # once, after model is final
                run.save_predictions(fold, seed, preds_df)
                run.save_metrics(fold, seed, metrics_dict)
    # leaving the block -> summary.json + status "completed" + registry line
    # an exception inside the block -> status "failed" + registry line, then re-raised
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform as _platform
import re
import socket
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from src.config import REPO_ROOT, get_path, validate_config, yaml_load

DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
TRAIN_LOG_COLUMNS = ["epoch", "step", "train_loss", "val_loss", "val_macro_f1", "val_acc", "lr", "seconds"]
RUN_ID_RE = re.compile(r"^(?P<exp>E\d{2}[\w-]*)__(?P<name>[\w-]+)__(?P<ts>\d{8}-\d{6})__(?P<git>\w{7})$")


# --------------------------------------------------------------------------- helpers
def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str | Path | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(cwd: str | Path = REPO_ROOT) -> dict:
    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": commit,
        "git7": commit[:7] if commit else "nogit00",
        "git_dirty": bool(status) if status is not None else None,
    }


def detect_platform() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    if os.environ.get("RUNPOD_POD_ID"):
        return "runpod"
    return "local"


def _version(mod: str) -> str | None:
    try:
        m = __import__(mod)
        return getattr(m, "__version__", None)
    except Exception:
        return None


def collect_env(cfg: dict, git: dict | None = None) -> dict:
    git = git or git_info()
    cuda = gpu = None
    try:
        import torch  # noqa: WPS433

        cuda = torch.version.cuda
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return {
        "git_commit": git["git_commit"],
        "git_dirty": git["git_dirty"],
        "python": _platform.python_version(),
        "torch": _version("torch"),
        "transformers": _version("transformers"),
        "peft": _version("peft"),
        "numpy": _version("numpy"),
        "sklearn": _version("sklearn"),
        "cuda": cuda,
        "gpu": gpu,
        "platform": detect_platform(),
        "os": _platform.platform(),
        "hostname": socket.gethostname(),
        "started_utc": iso(utcnow()),
        "data_metadata_sha256": sha256_file(get_path(cfg, "data.base_metadata")),
        "folds_file_sha256": sha256_file(get_path(cfg, "data.metadata")),
    }


def make_run_id(experiment_id: str, name: str, when: dt.datetime, git7: str) -> str:
    safe = re.sub(r"[^\w-]+", "-", name).strip("-")
    return f"{experiment_id}__{safe}__{when.strftime('%Y%m%d-%H%M%S')}__{git7[:7]}"


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (dt.datetime, dt.date)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _clean_nans(o: Any) -> Any:
    """JSON has no NaN; store missing/undefined values as null."""
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    if isinstance(o, dict):
        return {k: _clean_nans(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_nans(v) for v in o]
    return o


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean_nans(obj), f, indent=2, default=_json_default, ensure_ascii=False)
        f.write("\n")


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- registry
def registry_path(results_dir: str | Path = DEFAULT_RESULTS_DIR) -> Path:
    return Path(results_dir) / "registry.jsonl"


def append_registry(record: dict, results_dir: str | Path = DEFAULT_RESULTS_DIR) -> None:
    """Append-only. Never edit old lines; corrections are new lines."""
    p = registry_path(results_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_clean_nans(record), default=_json_default, ensure_ascii=False)
    with open(p, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_registry(results_dir: str | Path = DEFAULT_RESULTS_DIR) -> list[dict]:
    p = registry_path(results_dir)
    if not p.is_file():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"registry.jsonl line {i} is not valid JSON: {e}") from e
    return out


def mark_superseded(run_id: str, superseded_by: str, notes: str = "",
                    results_dir: str | Path = DEFAULT_RESULTS_DIR) -> None:
    """Append a correction record saying ``run_id`` is replaced by ``superseded_by``."""
    append_registry({"run_id": run_id, "record_type": "correction", "superseded_by": superseded_by,
                     "notes": notes, "created_utc": iso(utcnow())}, results_dir)


def resolve_registry(records: list[dict]) -> dict[str, dict]:
    """Collapse the append-only log into the latest state per run_id
    (later lines override earlier ones, field by field)."""
    state: dict[str, dict] = {}
    for rec in records:
        rid = rec.get("run_id")
        if not rid:
            continue
        merged = dict(state.get(rid, {}))
        merged.update({k: v for k, v in rec.items() if k != "record_type"})
        state[rid] = merged
    return state


def select_runs_for_paper(records: list[dict]) -> dict[tuple[str, str], dict]:
    """Latest non-superseded *completed* run per (paper_table, paper_row)."""
    chosen: dict[tuple[str, str], dict] = {}
    for rec in resolve_registry(records).values():
        if rec.get("status") != "completed" or rec.get("superseded_by"):
            continue
        key = (rec.get("paper_table"), rec.get("paper_row"))
        if None in key:
            continue
        prev = chosen.get(key)
        if prev is None or (rec.get("created_utc") or "") >= (prev.get("created_utc") or ""):
            chosen[key] = rec
    return chosen


# --------------------------------------------------------------------------- Run
class Run:
    def __init__(self, run_dir: Path, config: dict, results_dir: Path, env: dict):
        self.dir = Path(run_dir)
        self.run_id = self.dir.name
        self.config = config
        self.results_dir = Path(results_dir)
        self.env = env
        self._finished = False

    # ---- lifecycle
    @classmethod
    def create(cls, config: dict, results_dir: str | Path = DEFAULT_RESULTS_DIR,
               git: dict | None = None, when: dt.datetime | None = None) -> "Run":
        validate_config(config)
        git = git or git_info()
        when = when or utcnow()
        run_id = make_run_id(config["experiment_id"], config["name"], when, git["git7"])
        run_dir = Path(results_dir) / "runs" / run_id
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(exist_ok=False)  # never overwrite a run folder
        env = collect_env(config, git)
        env["started_utc"] = iso(when)
        with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        write_json(run_dir / "env.json", env)
        run = cls(run_dir, config, Path(results_dir), env)
        run._write_status("running")
        return run

    @classmethod
    def open(cls, run_id: str, results_dir: str | Path = DEFAULT_RESULTS_DIR) -> "Run":
        run_dir = Path(results_dir) / "runs" / run_id
        with open(run_dir / "config.yaml", encoding="utf-8") as f:
            cfg = yaml_load(f)
        return cls(run_dir, cfg, Path(results_dir), read_json(run_dir / "env.json"))

    def _write_status(self, status: str, error: str | None = None) -> None:
        p = self.dir / "status.json"
        prev = read_json(p) if p.is_file() else {}
        write_json(p, {"status": status, "error": error,
                       "started": prev.get("started", self.env.get("started_utc")),
                       "ended": iso(utcnow()) if status != "running" else None})

    def status(self) -> dict:
        return read_json(self.dir / "status.json")

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._finished:
            return False
        if exc_type is None:
            self.complete()
        else:
            self.fail("".join(traceback.format_exception(exc_type, exc, tb))[-4000:])
        return False  # never swallow exceptions

    def complete(self, notes: str = "") -> dict:
        from src.results.aggregate import aggregate_run

        summary = aggregate_run(self.dir)
        self._write_status("completed")
        self._register("completed", summary, notes)
        self._finished = True
        return summary

    def fail(self, error: str, notes: str = "") -> None:
        self._write_status("failed", error)
        self._register("failed", None, notes or error.strip().splitlines()[-1][:300] if error else notes)
        self._finished = True

    def _register(self, status: str, summary: dict | None, notes: str) -> None:
        m = (summary or {}).get("metrics", {})

        def _ms(key: str, stat: str) -> float | None:
            return (m.get(key) or {}).get(stat)

        cfg = self.config
        append_registry({
            "run_id": self.run_id,
            "experiment_id": cfg.get("experiment_id"),
            "name": cfg.get("name"),
            "status": status,
            "parent_run": cfg.get("parent_run"),
            "change_vs_parent": cfg.get("change_vs_parent"),
            "paper_table": get_path(cfg, "paper.table"),
            "paper_row": get_path(cfg, "paper.row"),
            "task": cfg.get("task"),
            "folds_file_sha256": self.env.get("folds_file_sha256"),
            "git_commit": self.env.get("git_commit"),
            "macro_f1_mean": _ms("macro_f1", "mean"),
            "macro_f1_std": _ms("macro_f1", "std"),
            "accuracy_mean": _ms("accuracy", "mean"),
            "accuracy_std": _ms("accuracy", "std"),
            "ece_cal_mean": _ms("ece_15_cal", "mean"),
            "created_utc": iso(utcnow()),
            "superseded_by": None,
            "notes": notes or cfg.get("notes", ""),
        }, self.results_dir)

    # ---- per fold/seed outputs
    def fold_dir(self, fold: int, seed: int) -> Path:
        d = self.dir / f"fold_{int(fold)}_seed_{int(seed)}"
        d.mkdir(exist_ok=True)
        return d

    def log_epoch(self, fold: int, seed: int, **row: Any) -> None:
        unknown = set(row) - set(TRAIN_LOG_COLUMNS)
        if unknown:
            raise KeyError(f"unknown train_log columns: {sorted(unknown)}")
        p = self.fold_dir(fold, seed) / "train_log.csv"
        df = pd.DataFrame([{c: row.get(c) for c in TRAIN_LOG_COLUMNS}])
        df.to_csv(p, mode="a", header=not p.is_file(), index=False)

    def save_val(self, fold: int, seed: int, logits: np.ndarray, meta: pd.DataFrame) -> None:
        """Validation logits of the selected (best-on-val) model; may be rewritten during training."""
        need = {"clip_id", "group", "label_idx"}
        if not need <= set(meta.columns):
            raise KeyError(f"val_meta needs columns {sorted(need)}")
        if len(meta) != len(logits):
            raise ValueError("val logits and meta have different lengths")
        d = self.fold_dir(fold, seed)
        np.save(d / "val_logits.npy", np.asarray(logits, dtype=np.float32))
        meta[["clip_id", "group", "label_idx"]].to_csv(d / "val_meta.csv", index=False)

    def save_test(self, fold: int, seed: int, logits: np.ndarray) -> None:
        """Test logits are written ONCE, after the model is final (test folds are sacred)."""
        p = self.fold_dir(fold, seed) / "test_logits.npy"
        if p.exists():
            raise FileExistsError(f"{p} already exists: test predictions are produced once per run")
        np.save(p, np.asarray(logits, dtype=np.float32))

    def save_predictions(self, fold: int, seed: int, df: pd.DataFrame) -> None:
        df.to_csv(self.fold_dir(fold, seed) / "predictions.csv", index=False)

    def save_metrics(self, fold: int, seed: int, metrics: dict) -> None:
        write_json(self.fold_dir(fold, seed) / "metrics.json", metrics)

    def save_calibration(self, fold: int, seed: int, calib: dict) -> None:
        write_json(self.fold_dir(fold, seed) / "calibration.json", calib)


# --------------------------------------------------------------------------- predictions.csv
def build_predictions(
    *, clip_ids: Sequence, groups: Sequence, datasets: Sequence, fold: int, seed: int,
    y_true: Sequence[int], logits: np.ndarray, labels: Sequence[str],
    temperature: float = 1.0, thresholds: dict[str, float] | None = None,
    coverages: Sequence[float] = (0.9, 0.7, 0.5),
) -> pd.DataFrame:
    """One row per test clip, columns per RESULTS_SPEC §4."""
    from src.results.metrics import softmax

    logits = np.asarray(logits, dtype=np.float64)
    probs = softmax(logits, temperature)
    pred = logits.argmax(axis=1)  # uncalibrated argmax (temperature does not change it)
    conf = probs.max(axis=1)
    df = pd.DataFrame({
        "clip_id": list(clip_ids), "group": list(groups), "dataset": list(datasets),
        "fold": int(fold), "seed": int(seed),
        "y_true": [labels[i] for i in y_true],
        "y_pred": [labels[i] for i in pred],
    })
    for j, lab in enumerate(labels):
        df[f"p_{lab}"] = probs[:, j]
    df["confidence"] = conf
    for c in coverages:
        tau = (thresholds or {}).get(str(c))
        df[f"abstain@{c}cov"] = (conf < tau) if tau is not None else pd.NA
    return df
