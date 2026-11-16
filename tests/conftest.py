import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import deep_merge, load_yaml, CONFIG_DIR  # noqa: E402
from src.results import metrics as M  # noqa: E402
from src.results.run import Run, build_predictions  # noqa: E402

LABELS = ["hungry", "pain_discomfort", "tired", "burping"]
FAKE_GIT = {"git_commit": "abcdef1234567890", "git7": "abcdef1", "git_dirty": False}


def make_cfg(**over):
    base = load_yaml(CONFIG_DIR / "defaults.yaml")
    cfg = deep_merge(base, {
        "experiment_id": "E10", "name": "ast_ft",
        "paper": {"table": "T3", "row": "AST, fine-tuned"},
        "data": {"metadata": "does/not/exist.csv"},
        "eval": {"folds": [0, 1], "seeds": [0, 1]},
    })
    return deep_merge(cfg, over)


def fill_fake_run(run: Run, acc_by_fold=None, rng_seed=0):
    """Write plausible outputs for every fold x seed of ``run``."""
    rng = np.random.default_rng(rng_seed)
    for fold in run.config["eval"]["folds"]:
        for seed in run.config["eval"]["seeds"]:
            n = 20
            y = rng.integers(0, len(LABELS), n)
            logits = rng.normal(size=(n, len(LABELS)))
            k = int(n * (acc_by_fold or {}).get(fold, 0.5))
            logits[np.arange(k), y[:k]] += 5.0  # first k correct
            run.log_epoch(fold, seed, epoch=1, train_loss=1.0, val_macro_f1=0.5)
            meta = pd.DataFrame({"clip_id": [f"v{i}" for i in range(n)], "group": [f"g{i % 5}" for i in range(n)],
                                 "label_idx": y})
            run.save_val(fold, seed, logits, meta)
            run.save_test(fold, seed, logits)
            run.save_predictions(fold, seed, build_predictions(
                clip_ids=[f"f{fold}c{i}" for i in range(n)], groups=[f"g{i % 5}" for i in range(n)],
                datasets=["donateacry"] * n, fold=fold, seed=seed, y_true=y, logits=logits, labels=LABELS))
            run.save_metrics(fold, seed, M.classification_metrics(y, M.softmax(logits), LABELS))
            run.save_calibration(fold, seed, {"temperature": 1.5, "fit_on": "val",
                                              "thresholds": {"0.9": 0.3, "0.7": 0.5, "0.5": 0.7}})


@pytest.fixture
def results_dir(tmp_path):
    return tmp_path / "results"


@pytest.fixture
def new_run(results_dir):
    counter = {"i": 0}

    def _make(**over):
        counter["i"] += 1
        when = dt.datetime(2026, 11, 20, 14, 0, counter["i"], tzinfo=dt.timezone.utc)
        return Run.create(make_cfg(**over), results_dir=results_dir, git=FAKE_GIT, when=when)

    return _make
