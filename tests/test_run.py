import json

import numpy as np
import pytest

from src.config import parse_overrides, validate_config
from src.results.aggregate import aggregate_run, flatten, mean_std
from src.results.run import (
    RUN_ID_RE, Run, append_registry, mark_superseded, read_registry, select_runs_for_paper,
)
from tests.conftest import FAKE_GIT, fill_fake_run, make_cfg


def test_run_id_format(new_run):
    run = new_run()
    assert run.run_id == "E10__ast_ft__20261120-140001__abcdef1"
    assert RUN_ID_RE.match(run.run_id)


def test_create_writes_config_env_status(new_run):
    run = new_run()
    for f in ("config.yaml", "env.json", "status.json"):
        assert (run.dir / f).is_file()
    env = json.loads((run.dir / "env.json").read_text())
    assert env["git_commit"] == FAKE_GIT["git_commit"]
    assert env["folds_file_sha256"] is None  # file does not exist in tests
    assert run.status()["status"] == "running"


def test_never_overwrite_run_folder(results_dir, new_run):
    run = new_run()
    with pytest.raises(FileExistsError):
        Run.create(run.config, results_dir=results_dir, git=FAKE_GIT,
                   when=__import__("datetime").datetime(2026, 11, 20, 14, 0, 1))


def test_test_logits_written_once(new_run):
    run = new_run()
    run.save_test(0, 0, np.zeros((3, 4)))
    with pytest.raises(FileExistsError):
        run.save_test(0, 0, np.ones((3, 4)))


def test_successful_run_completes_and_registers(results_dir, new_run):
    with new_run() as run:
        fill_fake_run(run, acc_by_fold={0: 0.8, 1: 0.6})
    assert run.status()["status"] == "completed"
    summary = json.loads((run.dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_found"] == summary["n_expected"] == 4 and summary["missing"] == []
    assert summary["metrics"]["accuracy"]["n"] == 4
    assert summary["calibration"]["temperature"]["mean"] == pytest.approx(1.5)
    assert summary["pooled"]["n_test"] == 40          # seed 0, 2 folds x 20 clips
    assert set(summary["per_class_recall_mean_std"]) == {"hungry", "pain_discomfort", "tired", "burping"}
    reg = read_registry(results_dir)
    assert len(reg) == 1 and reg[0]["status"] == "completed"
    assert reg[0]["macro_f1_mean"] == pytest.approx(summary["metrics"]["macro_f1"]["mean"])


def test_failed_run_is_logged(results_dir, new_run):
    with pytest.raises(RuntimeError):
        with new_run() as run:
            raise RuntimeError("CUDA out of memory")
    st = run.status()
    assert st["status"] == "failed" and "CUDA out of memory" in st["error"]
    reg = read_registry(results_dir)
    assert reg[-1]["status"] == "failed" and "CUDA out of memory" in reg[-1]["notes"]


def test_missing_folds_are_reported(new_run):
    with new_run() as run:
        fill_fake_run(run)
        (run.dir / "fold_1_seed_1" / "metrics.json").unlink()
    s = json.loads((run.dir / "summary.json").read_text())
    assert s["missing"] == ["fold_1_seed_1"] and s["n_found"] == 3


def test_registry_is_append_only_and_supersede(results_dir):
    a = {"run_id": "A", "status": "completed", "paper_table": "T3", "paper_row": "X", "created_utc": "2026-11-20T10:00:00Z"}
    b = {"run_id": "B", "status": "completed", "paper_table": "T3", "paper_row": "X", "created_utc": "2026-11-19T10:00:00Z"}
    append_registry(a, results_dir)
    append_registry(b, results_dir)
    assert select_runs_for_paper(read_registry(results_dir))[("T3", "X")]["run_id"] == "A"  # latest
    mark_superseded("A", "B", "A used wrong label set", results_dir)
    lines = (results_dir / "registry.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3 and json.loads(lines[0]) == a  # old line untouched
    assert select_runs_for_paper(read_registry(results_dir))[("T3", "X")]["run_id"] == "B"


def test_failed_runs_never_selected(results_dir):
    append_registry({"run_id": "F", "status": "failed", "paper_table": "T3", "paper_row": "X"}, results_dir)
    assert select_runs_for_paper(read_registry(results_dir)) == {}


def test_flatten_and_mean_std():
    flat = flatten({"a": 1, "b": {"c": 2.0, "d": None, "e": [1, 2]}, "f": float("nan"), "g": True})
    assert flat == {"a": 1.0, "b.c": 2.0}
    ms = mean_std([0.5, 0.7])
    assert ms["mean"] == pytest.approx(0.6) and ms["std"] == pytest.approx(np.std([0.5, 0.7], ddof=1))
    assert mean_std([0.3])["std"] == 0.0


def test_config_validation_and_overrides():
    validate_config(make_cfg())
    with pytest.raises(ValueError):
        validate_config(make_cfg(task="bogus"))
    with pytest.raises(ValueError):
        validate_config(make_cfg(parent_run="E05__x", change_vs_parent=None))
    assert parse_overrides(["train.lr=1e-4", "eval.folds=[0]"]) == {"train": {"lr": 1e-4}, "eval": {"folds": [0]}}


def test_aggregate_rebuild_is_idempotent(new_run):
    with new_run() as run:
        fill_fake_run(run)
    first = (run.dir / "summary.json").read_text()
    aggregate_run(run.dir)
    assert (run.dir / "summary.json").read_text() == first


def test_import_runs_from_zip(tmp_path, new_run):
    import zipfile

    from src.import_runs import import_runs

    with new_run() as run:  # a "remote" run written into the fixture results dir
        fill_fake_run(run)
    remote_reg = (run.results_dir / "registry.jsonl").read_text().splitlines()
    zpath = tmp_path / "crystal_results.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("registry_new.jsonl", "\n".join(remote_reg) + "\n")
        for f in run.dir.rglob("*"):
            if f.is_file():
                z.write(f, f"runs/{run.run_id}/{f.relative_to(run.dir).as_posix()}")
    local = tmp_path / "local_results"
    assert import_runs(zpath, local) == [run.run_id]
    assert (local / "runs" / run.run_id / "summary.json").is_file()
    assert read_registry(local)[0]["run_id"] == run.run_id
    assert import_runs(zpath, local) == []          # idempotent: nothing imported twice
    assert len(read_registry(local)) == 1
