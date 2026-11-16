import json

from src.config import load_project
from src.report import build, fmt, fmt_delta, fmt_p, slugify, write_outputs
from src.results.run import mark_superseded
from tests.conftest import fill_fake_run


def _report(results_dir):
    res = build(results_dir, load_project())
    write_outputs(res, results_dir)
    return res, json.loads((results_dir / "paper_numbers.json").read_text(encoding="utf-8")), \
        (results_dir / "RESULTS.md").read_text(encoding="utf-8")


def test_formatting_rules():
    assert fmt("pct", 0.8734, 0.0212) == "87.3 ± 2.1"
    assert fmt("f3", 0.71234, 0.0341) == "0.712 ± 0.034"
    assert fmt("pct", None) is None
    assert fmt_p(0.0004) == "< 0.001" and fmt_p(0.0321) == "0.032"
    assert fmt_delta(-0.0412) == "-0.041" and fmt_delta(0.05) == "+0.050"
    assert slugify("− domain-adaptive pretraining") == "minus_domain_adaptive_pretraining"
    assert slugify("MFCC + SVM") == "mfcc_plus_svm"


def test_empty_registry_lists_all_placeholders(results_dir):
    res, pn, md = _report(results_dir)
    assert pn["model_name"] == "CRYSTAL"
    assert pn["abstract.reason_macro_f1_ours"] is None
    assert pn["T3.ours_crystal.macro_f1"] is None
    # the literature constant is filled even with no runs
    assert pn["T3.deepinfant_reported.accuracy"] == "89.0 (reported)"
    assert "No runs registered yet" in md
    assert "- [ ] `abstract.reason_macro_f1_ours`" in md
    assert "- [ ] `T5.cov90.precision`" in md
    for t in ("T3", "T4", "T5", "T6"):
        assert (results_dir / "tables" / f"{t}.md").is_file()
        assert (results_dir / "tables" / f"{t}.csv").is_file()


def test_completed_run_fills_its_row_and_abstract(results_dir, new_run):
    with new_run(paper={"table": "T3", "row": "Ours (CRYSTAL)"}, experiment_id="E36", name="final") as ours:
        fill_fake_run(ours, acc_by_fold={0: 0.9, 1: 0.7})
    with new_run(paper={"table": "T3", "row": "DeepInfant (reproduced)"}, experiment_id="E02", name="di") as di:
        fill_fake_run(di, rng_seed=1)
    res, pn, md = _report(results_dir)
    summ = json.loads((ours.dir / "summary.json").read_text(encoding="utf-8"))
    m = summ["metrics"]["macro_f1"]
    assert pn["T3.ours_crystal.macro_f1"] == fmt("f3", m["mean"], m["std"])
    assert pn["abstract.reason_macro_f1_ours"] == pn["T3.ours_crystal.macro_f1"]
    assert pn["abstract.reason_macro_f1_deepinfant"] == pn["T3.deepinfant_reproduced.macro_f1"]
    assert pn["_sources"]["T3.ours_crystal.macro_f1"] == ours.run_id
    assert pn["T4.ours_crystal.delta"] == "+0.000"
    assert pn["T5.cov70.threshold"] == "0.500 ± 0.000"
    assert ours.run_id in md


def test_superseded_run_is_ignored(results_dir, new_run):
    with new_run(paper={"table": "T3", "row": "MFCC + SVM"}, experiment_id="E01", name="svm") as a:
        fill_fake_run(a, acc_by_fold={0: 0.9, 1: 0.9})
    with new_run(paper={"table": "T3", "row": "MFCC + SVM"}, experiment_id="E01", name="svm_v2") as b:
        fill_fake_run(b, acc_by_fold={0: 0.1, 1: 0.1}, rng_seed=3)
    mark_superseded(b.run_id, a.run_id, "v2 used wrong labels", results_dir)
    res, pn, md = _report(results_dir)
    assert pn["_sources"]["T3.mfcc_plus_svm.macro_f1"] == a.run_id
    assert "superseded runs (ignored)" in md


def test_mismatched_folds_files_warn(results_dir, new_run, tmp_path):
    f1, f2 = tmp_path / "f1.csv", tmp_path / "f2.csv"
    f1.write_text("a\n1\n")
    f2.write_text("a\n2\n")
    with new_run(data={"metadata": str(f1)}, paper={"table": "T3", "row": "MFCC + SVM"}) as a:
        fill_fake_run(a)
    with new_run(data={"metadata": str(f2)}, paper={"table": "T3", "row": "AST, fine-tuned"}) as b:
        fill_fake_run(b)
    res, _, md = _report(results_dir)
    assert any("DIFFERENT folds files" in w for w in res["warnings"])


def test_failed_run_listed_but_not_used(results_dir, new_run):
    try:
        with new_run(paper={"table": "T3", "row": "AST, LoRA"}):
            raise RuntimeError("kaggle session timed out")
    except RuntimeError:
        pass
    _, pn, md = _report(results_dir)
    assert pn["T3.ast_lora.macro_f1"] is None
    assert "kaggle session timed out" in md
