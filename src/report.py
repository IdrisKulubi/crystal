"""registry + run folders -> results/paper_numbers.json, results/RESULTS.md, results/tables/.

    python -m src.report

This is the ONLY way numbers reach the paper (CLAUDE.md rule 3). Every key that has no
source run yet is kept with value null and listed as unfilled in RESULTS.md.
RESULTS.md is regenerated on every call; never edit it by hand.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
from pathlib import Path
from typing import Any

from src.config import REPO_ROOT, load_project
from src.results.run import (
    DEFAULT_RESULTS_DIR, iso, read_json, read_registry, resolve_registry, select_runs_for_paper,
    utcnow, write_json,
)

UNFILLED = "[ ]"

# T3 columns: (key suffix, summary metric, kind)
T3_COLS = [
    ("accuracy", "accuracy", "pct"),
    ("macro_f1", "macro_f1", "f3"),
    ("balanced_acc", "balanced_accuracy", "pct"),
    ("ece", "ece_15_cal", "f3"),
]
T6_COLS = [
    ("accuracy", "accuracy", "pct"),
    ("f1", "f1", "f3"),
    ("precision", "precision", "pct"),
    ("recall", "recall", "pct"),
    ("roc_auc", "roc_auc", "f3"),
    ("false_alarms_per_hour", "false_alarms_per_hour", "f1"),
]
# T5b (every T3 model): calibration + selective prediction, mean ± std over folds × seeds
T5B_COLS = [
    ("ece_before", "ece_15_uncal", "f3"),
    ("ece_after", "ece_15_cal", "f3"),
    ("aurc", "aurc", "f3"),
    ("prec90", "precision_at_coverage.0.9", "pct"),
    ("prec70", "precision_at_coverage.0.7", "pct"),
    ("prec50", "precision_at_coverage.0.5", "pct"),
    ("mf1_90", "macro_f1_at_coverage.0.9", "f3"),
    ("mf1_70", "macro_f1_at_coverage.0.7", "f3"),
    ("mf1_50", "macro_f1_at_coverage.0.5", "f3"),
]
COVERAGES = [("cov100", "1.0"), ("cov90", "0.9"), ("cov70", "0.7"), ("cov50", "0.5")]
ABSTRACT_KEYS = [
    "abstract.detection_acc", "abstract.reason_macro_f1_ours", "abstract.reason_macro_f1_deepinfant",
    "abstract.selective_precision", "abstract.selective_coverage", "abstract.ece_ours",
]
OTHER_KEYS = [
    "datasets.names_for_abstract", "dapt.hours_unlabelled",
    "sig.ours_vs_deepinfant.delta", "sig.ours_vs_deepinfant.ci95", "sig.ours_vs_deepinfant.p",
    "leak.random_split_macro_f1", "leak.grouped_split_macro_f1",
]
FIGURES = [
    ("fig1_pipeline", "Figure 1: pipeline diagram"),
    ("fig2_risk_coverage", "Figure 2: risk–coverage curves"),
    ("fig3_reliability_before_after", "Figure 3: reliability before/after temperature scaling"),
    ("fig4_confusion_ours", "Figure 4: normalised confusion matrix (ours, pooled OOF)"),
    ("fig5_class_distribution", "Figure 5: class distribution"),
    ("fig6_dapt_loss", "Figure 6: DAPT loss curve"),
    ("figA_per_class_recall", "Figure A: per-class recall, ours vs DeepInfant reproduced"),
]


# --------------------------------------------------------------------------- formatting (RESULTS_SPEC §11)
def slugify(text: str) -> str:
    text = text.replace("−", "minus ").replace("+", " plus ")
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def fmt(kind: str, mean: float | None, std: float | None = None) -> str | None:
    if mean is None:
        return None
    if kind == "pct":
        s = f"{mean * 100:.1f}"
        return s if std is None else f"{s} ± {std * 100:.1f}"
    digits = {"f3": 3, "f1": 1}[kind]
    s = f"{mean:.{digits}f}"
    return s if std is None else f"{s} ± {std:.{digits}f}"


def fmt_delta(d: float | None) -> str | None:
    return None if d is None else f"{d:+.3f}"


def fmt_p(p: float | None) -> str | None:
    if p is None:
        return None
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


# --------------------------------------------------------------------------- builder
class Numbers:
    """Flat key -> formatted string, plus _raw and _sources (RESULTS_SPEC §11)."""

    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.raw: dict[str, Any] = {}
        self.sources: dict[str, str] = {}

    def declare(self, key: str) -> None:
        self.values.setdefault(key, None)

    def set(self, key: str, value: Any, raw: Any = None, source: str | None = None) -> None:
        self.values[key] = value
        if raw is not None:
            self.raw[key] = raw
        if source:
            self.sources[key] = source

    def unfilled(self) -> list[str]:
        return sorted(k for k, v in self.values.items() if v is None)

    def to_json(self) -> dict:
        out = dict(sorted(self.values.items()))
        out["_raw"] = dict(sorted(self.raw.items()))
        out["_sources"] = dict(sorted(self.sources.items()))
        return out


def _load_summary(results_dir: Path, run_id: str) -> dict | None:
    p = results_dir / "runs" / run_id / "summary.json"
    return read_json(p) if p.is_file() else None


def _metric(summary: dict | None, name: str) -> tuple[float | None, float | None]:
    m = (summary or {}).get("metrics", {}).get(name)
    return (m["mean"], m["std"]) if m else (None, None)


def build(results_dir: Path = DEFAULT_RESULTS_DIR, project: dict | None = None) -> dict:
    results_dir = Path(results_dir)
    project = project or load_project()
    records = read_registry(results_dir)
    chosen = select_runs_for_paper(records)
    summaries = {key: _load_summary(results_dir, rec["run_id"]) for key, rec in chosen.items()}
    warnings: list[str] = []
    nums = Numbers()

    nums.set("model_name", project["model_name"], source="configs/project.yaml")
    for k in ABSTRACT_KEYS + OTHER_KEYS:
        nums.declare(k)

    rows = project.get("paper_rows", {})

    # macro_f1_at_coverage is recomputed from saved predictions.csv for every reason run (one code path
    # for runs made before and after the metric was added; RESULTS_SPEC §5) and merged into its summary.
    from src.selective import macro_f1_at_coverage

    for key, rec in chosen.items():
        summ = summaries.get(key)
        if summ and rec.get("task", "reason") == "reason" and key[0] in ("T3", "T4"):
            for cov, ms in macro_f1_at_coverage(results_dir / "runs" / rec["run_id"]).items():
                summ.setdefault("metrics", {})[f"macro_f1_at_coverage.{cov}"] = ms
    constants = {(c["table"], c["row"]): c for c in project.get("constant_rows", [])}

    # ---- T3 main results
    t3_table = []
    for row in rows.get("T3", []):
        slug = slugify(row)
        rec, summ, const = chosen.get(("T3", row)), summaries.get(("T3", row)), constants.get(("T3", row))
        cells = {}
        for suffix, metric, kind in T3_COLS:
            key = f"T3.{slug}.{suffix}"
            nums.declare(key)
            if const and metric in const["values"]:
                v = const["values"][metric]
                nums.set(key, fmt(kind, v) + " (reported)", raw=v, source=f"constant:{const['experiment_id']}")
            elif const:
                nums.set(key, "n/a", source=f"constant:{const['experiment_id']}")
            elif summ:
                mean, std = _metric(summ, metric)
                if mean is not None:
                    nums.set(key, fmt(kind, mean, std), raw={"mean": mean, "std": std}, source=rec["run_id"])
            cells[suffix] = nums.values[key]
        t3_table.append({"row": row, **cells, "run_id": (rec or {}).get("run_id")})
        if summ and summ.get("missing"):
            warnings.append(f"T3 '{row}' ({rec['run_id']}): missing {len(summ['missing'])} fold×seed results")

    # ---- T4 ablations (delta vs full model)
    roles = project.get("roles", {})
    ours_key = (roles.get("ours", {}).get("table"), roles.get("ours", {}).get("row"))
    ours_f1, _ = _metric(summaries.get(ours_key), "macro_f1")
    t4_table = []
    for row in rows.get("T4", []):
        slug = slugify(row)
        key = ("T4", row)
        rec, summ = chosen.get(key), summaries.get(key)
        if summ is None and row == ours_key[1]:  # full model row comes from T3
            rec, summ = chosen.get(ours_key), summaries.get(ours_key)
        f1_key, d_key = f"T4.{slug}.macro_f1", f"T4.{slug}.delta"
        nums.declare(f1_key)
        nums.declare(d_key)
        mean, std = _metric(summ, "macro_f1")
        if mean is not None:
            nums.set(f1_key, fmt("f3", mean, std), raw={"mean": mean, "std": std}, source=rec["run_id"])
            if ours_f1 is not None:
                nums.set(d_key, fmt_delta(mean - ours_f1), raw=mean - ours_f1, source=rec["run_id"])
        # leakage rows: also the change vs the SAME model under the grouped protocol (Phase 8 discussion)
        base_role = {"leak_random_ours": "ours", "leak_random_deepinfant": "deepinfant_repro"}
        for leak_role, grouped_role in base_role.items():
            if roles.get(leak_role, {}).get("row") == row and mean is not None:
                g = roles.get(grouped_role, {})
                g_mean, _ = _metric(summaries.get((g.get("table"), g.get("row"))), "macro_f1")
                dg_key = f"T4.{slug}.delta_vs_grouped"
                nums.declare(dg_key)
                if g_mean is not None:
                    nums.set(dg_key, fmt_delta(mean - g_mean), raw=mean - g_mean, source=rec["run_id"])
        t4_table.append({"row": row, "macro_f1": nums.values[f1_key], "delta": nums.values[d_key],
                         "run_id": (rec or {}).get("run_id")})

    # ---- T5 selective prediction (ours)
    ours_summ = summaries.get(ours_key)
    ours_rec = chosen.get(ours_key)
    t5_table = []
    for name, cov in COVERAGES:
        pk, tk, fk = f"T5.{name}.precision", f"T5.{name}.threshold", f"T5.{name}.macro_f1"
        nums.declare(pk)
        nums.declare(tk)
        nums.declare(fk)
        mean, std = _metric(ours_summ, f"precision_at_coverage.{cov}")
        if mean is not None:
            nums.set(pk, fmt("pct", mean, std), raw={"mean": mean, "std": std}, source=ours_rec["run_id"])
        fmean, fstd = _metric(ours_summ, f"macro_f1_at_coverage.{cov}")
        if fmean is not None:
            nums.set(fk, fmt("f3", fmean, fstd), raw={"mean": fmean, "std": fstd}, source=ours_rec["run_id"])
        thr = (ours_summ or {}).get("calibration", {}).get(f"thresholds.{cov}")
        if thr:
            nums.set(tk, fmt("f3", thr["mean"], thr["std"]), raw=thr["mean"], source=ours_rec["run_id"])
        elif cov == "1.0" and mean is not None:
            nums.set(tk, "—", source=ours_rec["run_id"])
        t5_table.append({"coverage": cov, "precision": nums.values[pk], "macro_f1": nums.values[fk],
                         "threshold": nums.values[tk]})

    # ---- T5b calibration + selective prediction for every T3 model (HANDOFF Phase 6)
    t5b_table = []
    for row in rows.get("T3", []):
        if ("T3", row) in constants:
            continue
        slug = slugify(row)
        rec, summ = chosen.get(("T3", row)), summaries.get(("T3", row))
        cells = {}
        for suffix, metric, kind in T5B_COLS:
            key = f"T5b.{slug}.{suffix}"
            nums.declare(key)
            mean, std = _metric(summ, metric)
            if mean is not None:
                nums.set(key, fmt(kind, mean, std), raw={"mean": mean, "std": std}, source=rec["run_id"])
            cells[suffix] = nums.values[key]
        t5b_table.append({"row": row, **cells, "run_id": (rec or {}).get("run_id")})

    # ---- T6 detection
    t6_table = []
    for row in rows.get("T6", []):
        slug = slugify(row)
        rec, summ = chosen.get(("T6", row)), summaries.get(("T6", row))
        cells = {}
        for suffix, metric, kind in T6_COLS:
            key = f"T6.{slug}.{suffix}"
            nums.declare(key)
            mean, std = _metric(summ, metric)
            if mean is not None:
                nums.set(key, fmt(kind, mean, std), raw={"mean": mean, "std": std}, source=rec["run_id"])
            cells[suffix] = nums.values[key]
        t6_table.append({"row": row, **cells, "run_id": (rec or {}).get("run_id")})

    # ---- abstract
    def _copy(dst: str, src: str) -> None:
        if nums.values.get(src) is not None:
            nums.set(dst, nums.values[src], raw=nums.raw.get(src), source=nums.sources.get(src))

    ours_slug = slugify(ours_key[1] or "")
    di = roles.get("deepinfant_repro", {})
    det = roles.get("detector", {})
    _copy("abstract.reason_macro_f1_ours", f"T3.{ours_slug}.macro_f1")
    _copy("abstract.reason_macro_f1_deepinfant", f"T3.{slugify(di.get('row', ''))}.macro_f1")
    _copy("abstract.ece_ours", f"T3.{ours_slug}.ece")
    _copy("abstract.detection_acc", f"T6.{slugify(det.get('row', ''))}.accuracy")
    _copy("abstract.selective_precision", "T5.cov90.precision")
    if nums.values.get("abstract.selective_precision") is not None:
        nums.set("abstract.selective_coverage", "90", source=nums.sources["T5.cov90.precision"])
    leak = roles.get("leak_random_ours", {})
    _copy("leak.random_split_macro_f1", f"T4.{slugify(leak.get('row', ''))}.macro_f1")
    _copy("leak.grouped_split_macro_f1", f"T3.{ours_slug}.macro_f1")

    # ---- significance (computed here if missing: deterministic, rng seed 0)
    di_rec = chosen.get((di.get("table"), di.get("row")))
    sig_dir = results_dir / "significance"
    if ours_rec and di_rec:
        p = sig_dir / f"{ours_rec['run_id']}__vs__{di_rec['run_id']}.json"
        if not p.is_file():
            from src.significance import compare

            try:
                compare(ours_rec["run_id"], di_rec["run_id"], results_dir)
            except (ValueError, FileNotFoundError) as e:
                warnings.append(f"significance ours vs DeepInfant not computed: {e}")
        if p.is_file():
            s = read_json(p)
            src = p.name
            nums.set("sig.ours_vs_deepinfant.delta", fmt_delta(s["delta_macro_f1"]), raw=s["delta_macro_f1"], source=src)
            nums.set("sig.ours_vs_deepinfant.ci95", f"[{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]", raw=s["ci95"], source=src)
            nums.set("sig.ours_vs_deepinfant.p", fmt_p(s["p_value"]), raw=s["p_value"], source=src)
    sig_files = sorted(sig_dir.glob("*.json")) if sig_dir.is_dir() else []

    # ---- datasets (Table 1) from dataset_stats.json
    ds_path = results_dir / "dataset_stats.json"
    if ds_path.is_file():
        stats = read_json(ds_path)
        for ds_id, d in (stats.get("datasets") or {}).items():
            for field in ("n_clips", "n_groups", "total_minutes"):
                if d.get(field) is not None:
                    v = d[field]
                    nums.set(f"datasets.{ds_id}.{field}", f"{v:.1f}" if isinstance(v, float) else str(v),
                             raw=v, source="dataset_stats.json")
            for lab, c in (d.get("class_counts") or {}).items():
                nums.set(f"datasets.{ds_id}.class_counts.{lab}", str(c), raw=c, source="dataset_stats.json")
                pct = (d.get("class_percent") or {}).get(lab)
                if pct is not None:
                    nums.set(f"datasets.{ds_id}.class_percent.{lab}", f"{pct:.1f}", raw=pct, source="dataset_stats.json")
            for field in ("n_duplicates_removed", "n_clips_raw"):
                if d.get(field) is not None:
                    nums.set(f"datasets.{ds_id}.{field}", str(d[field]), raw=d[field], source="dataset_stats.json")
        split_ls = stats.get("label_set_for_splits")
        for ls, per_ds in (stats.get("label_sets") or {}).items():
            for ds_id, info in per_ds.items():
                pre = f"datasets.label_set.{ls}"
                nums.set(f"{pre}.k", str(len(info["labels"])), raw=len(info["labels"]), source="dataset_stats.json")
                nums.set(f"{pre}.classes", ", ".join(info["labels"]), source="dataset_stats.json")
                nums.set(f"{pre}.min_count", str(info["min_class_count"]), raw=info["min_class_count"],
                         source="dataset_stats.json")
        if split_ls:
            nums.set("datasets.label_set.primary", split_ls, source="dataset_stats.json")
        if stats.get("names_for_abstract"):
            nums.set("datasets.names_for_abstract", stats["names_for_abstract"], source="dataset_stats.json")
        for ds_id, o in (stats.get("overlap") or {}).items():
            for field in ("n_clips", "n_clips_duplicating_eval_set", "n_generated_copies"):
                if o.get(field) is not None:
                    nums.set(f"overlap.{ds_id}.{field}", str(o[field]), raw=o[field], source="dataset_stats.json")
    else:
        warnings.append("results/dataset_stats.json not found (Phase 1: python -m src.dataset_stats)")

    # ---- split leakage (how many test clips a random split puts next to same-group training clips)
    sp_path = results_dir / "splits_report.json"
    if sp_path.is_file():
        sp = read_json(sp_path)
        for ds_id, d in (sp.get("datasets") or {}).items():
            v = (d.get("random_split_test_clips_with_group_in_train") or {}).get("mean")
            if v is not None:
                nums.set(f"leak.{ds_id}.random_split_test_share_group_with_train", fmt("pct", v), raw=v,
                         source="splits_report.json")
        nums.set("splits.folds_file_sha256", sp.get("folds_file_sha256"), source="splits_report.json")

    # ---- DAPT corpus size
    dapt_path = results_dir / "dapt_stats.json"
    if dapt_path.is_file():
        h = read_json(dapt_path).get("hours_unlabelled")
        if h is not None:
            nums.set("dapt.hours_unlabelled", f"{h:.1f}", raw=h, source="dapt_stats.json")
    dapt_runs = []
    for st in sorted((results_dir / "dapt").glob("D01__*/status.json")):
        if read_json(st).get("status") == "completed" and (st.parent / "train_log.csv").is_file():
            dapt_runs.append((read_json(st).get("finished_utc", ""), st.parent))
    if dapt_runs:
        from src.dapt import plot_loss

        run_dir = max(dapt_runs)[1]
        plot_loss(run_dir / "train_log.csv", results_dir / "figures" / "fig6_dapt_loss")
        s = read_json(run_dir / "summary.json")
        nums.set("dapt.steps", str(s.get("steps_run")), raw=s.get("steps_run"), source=run_dir.name)

    # ---- Phase 6 figures (fig2 risk-coverage, fig3 reliability)
    fig_sources: dict[str, list[str]] = {}
    if project.get("figures"):
        from src.selective import make_figures

        fig_sources = make_figures(results_dir, chosen, project["figures"],
                                   project.get("evaluation", {}).get("ece_bins", 15))
        write_json(results_dir / "figures" / "figure_sources.json", fig_sources)

    # ---- per-class recall (every T3 run; mean ± std over folds × seeds) + fig1 / fig4 / figA (src/figures.py)
    from src import figures as F

    for row in rows.get("T3", []):
        summ, rec = summaries.get(("T3", row)), chosen.get(("T3", row))
        for lab, ms in ((summ or {}).get("per_class_recall_mean_std") or {}).items():
            nums.set(f"perclass.{slugify(row)}.recall.{lab}", fmt("pct", ms["mean"], ms["std"]),
                     raw={"mean": ms["mean"], "std": ms["std"]}, source=rec["run_id"])
    fig_dir = results_dir / "figures"
    F.plot_pipeline(fig_dir / "fig1_pipeline")
    pooled = (ours_summ or {}).get("pooled") or {}
    if pooled.get("confusion_matrix") and pooled.get("labels"):
        F.plot_confusion(pooled["labels"], pooled["confusion_matrix"],
                         f"{F.display_name(ours_key[1])}: pooled out-of-fold test, seed 0", fig_dir / "fig4_confusion_ours")
        fig_sources["fig4_confusion_ours"] = [ours_rec["run_id"]]
    di_summ = summaries.get((di.get("table"), di.get("row")))
    series = [(F.display_name(r), s["per_class_recall_mean_std"]) for r, s in
              ((ours_key[1], ours_summ), (di.get("row"), di_summ)) if s and s.get("per_class_recall_mean_std")]
    if series:
        F.plot_per_class_recall(series, fig_dir / "figA_per_class_recall")
        fig_sources["figA_per_class_recall"] = [r["run_id"] for r in (ours_rec, chosen.get((di.get("table"), di.get("row")))) if r]
    if fig_sources:
        write_json(fig_dir / "figure_sources.json", fig_sources)

    # ---- T2 implementation details, read from the saved config/env of the final model and DAPT run
    if ours_rec:
        import yaml

        rd = results_dir / "runs" / ours_rec["run_id"]
        if (rd / "config.yaml").is_file() and (rd / "env.json").is_file():
            c, env = yaml.safe_load((rd / "config.yaml").read_text(encoding="utf-8")), read_json(rd / "env.json")
            tr, aug, ev = c["train"], c["train"].get("augment") or {}, c.get("eval", {})
            src = ours_rec["run_id"]
            nums.set("T2.framework", f"PyTorch {env.get('torch', '?').split('+')[0]}, Hugging Face Transformers "
                     f"{env.get('transformers', '?')}, PEFT {env.get('peft', '?')}", source=src)
            nums.set("T2.hardware", f"1× {env.get('gpu', 'CPU')} ({env.get('platform', '?')})", source=src)
            nums.set("T2.clip_seconds", f"{float(c['data']['clip_seconds']):g}", source=src)
            ft = (f"{tr['epochs']} epochs max (early stopping on val macro-F1, patience {tr.get('patience')}), "
                  f"batch {tr['batch_size']} × {tr['grad_accum']} accumulation, AdamW LR {tr['lr']:g}, "
                  f"weight decay {tr['weight_decay']:g}, warm-up {float(tr.get('warmup_ratio', 0)) * 100:g}%, "
                  f"loss {tr.get('loss')}" + (f" (γ {tr.get('focal_gamma'):g})" if tr.get("loss") == "focal" else "")
                  + (", full fine-tuning (no LoRA)" if not c["model"].get("peft") else ", LoRA"))
            nums.set("T2.finetuning", ft, source=src)
            parts = []
            sa = aug.get("specaugment")
            if sa:
                parts.append(f"SpecAugment {sa['n_masks']} frequency mask ≤ {sa['freq_mask']} bins + "
                             f"{sa['n_masks']} time mask ≤ {sa['time_mask']} frames")
            if aug.get("mixup_alpha"):
                parts.append(f"mixup α {aug['mixup_alpha']:g}")
            if aug.get("noise_snr_db"):
                lo, hi = aug["noise_snr_db"]
                parts.append(f"ESC-50 background noise, SNR {lo:g}–{hi:g} dB, p {aug.get('noise_prob', 1):g}")
                nums.set("T2.finetune.snr_range", f"{lo:g}–{hi:g}", source=src)
            nums.set("T2.augmentation", "; ".join(parts) if parts else "none", source=src)
            seeds = ev.get("seeds") or []
            if seeds:
                nums.set("T2.seeds", f"{min(seeds)}–{max(seeds)}" if len(seeds) > 1 else str(seeds[0]), source=src)
                nums.set("T2.n_folds", str(len(ev.get("folds") or [])), source=src)
    if dapt_runs:
        import yaml

        dd = max(dapt_runs)[1]
        if (dd / "config.yaml").is_file():
            dc = yaml.safe_load((dd / "config.yaml").read_text(encoding="utf-8"))
            for k, v in (("steps", dc.get("max_steps")), ("batch", dc.get("batch_size")), ("lr", dc.get("lr")),
                         ("mask_prob", dc.get("mask_prob"))):
                if v is not None:
                    nums.set(f"T2.dapt.{k}", f"{v:g}" if isinstance(v, float) else str(v), source=dd.name)
            nums.set("T2.pretraining", f"{dc.get('max_steps')} steps, batch {dc.get('batch_size')}, LR "
                     f"{float(dc.get('lr')):g}, masking p {dc.get('mask_prob')} (span {dc.get('mask_length')})",
                     source=dd.name)

    # ---- Phase 9 efficiency (one machine for every model: src/efficiency.py)
    teff_table = []
    eff_path = results_dir / "efficiency.json"
    if eff_path.is_file():
        eff = read_json(eff_path)
        hw = eff.get("hardware", {})
        nums.set("eff.hardware", f"{hw.get('processor', '?')}, {hw.get('cpu_count_logical', '?')} logical CPUs, "
                 f"torch {hw.get('torch', '?')}", raw=hw, source="efficiency.json")
        for m in eff.get("models", {}).values():
            slug = slugify(m["row"])
            vals = {"params_m": (m["params_total"] / 1e6, "{:.2f}"),
                    "trainable_m": (m["params_trainable"] / 1e6, "{:.3f}"),
                    "size_mb": (m["size_mb_fp32"], "{:.1f}" if m["size_mb_fp32"] >= 10 else "{:.2f}"),
                    "cpu_ms": (m["ms_cpu_all_threads"]["median"], "{:.0f}" if m["ms_cpu_all_threads"]["median"] >= 10 else "{:.1f}"),
                    "cpu_ms_1t": (m["ms_cpu_1_thread"]["median"], "{:.0f}" if m["ms_cpu_1_thread"]["median"] >= 10 else "{:.1f}")}
            cells = {}
            for suffix, (v, f) in vals.items():
                key = f"eff.{slug}.{suffix}"
                nums.declare(key)
                nums.set(key, f.format(v), raw=v, source=f"efficiency.json ({m['experiment_id']})")
                cells[suffix] = nums.values[key]
            teff_table.append({"row": m["row"], "unit": "1 s frame" if m["table"] == "T6" else "7 s clip",
                               **cells, "run_id": m["experiment_id"]})
    else:
        warnings.append("results/efficiency.json not found (Phase 9: python -m src.efficiency)")

    # ---- protocol warnings: all reason-task paper runs must share one folds file (E90 excluded)
    hashes: dict[str, list[str]] = {}
    for rec in chosen.values():
        if rec.get("task") == "reason" and not str(rec.get("experiment_id", "")).startswith("E90"):
            hashes.setdefault(str(rec.get("folds_file_sha256")), []).append(rec["run_id"])
    if len(hashes) > 1:
        warnings.append("paper runs used DIFFERENT folds files (cannot be compared): "
                        + "; ".join(f"{h[:12]}…: {', '.join(r)}" for h, r in hashes.items()))
    if "None" in hashes:
        warnings.append(f"runs without a folds-file hash (debug/no data): {', '.join(hashes['None'])}")

    resolved = resolve_registry(records)
    superseded = [r for r in resolved.values() if r.get("superseded_by")]
    failed = [r for r in resolved.values() if r.get("status") == "failed"]

    return {
        "numbers": nums, "warnings": warnings, "records": list(resolved.values()),
        "superseded": superseded, "failed": failed, "sig_files": sig_files,
        "fig_sources": fig_sources,
        "tables": {"T3": t3_table, "T4": t4_table, "T5": t5_table, "T5b": t5b_table, "T6": t6_table,
                   "Teff": teff_table},
    }


# --------------------------------------------------------------------------- writers
def _cell(v: Any) -> str:
    return UNFILLED if v is None else str(v)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


TABLE_SPECS = {
    "T3": (["Model", "Accuracy (%)", "Macro-F1", "Balanced acc. (%)", "ECE", "Run"],
           lambda r: [r["row"], r["accuracy"], r["macro_f1"], r["balanced_acc"], r["ece"], r["run_id"] or ""]),
    "T4": (["Configuration", "Macro-F1", "Δ vs full", "Run"],
           lambda r: [r["row"], r["macro_f1"], r["delta"], r["run_id"] or ""]),
    "T5": (["Coverage", "Precision (%)", "Macro-F1", "Threshold τ (val)"],
           lambda r: [f"{float(r['coverage']) * 100:.0f}%", r["precision"], r["macro_f1"], r["threshold"]]),
    "T5b": (["Model", "ECE before", "ECE after", "AURC", "Prec. @90% (%)", "Prec. @70% (%)", "Prec. @50% (%)",
             "Macro-F1 @90%", "Macro-F1 @70%", "Macro-F1 @50%", "Run"],
            lambda r: [r["row"], r["ece_before"], r["ece_after"], r["aurc"], r["prec90"], r["prec70"], r["prec50"],
                       r["mf1_90"], r["mf1_70"], r["mf1_50"], r["run_id"] or ""]),
    "T6": (["Detector", "Accuracy (%)", "F1", "Precision (%)", "Recall (%)", "ROC-AUC", "False alarms / h", "Run"],
           lambda r: [r["row"], r["accuracy"], r["f1"], r["precision"], r["recall"], r["roc_auc"],
                      r["false_alarms_per_hour"], r["run_id"] or ""]),
    "Teff": (["Model", "Input", "Params (M)", "Trainable (M)", "Size fp32 (MB)", "CPU ms (all threads)",
              "CPU ms (1 thread)", "Experiment"],
             lambda r: [r["row"], r["unit"], r["params_m"], r["trainable_m"], r["size_mb"], r["cpu_ms"],
                        r["cpu_ms_1t"], r["run_id"] or ""]),
}
TABLE_TITLES = {"T3": "Table 3 — Cry-reason classification (mean ± std over folds × seeds)",
                "T4": "Table 4 — Ablations", "T5": "Table 5 — Selective prediction (ours)",
                "T5b": "Table 5b — Calibration and selective prediction, every model (test folds; τ from val)",
                "T6": "Table 6 — Stage-1 cry detection",
                "Teff": "Efficiency — one machine, batch 1, raw audio to logits incl. features (src/efficiency.py)"}


def write_outputs(result: dict, results_dir: Path = DEFAULT_RESULTS_DIR) -> dict[str, Path]:
    results_dir = Path(results_dir)
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    nums: Numbers = result["numbers"]

    pn = results_dir / "paper_numbers.json"
    payload = nums.to_json()
    payload["_generated_utc"] = iso(utcnow())
    write_json(pn, payload)

    table_md = {}
    for t, (headers, fn) in TABLE_SPECS.items():
        rows = [fn(r) for r in result["tables"][t]]
        table_md[t] = md_table(headers, rows)
        (tables_dir / f"{t}.md").write_text(f"### {TABLE_TITLES[t]}\n\n{table_md[t]}\n", encoding="utf-8")
        with open(tables_dir / f"{t}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            w.writerows([["" if c is None else c for c in r] for r in rows])

    recs = sorted(result["records"], key=lambda r: r.get("created_utc") or "")
    run_rows = [[r.get("run_id"), r.get("status"), r.get("paper_table"), r.get("paper_row"),
                 fmt("f3", r.get("macro_f1_mean"), r.get("macro_f1_std")), r.get("superseded_by") or ""]
                for r in recs]
    fig_dir = results_dir / "figures"
    fsrc = result.get("fig_sources", {})
    fig_rows = [[name, desc, "yes" if (fig_dir / f"{name}.png").is_file() else "missing", ", ".join(fsrc.get(name, []))]
                for name, desc in FIGURES]
    unfilled = nums.unfilled()
    warnings = list(result["warnings"])
    if result["superseded"]:
        warnings.append("superseded runs (ignored): " + ", ".join(r["run_id"] for r in result["superseded"]))

    parts = [
        f"# RESULTS — {nums.values['model_name']}",
        "",
        f"_Generated by `python -m src.report` at {iso(utcnow())}. Do not edit by hand._",
        "",
        f"**Paper numbers filled:** {len(nums.values) - len(unfilled)} / {len(nums.values)}",
        "",
        "## Run index",
        "",
        md_table(["run_id", "status", "table", "row", "macro-F1", "superseded_by"], run_rows) if run_rows
        else "_No runs registered yet._",
        "",
    ]
    for t in ("T3", "T4", "T5", "T5b", "T6", "Teff"):
        parts += [f"## {TABLE_TITLES[t]}", "", table_md[t], ""]
    parts += ["## Figures", "", md_table(["file", "paper figure", "exists", "source runs (Phase 6 figures)"], fig_rows), "",
              "## Significance", ""]
    if result["sig_files"]:
        for p in result["sig_files"]:
            s = read_json(p)
            ci = s.get("ci95") or [None, None]
            ci_s = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if None not in ci else "n/a"
            parts.append(f"- `{s.get('run_a', p.name)}` vs `{s.get('run_b', '')}`: Δ macro-F1 "
                         f"{fmt_delta(s.get('delta_macro_f1'))}, 95% CI {ci_s}, p {fmt_p(s.get('p_value'))} "
                         f"({s.get('method', '')}, {s.get('resamples', '?')} resamples)")
    else:
        parts.append("_No significance tests yet._")
    parts += ["", "## Failed runs", ""]
    parts += [f"- `{r['run_id']}`: {r.get('notes', '')}" for r in result["failed"]] or ["_None._"]
    parts += ["", "## Warnings", ""]
    parts += [f"- {w}" for w in warnings] or ["_None._"]
    parts += ["", f"## Unfilled paper placeholders ({len(unfilled)})", ""]
    parts += [f"- [ ] `{k}`" for k in unfilled] or ["_All placeholders filled._"]
    md = results_dir / "RESULTS.md"
    md.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return {"paper_numbers": pn, "results_md": md, "tables": tables_dir}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    args = ap.parse_args(argv)
    result = build(Path(args.results_dir))
    paths = write_outputs(result, Path(args.results_dir))
    n = result["numbers"]
    print(f"{len(n.values) - len(n.unfilled())}/{len(n.values)} paper numbers filled; "
          f"{len(result['warnings'])} warning(s)")
    for p in paths.values():
        try:
            print(f"  wrote {Path(p).resolve().relative_to(REPO_ROOT)}")
        except ValueError:
            print(f"  wrote {p}")


if __name__ == "__main__":
    main()
