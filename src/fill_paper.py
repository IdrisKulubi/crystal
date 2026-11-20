"""Phase 10 — fill the paper template's yellow placeholders from results/paper_numbers.json.

    python -m src.report && python -m src.fill_paper
    # -> paper/drafts/<model>_paper_filled.docx + paper/drafts/fill_report.md

Rules (CLAUDE.md, HANDOFF §9-10):
* The original template is never modified: it is copied, and only the copy is edited.
* Every number written comes from paper_numbers.json (no number is typed here). The only literal text
  inserted is plain fact from the repo's own config (encoder list, test name, label names).
* A filled run loses its yellow highlight; anything not filled stays yellow, so what is left to do is
  visible in Word. Writing tasks ("[Describe ...]", "[Interpret ...]") are never auto-filled: they are
  drafted in paper/drafts/*.md for Idris to review.
* Tables 3-5 in the template have fewer / different rows than the experiments; their body rows are
  rebuilt from the report's tables (same keys as RESULTS.md), keeping the template's formatting.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
from pathlib import Path

from src.config import REPO_ROOT, load_project
from src.report import slugify

TEMPLATE = REPO_ROOT / "paper" / "Infant_Cry_Research_Paper_Template.docx"
DRAFTS = REPO_ROOT / "paper" / "drafts"
NUMBERS = REPO_ROOT / "results" / "paper_numbers.json"

# Body paragraphs: anchor text (unique substring of the paragraph) -> one spec per highlighted run, in
# order. Spec: "key" (value as stored), "mean:key" (value before " ±"), "tmpl:...{key}..." (key inside
# fixed text), "lit:text" (fact from config), None (leave highlighted for Idris / the drafts).
BODY = {
    "Automatic analysis of infant cries promises": [
        "model_name", "datasets.names_for_abstract", "model_name", "mean:abstract.detection_acc",
        "mean:abstract.reason_macro_f1_ours", "mean:abstract.reason_macro_f1_deepinfant",
        "mean:abstract.selective_precision", "abstract.selective_coverage", "mean:abstract.ece_ours", None],
    "Domain-adaptive self-supervised pretraining. We continue": ["tmpl:{dapt.hours_unlabelled} hours"],
    "Calibrated selective prediction. The system abstains": ["mean:abstract.selective_precision"],
    "We map them to a common taxonomy of": ["datasets.label_set.donateacry_v1.k",
                                          "datasets.label_set.donateacry_v1.classes", None],
    "Training clips are cut to": ["T2.clip_seconds"],
    "Starting from a general-purpose checkpoint of": [
        "lit:wav2vec 2.0", "dapt.hours_unlabelled", None, "lit:contrastive masked prediction",
        "T2.dapt.steps", "T2.dapt.batch", "T2.dapt.lr"],
    "The adapted encoder is fine-tuned for cry-reason": [None, None, None, "T2.finetune.snr_range"],
    "Frozen pretrained encoders + linear probe": ["lit:wav2vec 2.0, HuBERT, AST, Whisper encoder"],
    "Statistical significance between our model and baselines": [
        "tmpl:a paired cluster bootstrap over infants ({bootstrap_resamples} resamples)"],
    "We presented [MODEL NAME]": ["model_name", None, None],
}

# Whole-cell fills: (template table index, row, col) -> spec.
CELLS = {
    (0, 1, 1): "tmpl:{datasets.donateacry.n_clips} clips, ~7 s each",
    (1, 1, 1): "T2.framework", (1, 2, 1): "T2.hardware", (1, 3, 1): "T2.pretraining",
    (1, 4, 1): "T2.finetuning", (1, 5, 1): "T2.augmentation",
    (1, 6, 1): "tmpl:{T2.seeds}; results averaged across seeds and {T2.n_folds} infant-grouped folds",
    (5, 2, 2): "lit:pain_discomfort", (5, 4, 2): "lit:burping",
}


class Filler:
    def __init__(self, numbers: dict, extra: dict | None = None):
        self.n = {k: v for k, v in numbers.items() if not k.startswith("_")}
        self.n.update(extra or {})
        self.filled, self.missing = [], []

    def value(self, spec: str | None) -> str | None:
        if spec is None:
            return None
        if spec.startswith("lit:"):
            return spec[4:]
        if spec.startswith("tmpl:"):
            keys = re.findall(r"{([^}]+)}", spec[5:])
            if any(self.n.get(k) is None for k in keys):
                self.missing += [k for k in keys if self.n.get(k) is None]
                return None
            return re.sub(r"{([^}]+)}", lambda m: str(self.n[m.group(1)]), spec[5:])
        key, mean = (spec[5:], True) if spec.startswith("mean:") else (spec, False)
        v = self.n.get(key)
        if v is None:
            self.missing.append(key)
            return None
        v = str(v)
        return v.split(" ±")[0] if mean else v


def _highlighted(p):
    return [r for r in p.runs if r.font.highlight_color is not None]


def _set(run, text: str) -> None:
    run.text = text
    run.font.highlight_color = None


def fill_body(doc, f: Filler) -> None:
    for anchor, specs in BODY.items():
        paras = [p for p in doc.paragraphs if anchor in p.text]
        if len(paras) != 1:
            raise ValueError(f"anchor {anchor!r} matched {len(paras)} paragraphs (template changed?)")
        runs = _highlighted(paras[0])
        if len(runs) != len(specs):
            raise ValueError(f"{anchor!r}: {len(runs)} highlighted runs, spec has {len(specs)}")
        for run, spec in zip(runs, specs):
            v = f.value(spec)
            if v is not None:
                f.filled.append((anchor[:40], run.text, v))
                _set(run, v)


def fill_cell(cell, text: str) -> None:
    p = cell.paragraphs[0]
    runs = p.runs
    _set(runs[0], text)
    for r in runs[1:]:
        r._r.getparent().remove(r._r)
    for extra in cell.paragraphs[1:]:
        extra._p.getparent().remove(extra._p)


def fill_cells(doc, f: Filler) -> None:
    for (ti, ri, ci), spec in CELLS.items():
        cell = doc.tables[ti].rows[ri].cells[ci]
        v = f.value(spec)
        if v is not None:
            f.filled.append((f"table{ti} r{ri}c{ci}", cell.text, v))
            fill_cell(cell, v)


def rebuild_rows(table, rows: list[list[str | None]], keep_header: int = 1) -> None:
    """Replace the body rows with ``rows`` (None = leave a yellow "[ ]"), copying the first body row's format."""
    proto = copy.deepcopy(table.rows[keep_header]._tr)
    for r in list(table.rows)[keep_header:]:
        r._tr.getparent().remove(r._tr)
    for values in rows:
        tr = copy.deepcopy(proto)
        table._tbl.append(tr)
        row = table.rows[-1]
        for cell, v in zip(row.cells, values):
            p = cell.paragraphs[0]
            if not p.runs:
                p.add_run("")
            for r in p.runs[1:]:
                r._r.getparent().remove(r._r)
            run = p.runs[0]
            if v is None:
                run.text = "[ ]"
                from docx.enum.text import WD_COLOR_INDEX

                run.font.highlight_color = WD_COLOR_INDEX.YELLOW
            else:
                _set(run, v)


def fill_tables(doc, f: Filler, project: dict) -> None:
    n = f.n
    rows3 = []
    for row in project["paper_rows"]["T3"]:
        s = slugify(row)
        acc = n.get(f"T3.{s}.accuracy")
        if row == "DeepInfant (reported)" and acc:
            acc = acc.replace(" (reported)", "*")
        cells = [acc, n.get(f"T3.{s}.macro_f1"), n.get(f"T3.{s}.balanced_acc"), n.get(f"T3.{s}.ece")]
        if row == "DeepInfant (reported)":
            cells = [c if c is not None else "—" for c in cells]
        rows3.append([row] + cells)
    rebuild_rows(doc.tables[2], rows3)

    rows4 = []
    leak_rows = {project["roles"][r]["row"] for r in ("leak_random_ours", "leak_random_deepinfant")
                 if r in project.get("roles", {})}
    ours_row = project["roles"]["ours"]["row"]
    for row in project["paper_rows"]["T4"]:
        s = slugify(row)
        if row == ours_row:
            rows4.append([f"Full model ({project['model_name']})", n.get(f"T3.{slugify(ours_row)}.macro_f1"), "—"])
        elif row in leak_rows:
            d = n.get(f"T4.{s}.delta_vs_grouped")
            rows4.append([row, n.get(f"T4.{s}.macro_f1"), f"{d} vs. its grouped split" if d else None])
        else:
            rows4.append([row, n.get(f"T4.{s}.macro_f1"), n.get(f"T4.{s}.delta")])
    rebuild_rows(doc.tables[3], rows4)

    covs = ["cov100", "cov90", "cov70", "cov50"]
    t5 = doc.tables[4]
    for ci, c in enumerate(covs, start=1):
        for ri, metric in ((1, "precision"), (2, "threshold")):
            cell = t5.rows[ri].cells[ci]
            v = n.get(f"T5.{c}.{metric}")
            if c == "cov100" and metric == "threshold":
                continue  # template already has "—"
            if v is not None:
                f.filled.append((f"Table 5 {c} {metric}", cell.text, v))
                fill_cell(cell, v)
    for table, rows in ((3, rows3), (4, rows4)):
        f.filled += [(f"Table {table} rebuilt", r[0], " | ".join(str(x) for x in r[1:])) for r in rows]


def remaining(doc) -> list[str]:
    out = []
    for i, p in enumerate(doc.paragraphs):
        out += [f"p{i}: {r.text}" for r in _highlighted(p)]
    for ti, t in enumerate(doc.tables):
        for ri, row in enumerate(t.rows):
            for ci, cell in enumerate(row.cells):
                for p in cell.paragraphs:
                    out += [f"table{ti} r{ri}c{ci}: {r.text}" for r in _highlighted(p)]
    return list(dict.fromkeys(out))  # merged cells repeat


DRAFTS_SRC = REPO_ROOT / "paper" / "drafts_src"
_TOKEN = re.compile(r"(?<!\{)\{([A-Za-z0-9_.]+)\}(?!\})")


def render_text(text: str, numbers: dict) -> tuple[str, list[str]]:
    """``{key}`` -> value from paper_numbers; a key without a value becomes ``[key: not yet produced]``.
    ``{{x}}`` is a literal ``{x}``."""
    missing = []

    def sub(m):
        v = numbers.get(m.group(1))
        if v is None:
            missing.append(m.group(1))
            return f"[{m.group(1)}: not yet produced]"
        return str(v)
    return _TOKEN.sub(sub, text).replace("{{", "{").replace("}}", "}"), missing


# LaTeX templates use <<key>> because {key} collides with LaTeX braces; values are escaped for LaTeX.
_TEX_TOKEN = re.compile(r"<<([A-Za-z0-9_.]+)>>")
_TEX_ESCAPES = {"\\": r"\textbackslash{}", "%": r"\%", "&": r"\&", "#": r"\#", "_": r"\_", "$": r"\$",
                "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
                "±": r"$\pm$", "−": r"$-$", "Δ": r"$\Delta$", "×": r"$\times$", "≤": r"$\le$", "≥": r"$\ge$"}


def tex_escape(s: str) -> str:
    return "".join(_TEX_ESCAPES.get(c, c) for c in str(s))


def render_tex(text: str, numbers: dict) -> tuple[str, list[str]]:
    """``<<key>>`` -> LaTeX-escaped value; a key without a value becomes a bold ``[key: not yet produced]``."""
    missing = []

    def sub(m):
        v = numbers.get(m.group(1))
        if v is None:
            missing.append(m.group(1))
            return r"\textbf{[" + tex_escape(m.group(1)) + ": not yet produced]}"
        return tex_escape(v)
    return _TEX_TOKEN.sub(sub, text), missing


FIGURES = REPO_ROOT / "results" / "figures"


def render_drafts(numbers: dict, src_dir: Path = DRAFTS_SRC, out_dir: Path = DRAFTS,
                  figures_dir: Path = FIGURES) -> list[Path]:
    """Render every ``*.md.tmpl`` / ``*.tex.tmpl`` under ``src_dir`` (sub-folders kept) into ``out_dir``.
    Other files in sub-folders (e.g. ``refs.bib``) are copied as they are; a folder with a LaTeX
    template also gets the report's figure PDFs in ``figures/`` (copied, never regenerated here)."""
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    out = []
    if not src_dir.is_dir():
        return out
    for t in sorted(src_dir.rglob("*")):
        if not t.is_file():
            continue
        rel = t.relative_to(src_dir)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if t.name.endswith((".md.tmpl", ".tex.tmpl")):
            render = render_tex if t.name.endswith(".tex.tmpl") else render_text
            text, missing = render(t.read_text(encoding="utf-8"), numbers)
            if missing:
                print(f"  {rel}: no value yet for {sorted(set(missing))}")
            dst = dst.with_name(t.name[:-len(".tmpl")])
            dst.write_text(text, encoding="utf-8")
            if t.name.endswith(".tex.tmpl") and Path(figures_dir).is_dir():
                (dst.parent / "figures").mkdir(exist_ok=True)
                for fig in sorted(Path(figures_dir).glob("*.pdf")):
                    shutil.copyfile(fig, dst.parent / "figures" / fig.name)
            out.append(dst)
        elif rel.parent != Path("."):
            shutil.copyfile(t, dst)
            out.append(dst)
    return out


def fill(template: Path = TEMPLATE, numbers_path: Path = NUMBERS, out_dir: Path = DRAFTS) -> Path:
    import docx

    project = load_project()
    numbers = json.loads(Path(numbers_path).read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{project['model_name']}_paper_filled.docx"
    shutil.copyfile(template, out)  # the template itself is never opened for writing
    doc = docx.Document(out)
    f = Filler(numbers, {"bootstrap_resamples": f"{int(project['evaluation']['bootstrap_resamples']):,}"})
    fill_body(doc, f)
    fill_cells(doc, f)
    fill_tables(doc, f, project)
    doc.save(out)
    left = remaining(docx.Document(out))
    lines = [f"# Fill report — {out.name}", "", f"Generated by `python -m src.fill_paper` from "
             f"`results/paper_numbers.json` ({numbers.get('_generated_utc', '?')}). Do not edit by hand.", "",
             f"Filled: {len(f.filled)} · still highlighted: {len(left)}", "", "## Filled", "",
             "| where | placeholder | value |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in f.filled]
    lines += ["", "## Still highlighted (writing tasks, decisions for Idris, or numbers not yet produced)", ""]
    lines += [f"- [ ] {x}" for x in left]
    if f.missing:
        lines += ["", "## Mapped keys with no value yet", ""] + [f"- `{k}`" for k in sorted(set(f.missing))]
    (out_dir / "fill_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(f.filled)} filled, {len(left)} still highlighted) and fill_report.md")
    nums = {k: v for k, v in numbers.items() if not k.startswith("_")}
    for d in render_drafts(nums, DRAFTS_SRC, out_dir):
        print("wrote", d)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default=str(TEMPLATE))
    ap.add_argument("--out-dir", default=str(DRAFTS))
    a = ap.parse_args(argv)
    fill(Path(a.template), NUMBERS, Path(a.out_dir))


if __name__ == "__main__":
    main()
