import hashlib
import json

import docx
import pytest

from src import fill_paper as FP


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_render_text_substitutes_and_flags_missing():
    text, missing = FP.render_text("F1 {a.b} vs {c}; literal {{x}}", {"a.b": "0.194 ± 0.025", "c": None})
    assert text == "F1 0.194 ± 0.025 vs [c: not yet produced]; literal {x}"
    assert missing == ["c"]


def test_filler_specs():
    f = FP.Filler({"k": "0.194 ± 0.025", "h": "6.5", "_sources": {}})
    assert f.value("k") == "0.194 ± 0.025"
    assert f.value("mean:k") == "0.194"
    assert f.value("tmpl:{h} hours") == "6.5 hours"
    assert f.value("lit:wav2vec 2.0") == "wav2vec 2.0"
    assert f.value(None) is None and f.value("nope") is None and f.value("tmpl:{nope} x") is None
    assert "nope" in f.missing


@pytest.fixture
def numbers(tmp_path):
    n = {"model_name": "CRYSTAL", "abstract.reason_macro_f1_ours": "0.194 ± 0.025",
         "T3.ours_crystal.macro_f1": "0.194 ± 0.025", "T5.cov90.precision": "78.8 ± 6.5"}
    p = tmp_path / "paper_numbers.json"
    p.write_text(json.dumps(n), encoding="utf-8")
    return p


@pytest.mark.skipif(not FP.TEMPLATE.is_file(), reason="paper template is kept locally, not in the public repo")
def test_fill_copies_template_and_keeps_unfilled_highlighted(tmp_path, numbers, monkeypatch):
    monkeypatch.setattr(FP, "DRAFTS_SRC", tmp_path / "no_drafts")
    before = _sha(FP.TEMPLATE)
    out = FP.fill(FP.TEMPLATE, numbers, tmp_path / "drafts")
    assert _sha(FP.TEMPLATE) == before  # original never modified
    d = docx.Document(out)
    abstract = next(p for p in d.paragraphs if "Automatic analysis of infant cries" in p.text)
    assert "CRYSTAL" in abstract.text and "[MODEL NAME]" not in abstract.text
    assert "macro-F1 of 0.194 (versus [0.XX]" in abstract.text  # mean only; missing stays a placeholder
    still = [r.text for r in abstract.runs if r.font.highlight_color is not None]
    assert "[0.XX]" in still and "0.194" not in still
    t3 = d.tables[2]
    ours = next(r for r in t3.rows if r.cells[0].text == "Ours (CRYSTAL)")
    assert ours.cells[2].text == "0.194 ± 0.025" and ours.cells[1].text == "[ ]"
    assert d.tables[4].rows[1].cells[2].text == "78.8 ± 6.5"
    report = (tmp_path / "drafts" / "fill_report.md").read_text(encoding="utf-8")
    assert "Still highlighted" in report and "abstract.detection_acc" in report


def test_render_drafts(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "x.md.tmpl").write_text("value {a}", encoding="utf-8")
    out = FP.render_drafts({"a": "1.0"}, src, tmp_path)
    assert out[0].name == "x.md" and out[0].read_text(encoding="utf-8") == "value 1.0"


def test_render_tex_escapes_and_flags_missing():
    text, missing = FP.render_tex(r"\textbf{F1} <<a.b>> and <<c>> at <<p>>", {"a.b": "0.194 ± 0.025", "c": None, "p": "50%"})
    assert text == r"\textbf{F1} 0.194 $\pm$ 0.025 and \textbf{[c: not yet produced]} at 50\%"
    assert missing == ["c"]


def test_render_drafts_tex_subfolder(tmp_path):
    src, figs, out = tmp_path / "src", tmp_path / "figs", tmp_path / "out"
    (src / "short").mkdir(parents=True)
    figs.mkdir()
    (src / "short" / "main.tex.tmpl").write_text(r"\section{R} <<k_1>>", encoding="utf-8")
    (src / "short" / "refs.bib").write_text("@misc{a}", encoding="utf-8")
    (figs / "fig2.pdf").write_bytes(b"%PDF")
    FP.render_drafts({"k_1": "a_b"}, src, out, figs)
    assert (out / "short" / "main.tex").read_text(encoding="utf-8") == r"\section{R} a\_b"
    assert (out / "short" / "refs.bib").read_text(encoding="utf-8") == "@misc{a}"
    assert (out / "short" / "figures" / "fig2.pdf").read_bytes() == b"%PDF"


def _doc_with(paragraphs, table_rows=None):
    import docx

    d = docx.Document()
    for text in paragraphs:
        d.add_paragraph(text)
    if table_rows:
        t = d.add_table(rows=len(table_rows), cols=len(table_rows[0]))
        for r, row in enumerate(table_rows):
            for c, text in enumerate(row):
                t.cell(r, c).text = text
    return d


def test_apply_rewrites_ops(tmp_path):
    from src import figures

    figures.plot_pipeline(tmp_path / "fig")
    d = _doc_with(["Old abstract text", "Delete me", "Anchor here", "Same", "Same"],
                  [["Dataset", "Role"], ["Ours", "x"], ["Other", "y"]])
    spec = [
        {"find": "Old abstract", "text": "F1 {a} with **bold** and ==todo== and {missing}"},
        {"find": "Delete me", "delete": True},
        {"after": "Anchor here", "insert": ["first {a}", "second"]},
        {"find": "Same", "nth": 1, "text": "second same"},
        {"after": "first", "image": "fig.png", "caption": "Figure 1. {a}"},
        {"table": 0, "row": "Ours", "cells": [None, "role {a}"]},
        {"table": 0, "row": "Other", "delete_row": True},
        {"find": "No such paragraph", "delete": True},
    ]
    applied, problems = FP.apply_rewrites(d, spec, {"a": "0.194 ± 0.025"}, tmp_path)
    texts = [p.text for p in d.paragraphs]
    assert texts[0] == "F1 0.194 ± 0.025 with bold and todo and [missing: not yet produced]"
    runs = {r.text: r for r in d.paragraphs[0].runs}
    assert runs["bold"].bold and runs["todo"].font.highlight_color is not None
    assert runs["[missing: not yet produced]"].font.highlight_color is not None  # gap stays visible
    assert "Delete me" not in texts
    assert texts[1:7] == ["Anchor here", "first 0.194 ± 0.025", "", "Figure 1. 0.194 ± 0.025", "second", "Same"]
    assert texts[7] == "second same"
    assert len(d.inline_shapes) == 1
    assert [r.cells[1].text for r in d.tables[0].rows] == ["Role", "role 0.194 ± 0.025"]
    assert len(applied) == 7 and problems == ["'No such paragraph': 0 matching paragraphs"]


def test_figures_render(tmp_path):
    from src import figures

    figures.plot_confusion(["a", "b"], [[3, 1], [0, 2]], "Ours", tmp_path / "cm")
    figures.plot_per_class_recall([("Ours", {"a": {"mean": 0.5, "std": 0.1}, "b": {"mean": 0.0, "std": 0.0}})],
                                  tmp_path / "rec")
    assert all((tmp_path / f"{s}.{e}").stat().st_size > 0 for s in ("cm", "rec") for e in ("png", "pdf"))
    assert figures.display_name("Ours (CRYSTAL)") == "Ours" and figures.display_name("MFCC + SVM") == "MFCC + SVM"
    assert b"CRYSTAL" not in (tmp_path / "cm.pdf").read_bytes()
