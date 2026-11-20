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
