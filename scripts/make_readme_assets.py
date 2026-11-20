"""README visuals, generated ONLY from results/paper_numbers.json (no number is typed here).

    python -m src.report && python scripts/make_readme_assets.py     # -> assets/

* assets/banner.png              hero banner with the headline results
* assets/badges/*.svg            shields-style badges (licence, framework, protocol, key metrics)
* assets/model_comparison.png    test macro-F1 of every model vs the majority-class baseline
* assets/leakage.png             infant-grouped vs random clip-level split, DeepInfant vs ours
The waveform in the banner is a synthetic sine pattern (decoration), never real audio.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_project  # noqa: E402

ASSETS = ROOT / "assets"
NAVY, TEAL, MINT, AMBER, GREY, INK = "#0B1F3A", "#0E7C86", "#5FD3BC", "#F2A541", "#8A94A6", "#E8EEF5"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path: Path, dpi: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, metadata={"Software": None}, facecolor=fig.get_facecolor())


def load_numbers() -> tuple[dict, dict]:
    n = json.loads((ROOT / "results" / "paper_numbers.json").read_text(encoding="utf-8"))
    return n, n.get("_raw", {})


def mean_of(raw: dict, key: str) -> float | None:
    v = raw.get(key)
    return v["mean"] if isinstance(v, dict) else v


# --------------------------------------------------------------------------- banner
def banner(n: dict, path: Path) -> None:
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import FancyBboxPatch

    fig = plt.figure(figsize=(12.8, 4.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 40)
    ax.axis("off")
    grad = np.linspace(0, 1, 512)[None, :]
    ax.imshow(grad, extent=(0, 128, 0, 40), aspect="auto", cmap=LinearSegmentedColormap.from_list("g", [NAVY, "#123A5C", TEAL]))

    # decorative synthetic waveform: bursts like cry expirations
    x = np.linspace(0, 128, 3000)
    env = sum(np.exp(-((x - c) / w) ** 2) for c, w in ((14, 4), (30, 5), (47, 3.5), (63, 6)))
    y = 7 + 4.5 * env * np.sin(2 * np.pi * x / 0.9) * (0.6 + 0.4 * np.sin(x / 3))
    ax.plot(x, y, color=MINT, lw=0.8, alpha=0.55)
    ax.plot(x, 7 + 0 * x, color=MINT, lw=0.4, alpha=0.3)

    name = n["model_name"]
    ax.text(6, 30, name, color="white", fontsize=46, fontweight="bold", va="center")
    ax.text(6.3, 22.5, "CRY Self-supervised Transformer with Abstention Learning", color=INK, fontsize=12.5, va="center")
    ax.text(6.3, 17.5, "Leak-free, calibrated infant-cry analysis — evaluated by baby, not by clip",
            color=MINT, fontsize=12, va="center", style="italic")

    cards = [
        ("Cry detection F1", n["T6.ast_detector.f1"].split(" ±")[0], "AST detector,|infant-grouped folds"),
        ("Cry-reason macro-F1", n["T3.ours_crystal.macro_f1"].split(" ±")[0],
         f"vs {n['T3.deepinfant_reproduced.macro_f1'].split(' ±')[0]} for|DeepInfant (reproduced)"),
        ("Leakage inflation", n["T4.ours_random_clip_level_split.delta_vs_grouped"],
         "macro-F1 gained by|a random clip split"),
    ]
    x0, w, h, gap = 71, 17.2, 19, 1.4
    for i, (title, value, sub) in enumerate(cards):
        cx = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((cx, 11), w, h, boxstyle="round,pad=0.3,rounding_size=1.5",
                                    fc="white", ec="none", alpha=0.10))
        ax.text(cx + w / 2, 26.5, title, color=INK, fontsize=9.5, ha="center", va="center")
        ax.text(cx + w / 2, 20.3, value, color=AMBER if i == 2 else "white", fontsize=24, fontweight="bold",
                ha="center", va="center")
        ax.text(cx + w / 2, 14.6, sub.replace("|", "\n"), color="#B8C2D1", fontsize=7.5, ha="center",
                va="center", linespacing=1.3)
    ax.text(122, 3.2, "Donate-a-Cry · 5 folds × 3 seeds · numbers from results/paper_numbers.json",
            color=GREY, fontsize=7.5, ha="right")
    _save(fig, path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- badges
def _text_w(s: str) -> float:
    return 6.2 * len(s) + 10  # Verdana 11px approximation (shields.io style)


def badge(label: str, value: str, colour: str, path: Path) -> None:
    lw, vw = _text_w(label), _text_w(value)
    w = lw + vw
    esc = lambda s: s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")  # noqa: E731
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="20" role="img" aria-label="{esc(label)}: {esc(value)}">
<title>{esc(label)}: {esc(value)}</title>
<linearGradient id="s" x2="0" y2="100%"><stop offset="0" stop-color="#bbb" stop-opacity=".1"/><stop offset="1" stop-opacity=".1"/></linearGradient>
<clipPath id="r"><rect width="{w:.0f}" height="20" rx="3" fill="#fff"/></clipPath>
<g clip-path="url(#r)"><rect width="{lw:.0f}" height="20" fill="#555"/><rect x="{lw:.0f}" width="{vw:.0f}" height="20" fill="{colour}"/><rect width="{w:.0f}" height="20" fill="url(#s)"/></g>
<g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">
<text x="{lw / 2:.1f}" y="15" fill="#010101" fill-opacity=".3">{esc(label)}</text><text x="{lw / 2:.1f}" y="14">{esc(label)}</text>
<text x="{lw + vw / 2:.1f}" y="15" fill="#010101" fill-opacity=".3">{esc(value)}</text><text x="{lw + vw / 2:.1f}" y="14">{esc(value)}</text>
</g></svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def badges(n: dict, out: Path) -> list[str]:
    folds = n.get("T2.n_folds", "5")
    specs = [
        ("license", "license", "Apache 2.0", "#007ec6"),
        ("framework", "framework", "PyTorch", "#ee4c2c"),
        ("backbone", "backbone", "AST (AudioSet)", "#6f42c1"),
        ("protocol", "protocol", f"infant-grouped {folds}-fold", "#0e7c86"),
        ("macro_f1", "cry-reason macro-F1", n["T3.ours_crystal.macro_f1"].split(" ±")[0], "#dfb317"),
        ("detection_f1", "cry detection F1", n["T6.ast_detector.f1"].split(" ±")[0], "#44cc11"),
        ("paper", "paper", "preprint in preparation", "#9f9f9f"),
    ]
    names = []
    for fname, label, value, colour in specs:
        badge(label, value, colour, out / f"{fname}.svg")
        names.append(fname)
    return names


# --------------------------------------------------------------------------- charts
def model_comparison(n: dict, raw: dict, path: Path) -> None:
    plt = _plt()
    rows = [r for r in load_project()["paper_rows"]["T3"]]
    from src.report import slugify

    data = []
    for r in rows:
        m, k = mean_of(raw, f"T3.{slugify(r)}.macro_f1"), f"T3.{slugify(r)}.macro_f1"
        sd = raw.get(k, {}).get("std") if isinstance(raw.get(k), dict) else None
        if m is not None and sd is not None:
            data.append(("Ours" if r.startswith("Ours") else r, m, sd))
    data.sort(key=lambda t: t[1])
    maj = mean_of(raw, "T3.majority_class.macro_f1")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colours = [AMBER if d[0] == "Ours" else (GREY if d[0] in ("Majority class", "DeepInfant (reproduced)") else TEAL)
               for d in data]
    ax.barh([d[0] for d in data], [d[1] for d in data], xerr=[d[2] for d in data], color=colours,
            error_kw={"lw": 0.7, "ecolor": "#555"}, height=0.65)
    if maj is not None:
        ax.axvline(maj, color="#C0392B", ls="--", lw=1)
        ax.text(maj, len(data) - 0.4, "  always 'hungry'", color="#C0392B", fontsize=7.5, va="bottom")
    ax.set_xlabel("Test macro-F1 (mean ± s.d., infant-grouped folds × seeds)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.set_xlim(0, max(d[1] + d[2] for d in data) * 1.15)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Every model is close to the majority-class baseline", fontsize=9.5)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def leakage(n: dict, raw: dict, path: Path) -> None:
    plt = _plt()
    pairs = [("DeepInfant\n(reproduced)", "T3.deepinfant_reproduced.macro_f1", "T4.deepinfant_random_clip_level_split.macro_f1"),
             ("Ours", "T3.ours_crystal.macro_f1", "T4.ours_random_clip_level_split.macro_f1")]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(len(pairs))
    for j, (lab, col, off) in enumerate((("Infant-grouped split", TEAL, -0.18), ("Random clip-level split", AMBER, 0.18))):
        vals = [raw.get(p[1 + j]) for p in pairs]
        ax.bar(x + off, [v["mean"] for v in vals], 0.34, yerr=[v["std"] for v in vals], color=col, label=lab,
               error_kw={"lw": 0.7})
    d = n["T4.ours_random_clip_level_split.delta_vs_grouped"]
    top = raw["T4.ours_random_clip_level_split.macro_f1"]["mean"]
    ax.annotate(f"{d} from leakage", xy=(1.18, top), xytext=(0.35, top + 0.06), fontsize=8,
                arrowprops={"arrowstyle": "->", "lw": 0.8})
    ax.set_xticks(x, [p[0] for p in pairs], fontsize=8)
    ax.set_ylabel("Test macro-F1", fontsize=8)
    ax.set_ylim(0, top + 0.12)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    share = n.get("leak.donateacry.random_split_test_share_group_with_train")
    ax.set_title(f"Random splits put the same baby in train and test ({share}% of test clips)", fontsize=8.5)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def main() -> None:
    n, raw = load_numbers()
    banner(n, ASSETS / "banner.png")
    names = badges(n, ASSETS / "badges")
    model_comparison(n, raw, ASSETS / "model_comparison.png")
    leakage(n, raw, ASSETS / "leakage.png")
    print("wrote assets/banner.png, model_comparison.png, leakage.png and badges:", ", ".join(names))


if __name__ == "__main__":
    main()
