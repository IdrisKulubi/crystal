"""Paper figures that are not risk–coverage / reliability (those live in src/selective.py).

Called by ``python -m src.report`` (RESULTS_SPEC §13):
* fig1_pipeline         static pipeline diagram (no numbers)
* fig4_confusion_ours   row-normalised confusion matrix of the final model, pooled out-of-fold test
                        predictions of seed 0 (summary.json -> pooled.confusion_matrix)
* figA_per_class_recall per-class recall, mean ± s.d. over folds × seeds (summary.json ->
                        per_class_recall_mean_std), ours vs DeepInfant reproduced
Colour-blind-safe palette (Okabe–Ito), 200 dpi PNG + vector PDF.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, out_stem: Path) -> None:
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    # no creator/date metadata: files are reproducible byte-for-byte and carry nothing identifying
    meta = {"png": {"Software": None}, "pdf": {"Creator": None, "Producer": None, "CreationDate": None}}
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=200, metadata=meta[ext])


def display_name(row: str) -> str:
    """'Ours (NAME)' -> 'Ours' so figures never carry the model name (double-blind versions reuse them)."""
    return "Ours" if row.startswith("Ours") else row


def plot_pipeline(out_stem: Path) -> None:
    plt = _plt()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.0, 1.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(4, 28)
    ax.axis("off")
    boxes = [
        (1, "Household audio\n16 kHz mono", "#FFFFFF"),
        (18, "Cry detector\n(1 s frames)", OKABE_ITO[4]),
        (35, "Cry segment\n(7 s clip)", "#FFFFFF"),
        (52, "AST encoder\n+ attention pooling", OKABE_ITO[0]),
        (69, "Classifier\n(+ age, sex)\ntemperature-scaled", OKABE_ITO[2]),
    ]
    w, h, y = 14.5, 13, 12
    for x, text, colour in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=colour, ec="black", lw=0.8, alpha=0.9 if colour != "#FFFFFF" else 1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7,
                color="white" if colour in (OKABE_ITO[0], OKABE_ITO[2]) else "black")
    for (x1, *_), (x2, *_) in zip(boxes, boxes[1:]):
        ax.add_patch(FancyArrowPatch((x1 + w + 0.6, y + h / 2), (x2 - 0.6, y + h / 2),
                                     arrowstyle="-|>", mutation_scale=9, lw=0.9, color="black"))
    # decision
    xd = 88
    ax.add_patch(FancyArrowPatch((69 + w + 0.6, y + h / 2), (xd - 0.4, y + h - 1), arrowstyle="-|>",
                                 mutation_scale=9, lw=0.9))
    ax.add_patch(FancyArrowPatch((69 + w + 0.6, y + h / 2), (xd - 0.4, y + 1), arrowstyle="-|>",
                                 mutation_scale=9, lw=0.9))
    ax.text(xd, y + h - 1, r"max p $\geq$ $\tau$: reason", fontsize=7, va="center")
    ax.text(xd, y + 1, r"max p $<$ $\tau$: 'uncertain'", fontsize=7, va="center")
    ax.text(18 + w / 2, y - 4.5, "no cry: nothing is classified", fontsize=6.5, ha="center", style="italic")
    fig.tight_layout(pad=0.2)
    _save(fig, out_stem)
    plt.close(fig)


def plot_confusion(labels: list[str], cm, title: str, out_stem: Path) -> None:
    plt = _plt()
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=6,
                    color="white" if norm[i, j] > 0.55 else "black")
    names = [l.replace("_", " ") for l in labels]
    ax.set_xticks(range(len(labels)), names, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)), names, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=7)
    ax.set_ylabel("True", fontsize=7)
    ax.set_title(title, fontsize=7.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)
    fig.tight_layout()
    _save(fig, out_stem)
    plt.close(fig)


def plot_per_class_recall(series: list[tuple[str, dict]], out_stem: Path) -> None:
    """series: [(name, {label: {"mean", "std"}})] with the same labels."""
    plt = _plt()
    labels = list(series[0][1])
    x = np.arange(len(labels))
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for k, (name, rec) in enumerate(series):
        m = [rec[l]["mean"] for l in labels]
        s = [rec[l]["std"] for l in labels]
        ax.bar(x + (k - (len(series) - 1) / 2) * width, m, width, yerr=s, capsize=2, label=name,
               color=OKABE_ITO[k % len(OKABE_ITO)], error_kw={"lw": 0.7})
    ax.set_xticks(x, [l.replace("_", " ") for l in labels], fontsize=7)
    ax.set_ylabel("Recall", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    _save(fig, out_stem)
    plt.close(fig)
