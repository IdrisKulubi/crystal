"""Fold tables and waveform loading shared by all models."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from src.audio import TARGET_SR, load_audio
from src.config import REPO_ROOT, get_path, load_project


@dataclass
class FoldData:
    """One (fold, label set) view of the benchmark."""
    labels: list[str]
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    extra: dict = field(default_factory=dict)


def load_table(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Folds table restricted to cfg datasets + label set, with integer ``label_idx``."""
    project = load_project()
    ls = get_path(cfg, "data.label_set")
    labels = list(project["label_sets"][ls])
    path = REPO_ROOT / get_path(cfg, "data.metadata")
    df = pd.read_csv(path)
    datasets = get_path(cfg, "data.datasets") or sorted(df["dataset"].unique())
    df = df[df["dataset"].isin(datasets)].copy()
    col = f"label_{ls}"
    df = df[df[col].isin(labels)].copy()
    df["label"] = df[col]
    df["label_idx"] = df["label"].map({lab: i for i, lab in enumerate(labels)}).astype(int)
    return df.reset_index(drop=True), labels


def fold_data(df: pd.DataFrame, labels: list[str], fold: int, debug: bool = False,
              debug_per_class: int = 5, seed: int = 0) -> FoldData:
    col = f"split_f{fold}"
    parts = {role: df[df[col] == role].reset_index(drop=True) for role in ("train", "val", "test")}
    if debug:
        parts = {role: (p.groupby("label", group_keys=False)
                        .apply(lambda g: g.sample(min(len(g), debug_per_class), random_state=seed))
                        .reset_index(drop=True)) for role, p in parts.items()}
    return FoldData(labels=labels, **parts)


@lru_cache(maxsize=4096)
def _load(path: str, sr: int) -> np.ndarray:
    wav, _ = load_audio(REPO_ROOT / path, sr)
    return wav


def load_waveforms(paths, sr: int = TARGET_SR) -> list[np.ndarray]:
    return [_load(p, sr) for p in paths]


def pad_or_crop(wav: np.ndarray, n: int) -> np.ndarray:
    """Right-pad with zeros or keep the first n samples (DeepInfant convention)."""
    if len(wav) >= n:
        return wav[:n]
    return np.pad(wav, (0, n - len(wav)))
