"""Exact + near-duplicate detection across all datasets, BEFORE splitting (CLAUDE.md rule 1).

    python -m src.dedup [--threshold 0.90]

* exact: identical decoded PCM (``pcm_sha256`` from prepare_data, native sample rate).
* near:  cosine similarity of time-aligned log-mel spectrograms (band-limited to 0-4 kHz,
         because Donate-a-Cry is 8 kHz audio), maximised over small time shifts
         (±``max_shift_s``) so re-encoded / re-cropped copies are still caught.
         Plain mean-pooled embeddings were rejected: different cries from the same baby
         are too similar on average (docs/DECISIONS.md).

Outputs
* data/dedup_report.csv  one row per duplicate pair (kind, similarity, labels)
* data/metadata.csv      updated in place with: dup_cluster, is_duplicate, duplicate_of, group
  - ``group``: split group after merging the groups of all clips in a duplicate cluster
  - ``is_duplicate``: True if another clip of the SAME dataset in the cluster is kept instead
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.audio import TARGET_SR, load_audio
from src.config import REPO_ROOT, load_project

FEAT_HOP = 640            # 40 ms at 16 kHz
FEAT_MELS = 32
FEAT_FMAX = 4000
FEAT_SECONDS = 7.0


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


def logmel_features(wav: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """[frames, mels] log-mel, fixed length, per-clip standardised."""
    import librosa

    n = int(FEAT_SECONDS * sr)
    wav = np.pad(wav, (0, max(0, n - len(wav))))[:n]
    m = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=1024, hop_length=FEAT_HOP,
                                       n_mels=FEAT_MELS, fmax=FEAT_FMAX)
    x = np.log(m + 1e-6).T.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-6)


def shifted_cosine(feats: np.ndarray, max_shift: int, block: int = 256) -> np.ndarray:
    """Max over shifts s in [-max_shift, max_shift] of the cosine between clip i and clip j
    shifted by s frames (overlapping region only). feats: [n, T, M] -> [n, n]."""
    n, T, M = feats.shape
    best = np.full((n, n), -1.0, dtype=np.float32)
    for s in range(-max_shift, max_shift + 1):
        a = feats[:, max(0, s):T + min(0, s), :].reshape(n, -1)
        b = feats[:, max(0, -s):T + min(0, -s), :].reshape(n, -1)
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        for i in range(0, n, block):
            sim = a[i:i + block] @ b.T
            np.maximum(best[i:i + block], sim, out=best[i:i + block])
    best = np.maximum(best, best.T)  # symmetric: shift +s for (i,j) == shift -s for (j,i)
    np.fill_diagonal(best, 1.0)
    return best


def find_duplicates(meta: pd.DataFrame, feats: np.ndarray | None, threshold: float,
                    max_shift: int) -> pd.DataFrame:
    """Return pair table: clip_a, clip_b, kind, similarity."""
    ids = meta["clip_id"].tolist()
    pairs = []
    exact_pairs = set()
    for _, g in meta.groupby("pcm_sha256"):
        members = sorted(g["clip_id"])
        for b in members[1:]:
            pairs.append((members[0], b, "exact", 1.0))
            exact_pairs.add((members[0], b))
    if feats is not None and len(ids) > 1:
        sim = shifted_cosine(feats, max_shift)
        iu, ju = np.where(np.triu(sim >= threshold, k=1))
        for i, j in zip(iu, ju):
            a, b = sorted((ids[i], ids[j]))
            if (a, b) not in exact_pairs and meta["pcm_sha256"].iat[i] != meta["pcm_sha256"].iat[j]:
                pairs.append((a, b, "near", float(sim[i, j])))
    return pd.DataFrame(pairs, columns=["clip_a", "clip_b", "kind", "similarity"])


def resolve(meta: pd.DataFrame, pairs: pd.DataFrame, eval_datasets: list[str]) -> pd.DataFrame:
    """Add dup_cluster / is_duplicate / duplicate_of / group to ``meta``."""
    meta = meta.copy()
    uf = UnionFind(meta["clip_id"])
    for a, b in zip(pairs["clip_a"], pairs["clip_b"]):
        uf.union(a, b)
    meta["dup_cluster"] = meta["clip_id"].map(uf.find)

    # merge split groups across each duplicate cluster
    guf = UnionFind(meta["group_raw"].unique())
    for _, g in meta.groupby("dup_cluster"):
        gs = g["group_raw"].unique()
        for other in gs[1:]:
            guf.union(gs[0], other)
    meta["group"] = meta["group_raw"].map(guf.find)

    # canonical clip per (cluster, dataset): original (not generated) first, then clip_id
    meta["is_duplicate"] = False
    meta["duplicate_of"] = None
    for (_, _), g in meta.groupby(["dup_cluster", "dataset"]):
        if len(g) < 2:
            continue
        g = g.sort_values(["is_generated", "clip_id"])
        keep = g["clip_id"].iat[0]
        drop = g.index[1:]
        meta.loc[drop, "is_duplicate"] = True
        meta.loc[drop, "duplicate_of"] = keep
    return meta


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--max-shift-s", type=float, default=0.5)
    ap.add_argument("--recompute", action="store_true", help="ignore the feature cache")
    args = ap.parse_args(argv)
    project = load_project()
    meta_path = REPO_ROOT / "data" / "metadata.csv"
    meta = pd.read_csv(meta_path)
    meta = meta.drop(columns=[c for c in ("dup_cluster", "is_duplicate", "duplicate_of", "group")
                              if c in meta], errors="ignore")

    cache = REPO_ROOT / "data" / "processed" / "dedup_feats.npz"
    params = np.array([FEAT_HOP, FEAT_MELS, FEAT_FMAX, FEAT_SECONDS], dtype=np.float64)
    cached = {}
    if cache.is_file() and not args.recompute:
        z = np.load(cache, allow_pickle=False)
        if "params" in z and np.array_equal(z["params"], params):
            cached = dict(zip(z["ids"].tolist(), z["feats"]))
    feats = []
    for cid, p in zip(meta["clip_id"], meta["path"]):
        if cid not in cached:
            wav, _ = load_audio(REPO_ROOT / p, TARGET_SR)
            cached[cid] = logmel_features(wav)
        feats.append(cached[cid])
    feats = np.stack(feats)
    np.savez(cache, ids=np.array(meta["clip_id"], dtype=str), feats=feats, params=params)

    max_shift = int(round(args.max_shift_s * TARGET_SR / FEAT_HOP))
    pairs = find_duplicates(meta, feats, args.threshold, max_shift)
    lab = meta.set_index("clip_id")
    for side in ("a", "b"):
        pairs[f"dataset_{side}"] = pairs[f"clip_{side}"].map(lab["dataset"])
        pairs[f"label_{side}"] = pairs[f"clip_{side}"].map(lab["label_orig"])
        pairs[f"group_{side}"] = pairs[f"clip_{side}"].map(lab["group_raw"])
    pairs["same_label"] = pairs["label_a"] == pairs["label_b"]
    pairs["same_group"] = pairs["group_a"] == pairs["group_b"]
    pairs.to_csv(REPO_ROOT / "data" / "dedup_report.csv", index=False)

    meta = resolve(meta, pairs, project.get("eval_datasets", []))
    meta.to_csv(meta_path, index=False)

    print(f"pairs: {len(pairs)} ({(pairs.kind == 'exact').sum()} exact, {(pairs.kind == 'near').sum()} near)")
    within = pairs[pairs.dataset_a == pairs.dataset_b]
    print("within-dataset pairs by dataset/kind:")
    print(within.groupby(["dataset_a", "kind"]).size().to_string())
    print("duplicates removed per dataset:", meta.groupby("dataset")["is_duplicate"].sum().to_dict())
    merged = meta.groupby("dataset").apply(lambda g: g["group_raw"].nunique() - g["group"].nunique(),
                                           include_groups=False)
    print("groups merged per dataset:", merged.to_dict())


if __name__ == "__main__":
    main()
