"""Phase 4 — build the UNLABELLED corpus for domain-adaptive pretraining (DAPT).

    python -m src.build_dapt_corpus [--chunk-seconds 8] [--debug]

Source: CryCeleb2023 (Ubenwa, Hugging Face ``Ubenwa/CryCeleb2023``, gated, CC BY-NC-ND 4.0), downloaded
to ``data/raw/cryceleb/{audio.zip,metadata.csv}``. All of its splits are used: only the audio is
used, never its labels (baby ids / verification pairs), and none of our evaluation data is in it.
Labelled datasets (Donate-a-Cry) are excluded entirely (HANDOFF Phase 4 "safest" option), and the
corpus is checked for exact-PCM overlap with ``data/metadata.csv`` (must be 0).

CryCeleb clips are single cry expirations (median ~0.8 s), too short for wav2vec2 pretraining, so
consecutive segments of the same baby and recording period (in ``chronological_index`` order) are
concatenated into chunks of at most ``chunk_seconds``.

Outputs (git-ignored audio, committed stats):
* data/dapt/chunks/<chunk_id>.wav  16 kHz mono
* data/dapt/manifest.csv           chunk_id, path, n_samples, n_segments, baby_id, period, source
* results/dapt_stats.json          hours_unlabelled (paper key dapt.hours_unlabelled), counts, licence
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile

import numpy as np
import pandas as pd

from src.audio import TARGET_SR, pcm16_sha256, save_audio
from src.config import REPO_ROOT

RAW = REPO_ROOT / "data" / "raw" / "cryceleb"
OUT = REPO_ROOT / "data" / "dapt"
STATS = REPO_ROOT / "results" / "dapt_stats.json"


def _read_wav(z: zipfile.ZipFile, name: str) -> np.ndarray:
    import soundfile as sf

    wav, sr = sf.read(io.BytesIO(z.read(name)), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR, res_type="soxr_hq").astype(np.float32)
    return wav


def pack_chunks(segments: list[np.ndarray], max_samples: int) -> list[list[int]]:
    """Greedy, order-preserving packing of segment indices into chunks of <= max_samples
    (a single segment longer than max_samples forms its own chunk and is cropped later)."""
    chunks, cur, cur_len = [], [], 0
    for i, s in enumerate(segments):
        if cur and cur_len + len(s) > max_samples:
            chunks.append(cur)
            cur, cur_len = [], 0
        cur.append(i)
        cur_len += len(s)
    if cur:
        chunks.append(cur)
    return chunks


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-seconds", type=float, default=8.0)
    ap.add_argument("--debug", action="store_true", help="first 20 babies only; nothing written to results/")
    args = ap.parse_args(argv)

    meta = pd.read_csv(RAW / "metadata.csv", dtype={"baby_id": str, "chronological_index": str})
    meta = meta.sort_values(["baby_id", "period", "chronological_index", "file_id"]).reset_index(drop=True)
    if args.debug:
        meta = meta[meta["baby_id"].isin(sorted(meta["baby_id"].unique())[:20])]
    labelled_sha = set()
    lab_path = REPO_ROOT / "data" / "metadata.csv"
    if lab_path.is_file():
        labelled_sha = set(pd.read_csv(lab_path, usecols=["pcm_sha256"])["pcm_sha256"])

    (OUT / "chunks").mkdir(parents=True, exist_ok=True)
    max_samples = int(args.chunk_seconds * TARGET_SR)
    rows, n_seg, n_overlap, seen, n_dup, total = [], 0, 0, set(), 0, 0
    with zipfile.ZipFile(RAW / "audio.zip") as z:
        for (baby, period), g in meta.groupby(["baby_id", "period"], sort=True):
            segs = []
            for name in g["file_name"]:
                w = _read_wav(z, name)
                h = pcm16_sha256(w)
                n_overlap += h in labelled_sha
                if h in seen or h in labelled_sha:      # drop exact duplicates and any labelled-set clip
                    n_dup += h in seen
                    continue
                seen.add(h)
                segs.append(w)
            for k, idx in enumerate(pack_chunks(segs, max_samples)):
                wav = np.concatenate([segs[i] for i in idx])[:max_samples]
                cid = f"cryceleb__{baby}_{period}_{k:03d}"
                path = OUT / "chunks" / f"{cid}.wav"
                save_audio(path, wav, TARGET_SR)
                rows.append({"chunk_id": cid, "path": path.relative_to(REPO_ROOT).as_posix(),
                             "n_samples": len(wav), "n_segments": len(idx), "baby_id": baby,
                             "period": period, "source": "cryceleb"})
                total += len(wav)
                n_seg += len(idx)
            if len(rows) and len(rows) % 500 < 3:
                print(f"  {len(rows)} chunks", flush=True)
    man = pd.DataFrame(rows)
    man.to_csv(OUT / "manifest.csv", index=False, lineterminator="\n")
    stats = {
        "sources": {"cryceleb": {"hf_id": "Ubenwa/CryCeleb2023", "licence": "CC BY-NC-ND 4.0",
                                 "used": "audio only, all splits; labels unused"}},
        "hours_unlabelled": round(total / TARGET_SR / 3600, 4),
        "n_segments_used": int(n_seg), "n_segments_in_metadata": int(len(meta)),
        "n_exact_duplicates_dropped": int(n_dup), "n_overlap_with_labelled": int(n_overlap),
        "n_chunks": int(len(man)), "n_babies": int(man["baby_id"].nunique()) if len(man) else 0,
        "chunk_seconds_max": args.chunk_seconds,
        "chunk_seconds_median": round(float(man["n_samples"].median()) / TARGET_SR, 3) if len(man) else None,
    }
    if not args.debug:
        STATS.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    main()
