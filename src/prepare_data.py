"""Raw datasets -> data/processed/<dataset>/<clip>.wav (16 kHz mono) + data/metadata.csv.

    python -m src.prepare_data [--debug]

Donate-a-Cry filename convention (corpus README), e.g.
    0D1AD73E-4C5E-45F3-85C4-9A3CB71E8856-1430742197-1.0-m-04-hu   (iOS, epoch seconds)
    0c8f14a9-6999-485b-97a2-913c1cbf099c-1431028888092-1.7-m-26-sc (Android, epoch ms)
    = app-install UUID - timestamp - app version - gender - age bracket - reason code
The UUID identifies an app installation (not a device or a baby). It is the best available
proxy for "same baby", so it is the split group (HANDOFF §5.2, see docs/DECISIONS.md).
DeepInfant's Data/v2 adds synthetic copies prefixed ``generated<N>_``; they are flagged.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd

from src.audio import TARGET_SR, load_audio, pcm16_sha256, save_audio
from src.config import REPO_ROOT, load_project

FNAME_RE = re.compile(
    r"^(?:(?P<gen>generated\d+)_)?"
    r"(?P<uuid>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
    r"-(?P<ts>\d+)-(?P<ver>[\d.]+)-(?P<gender>[mf])-(?P<age>\d{2})-(?P<reason>[a-z]{2})$"
)
REASON_CODES = {"hu": "hungry", "bu": "burping", "bp": "belly_pain", "dc": "discomfort", "ti": "tired",
                "lo": "lonely", "ch": "cold_hot", "sc": "scared", "dk": "unknown"}
# age code -> (ordinal bracket, description)
AGE_CODES = {"04": (0, "0-4 weeks"), "48": (1, "4-8 weeks"), "26": (2, "2-6 months"),
             "72": (3, "7 months-2 years"), "22": (4, "over 2 years")}

META_COLUMNS = [
    "clip_id", "dataset", "path", "orig_path", "label_orig", "uuid", "group_raw", "timestamp_unix",
    "app_version", "platform", "gender", "age_code", "age_bracket", "age_desc", "reason_code",
    "reason_from_filename", "label_matches_filename", "is_generated", "generated_tag", "orig_sr",
    "orig_channels", "duration_s", "pcm_sha256", "file_sha256",
]


def parse_filename(stem: str) -> dict | None:
    m = FNAME_RE.match(stem)
    if not m:
        return None
    ts = int(m["ts"])
    is_ms = len(m["ts"]) >= 13
    age = AGE_CODES.get(m["age"])
    return {
        "uuid": m["uuid"].lower(),
        "timestamp_unix": ts / 1000.0 if is_ms else float(ts),
        "app_version": m["ver"],
        "platform": "android" if is_ms else "ios",
        "gender": m["gender"],
        "age_code": m["age"],
        "age_bracket": age[0] if age else None,
        "age_desc": age[1] if age else None,
        "reason_code": m["reason"],
        "reason_from_filename": REASON_CODES.get(m["reason"]),
        "is_generated": m["gen"] is not None,
        "generated_tag": m["gen"],
    }


def safe_id(text: str) -> str:
    return re.sub(r"[^\w.-]+", "_", text)


def iter_dataset(ds_id: str, spec: dict, debug: bool = False):
    root = REPO_ROOT / spec["raw"]
    if not root.is_dir():
        print(f"[skip] {ds_id}: {root} not found")
        return
    files = sorted(root.glob(spec.get("glob", "*/*.wav")))
    if debug:
        by_label: dict[str, list[Path]] = {}
        for f in files:
            by_label.setdefault(f.parent.name, []).append(f)
        files = [f for fs in by_label.values() for f in fs[:5]]
    for f in files:
        yield f


def process_file(ds_id: str, f: Path, out_root: Path) -> dict:
    parsed = parse_filename(f.stem)
    label = f.parent.name
    if parsed is None:
        raise ValueError(f"{ds_id}: filename does not match the Donate-a-Cry convention: {f.name}")
    import soundfile as sf

    info = sf.info(str(f))
    wav, _ = load_audio(f, TARGET_SR)
    clip_id = safe_id(f"{ds_id}__{label}__{f.stem}")
    out = out_root / ds_id / f"{clip_id}.wav"
    save_audio(out, wav, TARGET_SR)
    with open(f, "rb") as fh:
        file_sha = hashlib.sha256(fh.read()).hexdigest()
    native, _ = load_audio(f, None)
    return {
        "clip_id": clip_id,
        "dataset": ds_id,
        "path": out.relative_to(REPO_ROOT).as_posix(),
        "orig_path": f.relative_to(REPO_ROOT).as_posix(),
        "label_orig": label,
        "group_raw": f"dac:{parsed['uuid']}",
        **parsed,
        "label_matches_filename": parsed["reason_from_filename"] == label,
        "orig_sr": info.samplerate,
        "orig_channels": info.channels,
        "duration_s": round(len(wav) / TARGET_SR, 4),
        "pcm_sha256": pcm16_sha256(native),
        "file_sha256": file_sha,
    }


def main(argv: list[str] | None = None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description="Build data/processed and data/metadata.csv")
    ap.add_argument("--debug", action="store_true", help="5 clips per class per dataset")
    args = ap.parse_args(argv)
    project = load_project()
    out_root = REPO_ROOT / "data" / "processed"
    rows = []
    for ds_id, spec in project["datasets"].items():
        n0 = len(rows)
        for f in iter_dataset(ds_id, spec, args.debug):
            rows.append(process_file(ds_id, f, out_root))
        print(f"{ds_id}: {len(rows) - n0} clips")
    df = pd.DataFrame(rows)[META_COLUMNS]
    if df["clip_id"].duplicated().any():
        raise AssertionError("duplicate clip_id values")
    mism = df[~df["label_matches_filename"]]
    if len(mism):
        print(f"WARNING: {len(mism)} clips whose folder label != filename reason code")
    out = REPO_ROOT / "data" / "metadata.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out.relative_to(REPO_ROOT)} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    main()
