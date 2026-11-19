"""Generate notebooks/kaggle_train.ipynb and notebooks/kaggle_dapt.ipynb (thin wrappers around the repo's CLI)."""
import json
from pathlib import Path

CELLS = [
    ("markdown", """# CRYSTAL — run experiments on a Kaggle GPU

Settings (right panel): **Accelerator = GPU T4 x1** (or P100), **Internet = On**.
Then *Run all*. When it finishes, download `crystal_results.zip` from the Output tab and run
locally: `python -m src.import_runs crystal_results.zip` (then `python -m src.report`).

Edit `CONFIGS` below to choose experiments. No audio leaves Kaggle; the zip holds only logits,
predictions, metrics and logs."""),
    ("code", """CONFIGS = [                                             # each runs 5 folds x 3 seeds
    "configs/exp/E36_crystal.yaml",                     # final model first: the ablations name it as parent (~1 h each on a T4)
    "configs/exp/E36-noMixup.yaml",                     # T4 ablations: E36 minus one component
    "configs/exp/E36-noNoise.yaml",
    "configs/exp/E36-noSpecAug.yaml",
    "configs/exp/E36-noContext.yaml",
    "configs/exp/E36-noFocal.yaml",
]
EXTRA_ARGS = ""                                         # e.g. "--fold 0 --seeds 0" for a quick check
REPO = "https://github.com/IdrisKulubi/crystal.git"
BRANCH = "main\""""),
    ("code", """import os, subprocess, sys, json, hashlib, pathlib
# work in /tmp (NOT /kaggle/working): Kaggle saves /kaggle/working as notebook output, and we
# never want the audio in data/ to end up there. Only crystal_results.zip is written to output.
os.chdir("/tmp")
if not pathlib.Path("crystal").exists():
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO, "crystal"], check=True)
os.chdir("/tmp/crystal")
subprocess.run(["git", "log", "--oneline", "-1"], check=True)
# Kaggle ships torchao 0.10, which recent peft refuses to import alongside (breaks LoRA); we don't use it
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "librosa", "soundfile", "pyyaml", "scikit-learn", "pandas", "transformers>=4.44", "peft>=0.12"], check=True)
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
assert torch.cuda.is_available() or "--debug" in EXTRA_ARGS, (
    "No GPU. Stop the session, set Settings -> Accelerator -> GPU T4 x2 / P100 (needs a phone-verified "
    "account), then run again. Training on CPU here would take days.")"""),
    ("code", """# data: same public sources and the same pipeline as locally
os.makedirs("data/raw", exist_ok=True); os.makedirs("baselines", exist_ok=True)
if not pathlib.Path("data/raw/donateacry-corpus").exists():
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/gveres/donateacry-corpus", "data/raw/donateacry-corpus"], check=True)
if not pathlib.Path("baselines/DeepInfant").exists():
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/skytells-research/DeepInfant", "baselines/DeepInfant"], check=True)
if not pathlib.Path("data/raw/ESC-50").exists():   # noise augmentation pool (public, CC BY-NC), training only
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/karolpiczak/ESC-50", "data/raw/ESC-50"], check=True)
expected = json.load(open("results/splits_report.json"))["folds_file_sha256"]
for step in ["prepare_data", "dedup", "make_splits"]:
    subprocess.run([sys.executable, "-m", f"src.{step}"], check=True)
got = hashlib.sha256(open("data/metadata_folds.csv", "rb").read()).hexdigest()
assert got == expected, f"folds file differs from the committed one: {got} != {expected}"
print("folds file identical to the committed protocol:", got[:16])
subprocess.run(["git", "checkout", "--", "results/"], check=True)  # keep committed results untouched"""),
    ("code", """reg = pathlib.Path("results/registry.jsonl")
n_before = len(reg.read_text().splitlines()) if reg.exists() else 0
for cfg in CONFIGS:
    cmd = [sys.executable, "-m", "src.train", "--config", cfg] + EXTRA_ARGS.split()
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=False)   # failures are logged in the registry, keep going"""),
    ("code", """import zipfile
lines = reg.read_text().splitlines()[n_before:]
out = pathlib.Path("/kaggle/working/crystal_results.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("registry_new.jsonl", "\\n".join(lines) + "\\n")
    for line in lines:
        rid = json.loads(line)["run_id"]
        for f in pathlib.Path("results/runs", rid).rglob("*"):
            if f.is_file():
                z.write(f, pathlib.Path("runs", rid, f.relative_to(pathlib.Path("results/runs", rid))))
print(out, round(out.stat().st_size / 1e6, 1), "MB;", len(lines), "registry lines")
for line in lines: r = json.loads(line); print(r["run_id"], r["status"], r.get("macro_f1_mean"))"""),
]

# Phase 4: DAPT on CryCeleb, then E20/E21 with the new checkpoint, in ONE session (checkpoint stays on disk).
DAPT_CELLS = [
    ("markdown", """# CRYSTAL — Phase 4: domain-adaptive pretraining (DAPT) on a Kaggle GPU

Settings: **Accelerator = GPU T4**, **Internet = On**, and **Add-ons -> Secrets -> `HF_TOKEN`** = a
Hugging Face *read* token of the account that accepted the CryCeleb2023 terms (attach it to this
notebook). Keep this notebook **private** (CryCeleb is CC BY-NC-ND; the checkpoint must not be shared).
Then **Save Version -> Save & Run All**. Download `crystal_results.zip` (logs/metrics, no audio) from the
Output tab; `dapt_checkpoint.zip` (the pretrained model, ~0.4 GB) is kept there too for later phases."""),
    ("code", """DAPT_CONFIG = "configs/dapt/D01_w2v2_cryceleb.yaml"
DAPT_MAX_STEPS = None                                   # None = config value (10k); e.g. 500 for a smoke test
CONFIGS = [                                             # run after DAPT with init_checkpoint dapt:D01
    "configs/exp/E20_dapt_w2v2_probe.yaml",
    "configs/exp/E21_dapt_w2v2_ft.yaml",
]
EXTRA_ARGS = ""
SAVE_CHECKPOINT = True
REPO = "https://github.com/IdrisKulubi/crystal.git"
BRANCH = "main\""""),
    CELLS[2],
    CELLS[3],
    ("code", """# CryCeleb (gated): token from Kaggle Secrets, audio stays in /tmp
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
from huggingface_hub import snapshot_download
snapshot_download("Ubenwa/CryCeleb2023", repo_type="dataset", local_dir="data/raw/cryceleb",
                  allow_patterns=["audio.zip", "metadata.csv", "README.md"])
expected_h = json.load(open("results/dapt_stats.json"))["hours_unlabelled"]
subprocess.run([sys.executable, "-m", "src.build_dapt_corpus"], check=True)
got_h = json.load(open("results/dapt_stats.json"))["hours_unlabelled"]
assert got_h == expected_h, f"DAPT corpus differs from the committed one: {got_h} h != {expected_h} h"
subprocess.run(["git", "checkout", "--", "results/"], check=True)"""),
    ("code", """cmd = [sys.executable, "-m", "src.dapt", "--config", DAPT_CONFIG]
if DAPT_MAX_STEPS: cmd += ["--max-steps", str(DAPT_MAX_STEPS)]
print(" ".join(cmd), flush=True)
subprocess.run(cmd, check=False)
dapt_runs = sorted(p.name for p in pathlib.Path("results/dapt").glob("*") if p.is_dir())
print("DAPT runs:", dapt_runs)
for d in dapt_runs: print(d, json.load(open(f"results/dapt/{d}/status.json")))"""),
    CELLS[4],
    ("code", """import zipfile
lines = reg.read_text().splitlines()[n_before:]
out = pathlib.Path("/kaggle/working/crystal_results.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("registry_new.jsonl", "\\n".join(lines) + ("\\n" if lines else ""))
    for line in lines:
        rid = json.loads(line)["run_id"]
        for f in pathlib.Path("results/runs", rid).rglob("*"):
            if f.is_file():
                z.write(f, pathlib.Path("runs", rid, f.relative_to(pathlib.Path("results/runs", rid))))
    for d in dapt_runs:
        for f in pathlib.Path("results/dapt", d).rglob("*"):
            if f.is_file():
                z.write(f, pathlib.Path("dapt", d, f.relative_to(pathlib.Path("results/dapt", d))))
print(out, round(out.stat().st_size / 1e6, 1), "MB;", len(lines), "registry lines;", len(dapt_runs), "DAPT runs")
for line in lines: r = json.loads(line); print(r["run_id"], r["status"], r.get("macro_f1_mean"))
if SAVE_CHECKPOINT:
    for d in dapt_runs:
        ck = pathlib.Path("checkpoints", d)
        if ck.is_dir():
            with zipfile.ZipFile("/kaggle/working/dapt_checkpoint.zip", "w", zipfile.ZIP_STORED) as z:
                for f in ck.rglob("*"):
                    z.write(f, pathlib.Path(d, f.relative_to(ck)))
            print("saved /kaggle/working/dapt_checkpoint.zip")"""),
]


# Phase 7: stage-1 cry detector (E40 AST, E41 small CNN).
DETECT_CELLS = [
    ("markdown", """# CRYSTAL — Phase 7: cry detector on a Kaggle GPU

Settings: **Accelerator = GPU T4**, **Internet = On**, **Add-ons -> Secrets -> `HF_TOKEN`** attached (CryCeleb
is one of the positive sources). Keep the notebook private. **Save Version -> Save & Run All**, then download
`crystal_results.zip` (scores/metrics/logs, no audio) from the Output tab."""),
    ("code", """CONFIGS = [                                             # each runs 5 folds x 3 seeds
    "configs/exp/E41_cnn_detector.yaml",
    "configs/exp/E40_ast_detector.yaml",
]
EXTRA_ARGS = ""                                         # e.g. "--fold 0 --seeds 0" for a quick check
REPO = "https://github.com/IdrisKulubi/crystal.git"
BRANCH = "main\""""),
    CELLS[2],
    CELLS[3],
    DAPT_CELLS[4],
    ("code", """# ESC-50 (public, CC BY-NC) -> detection table; folds must equal the committed protocol
if not pathlib.Path("data/raw/ESC-50").exists():
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/karolpiczak/ESC-50", "data/raw/ESC-50"], check=True)
expected = json.load(open("results/detection_splits_report.json"))["folds_file_sha256"]
subprocess.run([sys.executable, "-m", "src.prepare_detection"], check=True)
got = hashlib.sha256(open("data/detection_folds.csv", "rb").read()).hexdigest()
assert got == expected, f"detection folds differ from the committed ones: {got} != {expected}"
print("detection folds identical to the committed protocol:", got[:16])
subprocess.run(["git", "checkout", "--", "results/"], check=True)"""),
    ("code", """reg = pathlib.Path("results/registry.jsonl")
n_before = len(reg.read_text().splitlines()) if reg.exists() else 0
for cfg in CONFIGS:
    cmd = [sys.executable, "-m", "src.detection", "--config", cfg] + EXTRA_ARGS.split()
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=False)   # failures are logged in the registry, keep going"""),
    CELLS[5],
]


def _write(cells, name):
    nb = {
        "cells": [
            {"cell_type": t, "metadata": {}, "source": src.splitlines(keepends=True),
             **({"outputs": [], "execution_count": None} if t == "code" else {})}
            for t, src in cells
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = Path(__file__).resolve().parents[1] / "notebooks" / name
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)


def main():
    _write(CELLS, "kaggle_train.ipynb")
    _write(DAPT_CELLS, "kaggle_dapt.ipynb")
    _write(DETECT_CELLS, "kaggle_detect.ipynb")


if __name__ == "__main__":
    main()
