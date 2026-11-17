"""Generate notebooks/kaggle_train.ipynb (thin wrapper around the repo's own CLI)."""
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
    "configs/exp/E10_ast_ft.yaml",
    "configs/exp/E11_ast_lora.yaml",
    "configs/exp/E12_w2v2_ft.yaml",
    "configs/exp/E13_w2v2_lora.yaml",
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


def main():
    nb = {
        "cells": [
            {"cell_type": t, "metadata": {}, "source": src.splitlines(keepends=True),
             **({"outputs": [], "execution_count": None} if t == "code" else {})}
            for t, src in CELLS
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_train.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
