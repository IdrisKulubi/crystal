"""Phase 9 — efficiency numbers on ONE machine, so the models are comparable.

    python -m src.efficiency                 # -> results/efficiency.json (then python -m src.report)
    python -m src.efficiency --models ours,deepinfant --n 20

Per-run ``inference_ms_per_clip_cpu`` in metrics.json was measured on whatever machine trained the
run (mostly Kaggle CPUs), so it cannot be compared across rows. This benchmark rebuilds each paper
model's architecture locally (trained weights do not change cost or size) and measures, per model:

* params_total / params_trainable (trainable = what the paper run actually trained)
* size_mb_fp32: serialized fp32 state_dict (what a deployed checkpoint weighs)
* ms_per_clip_cpu: median wall-clock ms per input (reason models: one 7 s clip; detectors: one 1 s
  frame), batch 1, raw waveform -> logits *including feature extraction*, after 5 warm-up calls,
  over ``n`` real test clips of fold 0 (RESULTS_SPEC §5 protocol). Measured with all CPU threads and
  with 1 thread (a rough phone-like proxy). p90 is stored too.

No audio leaves the machine; only timings and sizes are written.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from src.config import REPO_ROOT, load_project
from src.results.run import DEFAULT_RESULTS_DIR

# key -> (paper table, paper row, source experiment)
MODELS = {
    "deepinfant": ("T3", "DeepInfant (reproduced)", "E02"),
    "ast_probe": ("T3", "AST, linear probe", "E07"),
    "w2v2_ft": ("T3", "wav2vec2-base, fine-tuned", "E12"),
    "ours": ("T3", "Ours (CRYSTAL)", "E36"),
    "ast_detector": ("T6", "AST detector", "E40"),
    "cnn_detector": ("T6", "Small CNN detector", "E41"),
}


def size_mb(module) -> float:
    import torch

    buf = io.BytesIO()
    torch.save({k: v.float() if v.is_floating_point() else v for k, v in module.state_dict().items()}, buf)
    return buf.getbuffer().nbytes / 1e6


def count(module) -> tuple[int, int]:
    return (sum(p.numel() for p in module.parameters()),
            sum(p.numel() for p in module.parameters() if p.requires_grad))


def build(key: str, n_classes: int):
    """-> (module for params/size, predict(wav) -> np.ndarray, params_trainable override or None)."""
    import torch

    if key == "deepinfant":
        from src.models.deepinfant_repro import DeepInfantRepro, deepinfant_mel

        m = DeepInfantRepro(n_classes).eval()

        @torch.no_grad()
        def predict(wav):
            return m(torch.from_numpy(deepinfant_mel(wav)[None, None].astype(np.float32))).numpy()
        return m, predict, None

    if key in ("ours", "w2v2_ft"):
        from transformers import AutoFeatureExtractor

        from src.models.encoders import ENCODERS
        from src.models.finetune import FineTuneModel, model_inputs

        backbone, head, ctx_dim = ("ast", "attention", 7) if key == "ours" else ("wav2vec2_base", "mean_pool", 0)
        m = FineTuneModel(backbone, n_classes, head, None, None, context_dim=ctx_dim).float().eval()
        fe = AutoFeatureExtractor.from_pretrained(ENCODERS[backbone]["hf_id"])
        kind, n = ENCODERS[backbone]["kind"], 7 * 16000
        c1 = torch.zeros(1, ctx_dim) if ctx_dim else None

        @torch.no_grad()
        def predict(wav):
            return m(torch.from_numpy(model_inputs(fe, kind, [wav], n)), c1).numpy()
        return m, predict, None

    if key == "ast_probe":
        from src.models.probe import get_encoder

        enc = get_encoder("ast", "cpu")
        w = np.random.default_rng(0).standard_normal((enc.model.config.hidden_size, n_classes)).astype(np.float32)

        def predict(wav):  # frozen encoder + standardise + logistic regression (one matmul)
            e = enc.layer_embeddings(wav)[-1]
            return ((e - e.mean()) / (e.std() + 1e-8)) @ w
        return enc.model, predict, int(w.size + n_classes)

    if key in ("ast_detector", "cnn_detector"):
        from src.detection import band_limit, features, make_model

        kind = key
        m = make_model(kind).float().eval()

        @torch.no_grad()
        def predict(wav):
            frame = band_limit(np.asarray(wav, np.float32))[:16000]
            frame = np.pad(frame, (0, 16000 - len(frame)))
            return m(torch.from_numpy(features(kind, frame[None]))).numpy()
        return m, predict, None
    raise ValueError(f"unknown model {key!r}")


def time_ms(predict, wavs, n: int, warmup: int = 5) -> dict:
    for i in range(warmup):
        predict(wavs[i % len(wavs)])
    t = []
    for i in range(n):
        t0 = time.perf_counter()
        predict(wavs[i % len(wavs)])
        t.append((time.perf_counter() - t0) * 1000)
    return {"median": float(np.median(t)), "p90": float(np.percentile(t, 90)), "n": n}


def test_clips(n: int) -> list[np.ndarray]:
    import pandas as pd

    from src.datasets import load_waveforms

    df = pd.read_csv(REPO_ROOT / "data" / "metadata_folds.csv")
    df = df[(df["dataset"] == "donateacry") & (df["split_f0"] == "test")]
    return load_waveforms(list(df["path"][:n]))


def hardware() -> dict:
    import torch

    return {"machine": platform.machine(), "processor": platform.processor(), "system": platform.platform(),
            "cpu_count_logical": os.cpu_count(), "torch": torch.__version__,
            "torch_threads_default": torch.get_num_threads(), "python": platform.python_version()}


def run(keys: list[str], n: int, results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    import torch

    labels = load_project()["label_sets"]["donateacry_v1"]
    wavs = test_clips(n)
    default_threads = torch.get_num_threads()
    out = {"protocol": "median ms per input, batch 1, CPU, raw waveform -> logits incl. features, "
                       "5 warm-up calls; reason models: 7 s clip; detectors: 1 s frame",
           "hardware": hardware(), "n_clips": len(wavs), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "models": {}}
    for key in keys:
        table, row, exp = MODELS[key]
        print(f"{key}: building", flush=True)
        module, predict, trainable_override = build(key, len(labels))
        total, trainable = count(module)
        if trainable_override is not None:
            total, trainable = total + trainable_override, trainable_override
        rec = {"table": table, "row": row, "experiment_id": exp, "params_total": total,
               "params_trainable": trainable, "size_mb_fp32": round(size_mb(module), 2)}
        torch.set_num_threads(default_threads)
        rec["ms_cpu_all_threads"] = time_ms(predict, wavs, n)
        torch.set_num_threads(1)
        rec["ms_cpu_1_thread"] = time_ms(predict, wavs, n)
        torch.set_num_threads(default_threads)
        print(f"  {total / 1e6:.2f} M params, {rec['size_mb_fp32']} MB, "
              f"{rec['ms_cpu_all_threads']['median']:.1f} ms (all threads), "
              f"{rec['ms_cpu_1_thread']['median']:.1f} ms (1 thread)", flush=True)
        out["models"][key] = rec
        del module, predict
    path = Path(results_dir) / "efficiency.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", path)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args(argv)
    run([k.strip() for k in a.models.split(",") if k.strip()], a.n)


if __name__ == "__main__":
    main()
