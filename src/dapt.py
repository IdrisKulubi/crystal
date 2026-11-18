"""Phase 4 — domain-adaptive pretraining (DAPT) of wav2vec2-base on unlabelled infant cries.

    python -m src.dapt --config configs/dapt/D01_w2v2_cryceleb.yaml [--max-steps N] [--debug]

Continues wav2vec2's own self-supervised objective (contrastive + codebook-diversity loss) from the
released ``facebook/wav2vec2-base`` weights (``Wav2Vec2ForPreTraining.from_pretrained`` — the stock
HF script would build a fresh model, HANDOFF Phase 4). Corpus: ``data/dapt/manifest.csv`` from
``src.build_dapt_corpus`` (no labelled / evaluation audio in it).

Recipe (configs/dapt/*.yaml): random ``crop_seconds`` crops (shorter chunks zero-padded, padding is
never masked nor used as a negative), masking p=0.65 / span 10 (wav2vec2 paper), 100 negatives,
diversity weight 0.1, gumbel temperature 2 -> 0.5 (x0.999995/step), AdamW + linear warm-up/decay,
fp16 autocast on CUDA. Loss is normalised by the number of masked frames, as in the HF example.

Collapse guard: if the codevector perplexity stays below ``collapse_perplexity`` for
``collapse_patience`` logged steps after warm-up, the run stops with status "failed".

Outputs (no audio):
* results/dapt/<run_id>/{config.yaml, env.json, train_log.csv, summary.json, status.json}
* checkpoints/<run_id>/  (HF save_pretrained, git-ignored; loaded by E20/E21 via model.init_checkpoint)
"""
from __future__ import annotations

import argparse
import math
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from src.audio import TARGET_SR, load_audio
from src.config import REPO_ROOT, load_yaml
from src.results.run import collect_env, git_info, iso, make_run_id, read_json, utcnow, write_json

DAPT_DIR = REPO_ROOT / "results" / "dapt"
CKPT_DIR = REPO_ROOT / "checkpoints"
DEFAULTS = {
    "base_model": "facebook/wav2vec2-base", "manifest": "data/dapt/manifest.csv",
    "crop_seconds": 4.0, "batch_size": 16, "grad_accum": 1, "max_steps": 10000, "lr": 5.0e-5,
    "weight_decay": 0.01, "warmup_ratio": 0.1, "mask_prob": 0.65, "mask_length": 10,
    "diversity_weight": 0.1, "gumbel_max": 2.0, "gumbel_min": 0.5, "gumbel_decay": 0.999995,
    "log_every": 25, "collapse_perplexity": 2.0, "collapse_patience": 20, "seed": 0,
}


def resolve_checkpoint(ref: str | None, dapt_dir: Path = DAPT_DIR, ckpt_dir: Path = CKPT_DIR) -> str | None:
    """``dapt:D01`` -> path of the latest completed D01 checkpoint; any other value is returned as is."""
    if not ref or not str(ref).startswith("dapt:"):
        return ref
    exp = str(ref)[5:]
    done = []
    for st in sorted(Path(dapt_dir).glob(f"{exp}__*/status.json")):
        s = read_json(st)
        if s.get("status") == "completed" and (Path(ckpt_dir) / st.parent.name).is_dir():
            done.append((s.get("finished_utc", ""), st.parent.name))
    if not done:
        raise FileNotFoundError(f"no completed DAPT run with a checkpoint for {ref!r} in {dapt_dir}")
    return (Path(ckpt_dir) / max(done)[1]).as_posix()


def crop_batch(wavs: list[np.ndarray], n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random n-sample crops (zero-padded when shorter) + true lengths."""
    x = np.zeros((len(wavs), n), np.float32)
    lens = np.zeros(len(wavs), np.int64)
    for i, w in enumerate(wavs):
        if len(w) > n:
            s = int(rng.integers(0, len(w) - n + 1))
            w = w[s:s + n]
        x[i, :len(w)] = w
        lens[i] = len(w)
    return x, lens


def normalise(x: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """Per-clip zero-mean/unit-variance over the real samples (wav2vec2 feature extractor, do_normalize)."""
    out = np.zeros_like(x)
    for i, n in enumerate(lens):
        seg = x[i, :n]
        out[i, :n] = (seg - seg.mean()) / np.sqrt(seg.var() + 1e-7)
    return out


def main(argv: list[str] | None = None) -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--debug", action="store_true", help="tiny run under results/debug, no checkpoint kept")
    args = ap.parse_args(argv)

    import torch
    from transformers import Wav2Vec2ForPreTraining
    from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices

    cfg = {**DEFAULTS, **load_yaml(args.config)}
    if args.max_steps:
        cfg["max_steps"] = args.max_steps
    git = git_info()
    run_id = make_run_id(cfg["experiment_id"], cfg["name"], utcnow(), git["git7"])
    out_dir = (REPO_ROOT / "results" / "debug" / "dapt" if args.debug else DAPT_DIR) / run_id
    ckpt = (REPO_ROOT / "results" / "debug" / "ckpt" if args.debug else CKPT_DIR) / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    import yaml

    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    env = collect_env({"data": {}}, git)
    write_json(out_dir / "env.json", env)
    write_json(out_dir / "status.json", {"run_id": run_id, "status": "running", "started_utc": iso(utcnow())})
    print("DAPT run", run_id, flush=True)

    log: list[dict] = []
    try:
        rng = np.random.default_rng(int(cfg["seed"]))
        torch.manual_seed(int(cfg["seed"]))
        man = pd.read_csv(REPO_ROOT / cfg["manifest"])
        if args.debug:
            man = man.head(64)
        t0 = time.time()
        wavs = [load_audio(REPO_ROOT / p)[0] for p in man["path"]]
        hours = sum(len(w) for w in wavs) / TARGET_SR / 3600
        print(f"loaded {len(wavs)} chunks, {hours:.2f} h in {time.time() - t0:.0f}s", flush=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Wav2Vec2ForPreTraining.from_pretrained(cfg["base_model"]).to(device)
        model.train()
        mcfg = model.config
        n = int(cfg["crop_seconds"] * TARGET_SR)
        bs, accum, steps = int(cfg["batch_size"]), int(cfg["grad_accum"]), int(cfg["max_steps"])
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]),
                                betas=(0.9, 0.98), eps=1e-6)
        warm = int(float(cfg["warmup_ratio"]) * steps)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        low_ppl, status, reason = 0, "completed", None
        acc = {"loss": 0.0, "contrastive": 0.0, "diversity": 0.0, "ppl": 0.0, "masked": 0.0, "k": 0}
        t_log = time.time()
        for step in range(1, steps + 1):
            model.set_gumbel_temperature(max(float(cfg["gumbel_max"]) * float(cfg["gumbel_decay"]) ** step,
                                             float(cfg["gumbel_min"])))
            for _ in range(accum):
                idx = rng.integers(0, len(wavs), bs)
                x, lens = crop_batch([wavs[i] for i in idx], n, rng)
                x = normalise(x, lens)
                t_len = int(model._get_feat_extract_output_lengths(n))
                f_lens = model._get_feat_extract_output_lengths(torch.from_numpy(lens)).numpy()
                sub = (np.arange(t_len)[None, :] < f_lens[:, None]).astype(np.int64)
                mask = _compute_mask_indices((bs, t_len), float(cfg["mask_prob"]), int(cfg["mask_length"]),
                                             attention_mask=torch.from_numpy(sub), min_masks=2)
                mask &= sub.astype(bool)
                neg = _sample_negative_indices((bs, t_len), mcfg.num_negatives, mask_time_indices=mask)
                n_masked = max(1, int(mask.sum()))
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    out = model(torch.from_numpy(x).to(device),
                                mask_time_indices=torch.from_numpy(mask).to(device),
                                sampled_negative_indices=torch.from_numpy(neg).long().to(device))
                contrastive = out.contrastive_loss.float() / n_masked
                diversity = out.diversity_loss.float() / n_masked
                loss = contrastive + float(cfg["diversity_weight"]) * diversity
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {step}")
                scaler.scale(loss / accum).backward()
                acc["loss"] += float(loss.detach())
                acc["contrastive"] += float(contrastive.detach())
                acc["diversity"] += float(diversity.detach())
                acc["ppl"] += float(out.codevector_perplexity.detach().float())
                acc["masked"] += n_masked / max(1, int(sub.sum()))
                acc["k"] += 1
            scaler.unscale_(opt)
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            if step % int(cfg["log_every"]) == 0 or step == steps:
                k = acc.pop("k")
                row = {"step": step, **{kk: v / k for kk, v in acc.items()}, "grad_norm": gnorm,
                       "lr": sched.get_last_lr()[0], "gumbel_temp": model.quantizer.temperature
                       if hasattr(model.quantizer, "temperature") else None,
                       "seconds": round(time.time() - t_log, 2)}
                row["codevector_perplexity"] = row.pop("ppl")
                row["mask_fraction"] = row.pop("masked")
                log.append(row)
                acc = {"loss": 0.0, "contrastive": 0.0, "diversity": 0.0, "ppl": 0.0, "masked": 0.0, "k": 0}
                t_log = time.time()
                print(f"  step {step}: loss {row['loss']:.3f} contr {row['contrastive']:.3f} "
                      f"div {row['diversity']:.3f} ppl {row['codevector_perplexity']:.1f}", flush=True)
                pd.DataFrame(log).to_csv(out_dir / "train_log.csv", index=False, lineterminator="\n")
                if step > warm and row["codevector_perplexity"] < float(cfg["collapse_perplexity"]):
                    low_ppl += 1
                    if low_ppl >= int(cfg["collapse_patience"]):
                        status, reason = "failed", "codebook collapse (perplexity below threshold)"
                        break
                else:
                    low_ppl = 0
        if status == "completed":
            ckpt.mkdir(parents=True, exist_ok=False)
            model.save_pretrained(ckpt)
            from transformers import AutoFeatureExtractor

            AutoFeatureExtractor.from_pretrained(cfg["base_model"]).save_pretrained(ckpt)
        first, last = (log[0], log[-1]) if log else ({}, {})
        write_json(out_dir / "summary.json", {
            "run_id": run_id, "status": status, "reason": reason, "steps_run": log[-1]["step"] if log else 0,
            "hours_corpus": round(hours, 4), "n_chunks": len(wavs),
            "base_model": cfg["base_model"],
            "base_revision": getattr(model.config, "_commit_hash", None),
            "loss_first": first.get("loss"), "loss_last": last.get("loss"),
            "perplexity_first": first.get("codevector_perplexity"),
            "perplexity_last": last.get("codevector_perplexity"),
            "checkpoint": ckpt.relative_to(REPO_ROOT).as_posix() if status == "completed" else None,
            "minutes": round((time.time() - t0) / 60, 1),
        })
    except Exception as e:  # log failures too (CLAUDE.md rule 4)
        status, reason = "failed", f"{type(e).__name__}: {e}"
        traceback.print_exc()
        if log:
            pd.DataFrame(log).to_csv(out_dir / "train_log.csv", index=False, lineterminator="\n")
    write_json(out_dir / "status.json", {"run_id": run_id, "status": status, "reason": reason,
                                         "started_utc": env["started_utc"], "finished_utc": iso(utcnow())})
    print("DAPT", status, reason or "", flush=True)
    return run_id


if __name__ == "__main__":
    main()
