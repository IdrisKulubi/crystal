"""Phase 7 — stage-1 cry detector (cry vs non-cry on 1 s frames): E40 AST, E41 small CNN.

    python -m src.detection --config configs/exp/E40_ast_detector.yaml [--fold 0 --seeds 0] [--debug]

Data: ``data/detection_folds.csv`` (src.prepare_detection), label set detect_v1, grouped folds.
Same run-folder format and registry as the reason task (RESULTS_SPEC; task: detection).

Front end (both models):
* every clip is band-limited to 4 kHz (16 kHz -> 8 kHz -> 16 kHz). Donate-a-Cry is 8 kHz phone audio while
  ESC-50 is 44.1 kHz; without this the sampling rate alone would reveal the source/label.
* clips are cut into non-overlapping 1 s frames (a final remainder >= 0.5 s is zero-padded, shorter is dropped).
* frames of CRY clips whose RMS is more than 20 dB below the clip's loudest frame are excluded (silence
  between cries is not "cry"); non-cry frames are all kept.
Models:
* ``ast_detector``: AST (AudioSet weights) on a 128-frame (1.28 s) fbank. The position embeddings are cut
  from the 12x101 patch grid to its first 12 time columns (as AST does for shorter inputs). Head: mean of
  [CLS]/[DIST] -> dropout -> linear (1 logit). Full fine-tuning.
* ``cnn_detector``: 64-band log-mel (25 ms / 10 ms) -> 4 conv blocks (16-32-64-64, BN, ReLU, 2x2 max-pool)
  -> global average pool -> linear. Trained from scratch; mobile-sized.
Training: BCE with pos_weight = n_non_cry/n_cry (train frames), AdamW, linear warm-up/decay, each epoch a
random subset of ``frames_per_epoch`` train frames; best epoch = highest validation ROC-AUC (threshold
free), early stop after ``patience``. Decision threshold = the one maximising validation F1. Test frames are
scored ONCE with the final model and threshold.
Metrics (metrics.json): frame-level accuracy / F1 / precision / recall / ROC-AUC / miss rate (the T6 row),
plus clip-level metrics (clip score = max frame score, own threshold on val) and per-source recall /
false-positive rate. false_alarms_per_hour stays null until a long household recording exists (HANDOFF P7).
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.audio import TARGET_SR
from src.config import REPO_ROOT, get_path, load_config
from src.datasets import load_table, load_waveforms
from src.results import metrics as M
from src.results.run import DEFAULT_RESULTS_DIR, Run, write_json
from src.train import resolve_parent

FRAME = TARGET_SR
AST_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_FRAMES = 128
LOG_COLUMNS = ["epoch", "step", "train_loss", "val_loss", "val_roc_auc", "val_f1_at_0.5", "lr", "seconds"]


# --------------------------------------------------------------------------- frames
def band_limit(wav: np.ndarray) -> np.ndarray:
    import librosa

    low = librosa.resample(wav, orig_sr=TARGET_SR, target_sr=8000, res_type="soxr_hq")
    return librosa.resample(low, orig_sr=8000, target_sr=TARGET_SR, res_type="soxr_hq")[:len(wav)].astype(np.float32)


def frames_of(wav: np.ndarray, is_cry: bool, gate_db: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """([n, FRAME] frames, [n] keep mask). Remainder >= FRAME/2 is padded, shorter dropped; >= 1 frame."""
    n = max(1, len(wav) // FRAME + (1 if len(wav) % FRAME >= FRAME // 2 else 0))
    x = np.zeros((n, FRAME), np.float32)
    for i in range(n):
        seg = wav[i * FRAME:(i + 1) * FRAME]
        x[i, :len(seg)] = seg
    keep = np.ones(n, bool)
    if is_cry:
        rms = np.sqrt((x ** 2).mean(1)) + 1e-10
        keep = 20 * np.log10(rms / rms.max()) > -gate_db
    return x, keep


def build_frames(df: pd.DataFrame) -> dict:
    """All frames of all clips (computed once; independent of folds). Returns arrays aligned by frame."""
    xs, clip_idx, keep, frame_no = [], [], [], []
    wavs = load_waveforms(df["path"])
    for i, (w, y) in enumerate(zip(wavs, df["label_idx"])):
        x, k = frames_of(band_limit(np.asarray(w, np.float32)), bool(y == 1))
        xs.append(x)
        keep.append(k)
        clip_idx += [i] * len(x)
        frame_no += list(range(len(x)))
    return {"x": np.concatenate(xs), "clip": np.asarray(clip_idx), "keep": np.concatenate(keep),
            "frame_no": np.asarray(frame_no)}


# --------------------------------------------------------------------------- models
class SmallCNN(nn.Module):
    def __init__(self, channels=(16, 32, 64, 64)):
        super().__init__()
        layers, c_in = [], 1
        for c in channels:
            layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(2)]
            c_in = c
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(c_in, 1)

    def forward(self, x):  # [B, mels, T]
        return self.head(self.body(x.unsqueeze(1)).mean((2, 3))).squeeze(-1)


def ast_short(max_length: int = AST_FRAMES):
    """ASTModel for ``max_length`` fbank frames with position embeddings cut from the pretrained grid."""
    from transformers import ASTConfig, ASTModel

    full = ASTModel.from_pretrained(AST_ID)
    cfg = ASTConfig.from_pretrained(AST_ID, max_length=max_length)
    model = ASTModel(cfg)
    sd = full.state_dict()
    f_dim = (cfg.num_mel_bins - cfg.patch_size) // cfg.frequency_stride + 1
    t_full = (full.config.max_length - cfg.patch_size) // cfg.time_stride + 1
    t_new = (max_length - cfg.patch_size) // cfg.time_stride + 1
    pos = sd["embeddings.position_embeddings"]
    grid = pos[:, 2:].reshape(1, f_dim, t_full, -1)[:, :, :t_new].reshape(1, f_dim * t_new, -1)
    sd["embeddings.position_embeddings"] = torch.cat([pos[:, :2], grid], 1)
    model.load_state_dict(sd)
    return model, getattr(full.config, "_commit_hash", None)


class ASTDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone, self.revision = ast_short()
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.backbone.config.hidden_size, 1))

    def forward(self, x):  # [B, T, mels]
        return self.head(self.backbone(x).last_hidden_state[:, :2].mean(1)).squeeze(-1)


def features(kind: str, frames: np.ndarray) -> np.ndarray:
    if kind == "ast_detector":
        from transformers import ASTFeatureExtractor

        fe = ASTFeatureExtractor.from_pretrained(AST_ID, max_length=AST_FRAMES)
        out = [fe(list(frames[i:i + 256]), sampling_rate=TARGET_SR, return_tensors="np")["input_values"]
               for i in range(0, len(frames), 256)]
        return np.concatenate(out).astype(np.float32)
    if kind == "cnn_detector":
        import librosa

        m = librosa.feature.melspectrogram(y=frames, sr=TARGET_SR, n_fft=400, hop_length=160, n_mels=64,
                                           fmax=4000)
        return np.log(m + 1e-6).astype(np.float32)
    raise ValueError(f"unknown detector {kind!r}")


def make_model(kind: str) -> nn.Module:
    return ASTDetector() if kind == "ast_detector" else SmallCNN()


# --------------------------------------------------------------------------- train / eval
@torch.no_grad()
def predict(model, x: np.ndarray, device, bs: int = 128) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i:i + bs]).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def best_f1_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising F1 on (validation) data; ties -> the higher threshold."""
    order = np.argsort(-scores)
    s, yy = scores[order], y[order]
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    f1 = 2 * tp / np.maximum(tp + fp + yy.sum(), 1)
    last = np.r_[s[1:] != s[:-1], True]       # only cut between distinct scores
    f1 = np.where(last, f1, -1)
    i = int(np.argmax(f1))
    return float(s[i])


def fit_fold(feat: np.ndarray, y: np.ndarray, idx: dict, cfg: dict, seed: int, kind: str) -> dict:
    tcfg = cfg["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = feat
    if kind == "cnn_detector":   # global standardisation from TRAIN frames
        mu, sd = float(x[idx["train"]].mean()), float(x[idx["train"]].std() + 1e-6)
        x = (x - mu) / sd
    model = make_model(kind).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    bs, epochs = int(tcfg["batch_size"]), int(tcfg["epochs"])
    per_epoch = min(int(tcfg.get("frames_per_epoch") or len(idx["train"])), len(idx["train"]))
    total = max(1, math.ceil(per_epoch / bs) * epochs)
    warm = int(float(tcfg.get("warmup_ratio", 0.0)) * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    ytr = y[idx["train"]]
    pos_w = torch.tensor(float((ytr == 0).sum()) / max(1.0, float((ytr == 1).sum())), device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    yva = y[idx["val"]]
    best, best_state, best_epoch, bad, log, step = -1.0, None, 0, 0, [], 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        sub = rng.choice(idx["train"], per_epoch, replace=False)
        tot = 0.0
        for i in range(0, len(sub), bs):
            b = sub[i:i + bs]
            xb, yb = torch.from_numpy(x[b]).to(device), torch.from_numpy(y[b].astype(np.float32)).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = crit(model(xb).float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            tot += float(loss.detach()) * len(b)
        sv = predict(model, x[idx["val"]], device)
        auc = float(roc_auc_score(yva, sv)) if len(set(yva)) == 2 else float("nan")
        vloss = float(nn.functional.binary_cross_entropy_with_logits(torch.from_numpy(sv),
                                                                     torch.from_numpy(yva.astype(np.float32))))
        f05 = M.detection_metrics(yva, 1 / (1 + np.exp(-sv)), 0.5)["f1"]
        log.append({"epoch": epoch, "step": step, "train_loss": tot / len(sub), "val_loss": vloss,
                    "val_roc_auc": auc, "val_f1_at_0.5": f05, "lr": sched.get_last_lr()[0],
                    "seconds": round(time.time() - t0, 2)})
        print(f"    epoch {epoch}: train_loss {tot / len(sub):.4f} val_auc {auc:.4f} val_f1@.5 {f05:.3f}", flush=True)
        if auc > best:
            best, best_epoch, bad = auc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(tcfg.get("patience", 3)):
                break
    model.load_state_dict(best_state)
    cpu = None

    def predict_one(frame_feat: np.ndarray) -> np.ndarray:
        nonlocal cpu
        if cpu is None:
            import copy

            cpu = copy.deepcopy(model).float().cpu().eval()
        return predict(cpu, frame_feat[None], torch.device("cpu"), bs=1)

    return {"val": predict(model, x[idx["val"]], device), "test": predict(model, x[idx["test"]], device),
            "log": log, "best_epoch": best_epoch, "best_val_roc_auc": best, "predict_one": predict_one,
            "x_test": x[idx["test"]],
            "params_total": sum(p.numel() for p in model.parameters()),
            "params_trainable": sum(p.numel() for p in params),
            "revision": getattr(model, "revision", None)}


def evaluate(y, s_val, s_test, y_val, frames_val: pd.DataFrame, frames_test: pd.DataFrame) -> dict:
    """Frame metrics at the val-F1 threshold, clip metrics (max frame score), per-source rates."""
    p_val, p_test = 1 / (1 + np.exp(-s_val)), 1 / (1 + np.exp(-s_test))
    thr = best_f1_threshold(y_val, p_val)
    m = M.detection_metrics(y, p_test, thr)
    pred = p_test >= thr
    by_source = {}
    for ds, g in frames_test.assign(p=pred, y=y).groupby("dataset"):
        if (g["y"] == 1).any():
            by_source[f"{ds}_recall"] = float(g.loc[g["y"] == 1, "p"].mean())
        if (g["y"] == 0).any():
            by_source[f"{ds}_false_positive_rate"] = float(g.loc[g["y"] == 0, "p"].mean())
    cv = frames_val.assign(p=p_val).groupby("clip_id").agg(p=("p", "max"), y=("y", "first"))
    ct = frames_test.assign(p=p_test, y=y).groupby("clip_id").agg(p=("p", "max"), y=("y", "first"))
    clip = M.detection_metrics(ct["y"].to_numpy(), ct["p"].to_numpy(),
                               best_f1_threshold(cv["y"].to_numpy(), cv["p"].to_numpy()))
    m.update({"clip": clip, "by_source": by_source})
    return m, p_test, thr


def main(argv: list[str] | None = None) -> str:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--debug", action="store_true", help="40 clips, fold 0, seed 0, 1 epoch -> results/debug/")
    args = ap.parse_args(argv)
    overrides = list(args.set)
    if args.debug:
        overrides += ["eval.folds=[0]", "eval.seeds=[0]", "train.epochs=1", "train.frames_per_epoch=64"]
    else:
        if args.fold and args.fold != "all":
            overrides.append(f"eval.folds=[{args.fold}]")
        if args.seeds:
            overrides.append(f"eval.seeds=[{args.seeds}]")
    cfg = load_config(args.config, overrides)
    if cfg["task"] != "detection":
        raise SystemExit("src.detection needs task: detection")
    cfg["parent_run"] = resolve_parent(cfg.get("parent_run"))
    results_dir = REPO_ROOT / "results" / "debug" if args.debug else DEFAULT_RESULTS_DIR
    kind = get_path(cfg, "model.type")

    df, labels = load_table(cfg)
    if args.debug:
        df = df.groupby("strat", group_keys=False).apply(lambda g: g.head(12)).reset_index(drop=True)
    t0 = time.time()
    fr = build_frames(df)
    feat = features(kind, fr["x"])
    y_all = df["label_idx"].to_numpy()[fr["clip"]]
    print(f"{len(df)} clips -> {len(fr['x'])} frames ({int((~fr['keep']).sum())} silent cry frames excluded), "
          f"features {feat.shape} in {time.time() - t0:.0f}s", flush=True)

    with Run.create(cfg, results_dir=results_dir) as run:
        first = True
        for fold in cfg["eval"]["folds"]:
            role = df[f"split_f{fold}"].to_numpy()[fr["clip"]]
            idx = {r: np.where((role == r) & fr["keep"])[0] for r in ("train", "val", "test")}
            if args.debug:
                for r in idx:
                    if len(idx[r]) == 0:
                        idx[r] = np.where(fr["keep"])[0][:16]
            for seed in cfg["eval"]["seeds"]:
                ts = time.time()
                out = fit_fold(feat, y_all, idx, cfg, seed, kind)
                secs = time.time() - ts
                d = run.fold_dir(fold, seed)
                pd.DataFrame(out["log"], columns=LOG_COLUMNS).to_csv(d / "train_log.csv", index=False)

                def frame_table(ix):
                    c = fr["clip"][ix]
                    return pd.DataFrame({"frame_id": df["clip_id"].to_numpy()[c] + "#" + fr["frame_no"][ix].astype(str),
                                         "clip_id": df["clip_id"].to_numpy()[c], "group": df["group"].to_numpy()[c],
                                         "dataset": df["dataset"].to_numpy()[c], "y": y_all[ix]})

                fv, ft = frame_table(idx["val"]), frame_table(idx["test"])
                np.save(d / "val_logits.npy", out["val"].astype(np.float32))
                fv.rename(columns={"y": "label_idx"})[["frame_id", "clip_id", "group", "label_idx"]].to_csv(
                    d / "val_meta.csv", index=False)
                run.save_test(fold, seed, out["test"])
                m, p_test, thr = evaluate(ft["y"].to_numpy(), out["val"], out["test"], fv["y"].to_numpy(), fv, ft)
                ft.assign(fold=fold, seed=seed, score=p_test, pred=(p_test >= thr).astype(int)).rename(
                    columns={"y": "y_true"}).to_csv(d / "predictions.csv", index=False)
                ms = None
                if first:
                    xs = out["x_test"][:50]
                    for i in range(5):
                        out["predict_one"](xs[i % len(xs)])
                    tt = []
                    for i in range(50):
                        t1 = time.perf_counter()
                        out["predict_one"](xs[i % len(xs)])
                        tt.append((time.perf_counter() - t1) * 1000)
                    ms = float(np.median(tt))
                    first = False
                m.update({"train_seconds": round(secs, 2), "params_total": int(out["params_total"]),
                          "params_trainable": int(out["params_trainable"]),
                          "inference_ms_per_frame_cpu": ms, "n_val_frames": int(len(idx["val"])),
                          "model_selection": {"best_epoch": out["best_epoch"],
                                              "best_val_roc_auc": out["best_val_roc_auc"],
                                              "epochs_run": len(out["log"]), "threshold_from_val": thr,
                                              "revision": out["revision"]}})
                run.save_metrics(fold, seed, m)
                print(f"  fold {fold} seed {seed}: acc {m['accuracy']:.3f} F1 {m['f1']:.3f} AUC {m['roc_auc']:.3f} "
                      f"clip-F1 {m['clip']['f1']:.3f} ({secs:.0f}s)", flush=True)
    return run.run_id


if __name__ == "__main__":
    main()
