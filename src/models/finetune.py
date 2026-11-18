"""E10–E14 — fine-tuning pretrained encoders (full or LoRA) with a pooling head.

Backbones: ``ast`` (MIT/ast-finetuned-audioset-10-10-0.4593) and ``wav2vec2_base`` (facebook/wav2vec2-base),
see ``src.models.encoders.ENCODERS``. Fixed choices (docs/DECISIONS.md):
* input: 16 kHz waveform padded/cropped to ``data.clip_seconds`` (7 s), then the model's own feature
  extractor (AST: 1024-frame fbank, as in the probe; wav2vec2: per-clip normalisation).
* heads: ``mean_pool`` = same pooling as the probes (AST: mean of [CLS]/[DIST]; wav2vec2: mean over
  frames) + dropout 0.1 + linear; ``attention`` = learned attention pooling over all output tokens.
* full FT: all backbone weights train, except the wav2vec2 CNN feature encoder, which is frozen
  (standard practice for wav2vec2 fine-tuning).
* LoRA (``model.peft: {type: lora, r, alpha, dropout}``): adapters on the attention query/value
  projections; only adapters + head train.
* optimiser AdamW(lr, weight_decay), linear warm-up (warmup_ratio) then linear decay, batch_size x
  grad_accum, loss ``weighted_ce`` (class weights n/(K*n_c) from the TRAIN split), ``ce`` or ``focal``
  (gamma ``focal_gamma``, alpha = the same class weights); ``sampler`` shuffle or balanced.
* best epoch by ``early_stopping_metric`` on validation (val_macro_f1, or val_loss_weighted = class-
  weighted CE with train weights; strictly better), early stop after ``patience`` epochs
  without improvement; test is predicted once with the best weights.
* fp16 autocast on CUDA.
Phase 5 (src/augment.py), training batches only: ``train.augment`` noise (ESC-50, SNR range; the
fbank/normalisation is then recomputed per batch), SpecAugment (AST), mixup; ``model.context_features``
(age bracket / gender one-hot concatenated to the pooled embedding). With no augmentation, model inputs
are computed once per clip, exactly as before.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from src.audio import TARGET_SR
from src.dapt import resolve_checkpoint
from src.datasets import FoldData, load_waveforms, pad_or_crop
from src.models.encoders import ENCODERS

LORA_TARGETS = {"ast": ["query", "value", "q_proj", "v_proj"], "w2v": ["q_proj", "v_proj"]}  # AST names: transformers 4.x / 5.x


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 4), nn.Tanh(), nn.Linear(dim // 4, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # [B, T, D] -> [B, D]
        w = torch.softmax(self.score(h).squeeze(-1).float(), dim=1).to(h.dtype)
        return (w.unsqueeze(-1) * h).sum(1)


class FineTuneModel(nn.Module):
    def __init__(self, backbone: str, n_classes: int, head: str = "mean_pool", peft: dict | None = None,
                 init_checkpoint: str | None = None, context_dim: int = 0):
        super().__init__()
        from src.models.encoders import load_backbone

        spec = ENCODERS[backbone]
        self.kind, self.head_type = spec["kind"], head
        if self.kind not in ("ast", "w2v"):
            raise ValueError(f"fine-tuning supports ast / wav2vec2-style backbones, not {backbone!r}")
        self.backbone, self.revision = load_backbone(backbone, init_checkpoint)
        dim = self.backbone.config.hidden_size
        if peft:
            if peft.get("type") != "lora":
                raise ValueError(f"unsupported peft {peft!r}")
            from peft import LoraConfig, inject_adapter_in_model

            cfg = LoraConfig(r=int(peft.get("r", 8)), lora_alpha=int(peft.get("alpha", 16)),
                             lora_dropout=float(peft.get("dropout", 0.1)), target_modules=LORA_TARGETS[self.kind])
            self.backbone = inject_adapter_in_model(cfg, self.backbone)
            n_lora = 0
            for name, p in self.backbone.named_parameters():
                p.requires_grad = "lora_" in name
                n_lora += p.numel() if "lora_" in name else 0
            if n_lora == 0:
                raise RuntimeError("LoRA matched no modules")
        elif self.kind == "w2v":
            self.backbone.freeze_feature_encoder()
        if head == "attention":
            self.pool = AttentionPool(dim)
        elif head != "mean_pool":
            raise ValueError(f"model.head must be mean_pool or attention, got {head!r}")
        self.context_dim = int(context_dim)
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(dim + self.context_dim, n_classes))

    def forward(self, x: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        h = self.backbone(x).last_hidden_state
        if self.head_type == "attention":
            z = self.pool(h)
        elif self.kind == "ast":
            z = h[:, :2].mean(1)
        else:
            z = h.mean(1)
        if self.context_dim:
            if ctx is None:
                ctx = torch.zeros(len(z), self.context_dim, device=z.device)
            z = torch.cat([z, ctx.to(z.dtype)], dim=1)
        return self.classifier(z)


def model_inputs(fe, kind: str, wavs: list[np.ndarray], n_samples: int) -> np.ndarray:
    wavs = [pad_or_crop(np.asarray(w, np.float32), n_samples) for w in wavs]
    out = []
    for i in range(0, len(wavs), 32):
        batch = fe(wavs[i:i + 32], sampling_rate=TARGET_SR, return_tensors="np")
        out.append(batch["input_values"].astype(np.float32))
    return np.concatenate(out) if out else np.zeros((0,), np.float32)


def class_weights(y: np.ndarray, k: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=k).astype(np.float64)
    w = np.where(counts > 0, len(y) / (k * np.maximum(counts, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al. 2017), alpha = per-class weights, normalised like weighted CE."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma, self.weight = float(gamma), weight

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, y, reduction="none")
        focal = (1.0 - torch.exp(-ce)).clamp(min=0.0) ** self.gamma * ce
        if self.weight is None:
            return focal.mean()
        w = self.weight[y]
        return (w * focal).sum() / w.sum().clamp(min=1e-12)


def epoch_order(y: np.ndarray, k: int, sampler: str, rng: np.random.Generator) -> np.ndarray:
    """Training order for one epoch. ``shuffle``: a permutation. ``balanced``: len(y) draws with
    replacement, P(clip) proportional to 1/count(its class), so every class is equally likely per draw."""
    if sampler == "shuffle":
        return rng.permutation(len(y))
    if sampler == "balanced":
        counts = np.bincount(y, minlength=k).astype(np.float64)
        p = 1.0 / counts[y]
        return rng.choice(len(y), size=len(y), replace=True, p=p / p.sum())
    raise ValueError(f"train.sampler must be shuffle or balanced, got {sampler!r}")


@torch.no_grad()
def _predict(model, x: np.ndarray, device, bs: int = 16, ctx: np.ndarray | None = None) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i:i + bs]).to(device)
        cb = torch.from_numpy(ctx[i:i + bs]).to(device) if ctx is not None and ctx.shape[1] else None
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model(xb, cb).float().cpu().numpy())
    return np.concatenate(out)


def finetune_fit_predict(fd: FoldData, cfg: dict, seed: int) -> dict:
    from transformers import AutoFeatureExtractor

    mcfg, tcfg = cfg["model"], cfg["train"]
    backbone = mcfg["backbone"]
    k = len(fd.labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    fe = AutoFeatureExtractor.from_pretrained(ENCODERS[backbone]["hf_id"])
    kind = ENCODERS[backbone]["kind"]
    n_samples = int(round(float(cfg["data"]["clip_seconds"]) * TARGET_SR))
    from src.augment import context_matrix, mixup_params, noise_bank, noisy_batch, spec_cfg, specaugment

    aug = dict(tcfg.get("augment") or {})
    snr = aug.get("noise_snr_db")
    noise_prob = float(aug.get("noise_prob", 0.5))
    spec = spec_cfg(aug.get("specaugment"))
    mix_alpha = aug.get("mixup_alpha")
    if aug.get("pitch_shift"):
        raise NotImplementedError("train.augment.pitch_shift is not implemented")
    if spec and kind != "ast":
        raise ValueError("SpecAugment is implemented for the AST fbank input only")
    bank = noise_bank() if snr else []
    if snr:
        print(f"    noise bank: {len(bank)} ESC-50 clips, SNR {snr} dB, p {noise_prob}", flush=True)
    ctx_feats = list(mcfg.get("context_features") or [])
    ctx = {role: context_matrix(part, ctx_feats)
           for role, part in (("train", fd.train), ("val", fd.val), ("test", fd.test))}

    train_wavs = load_waveforms(fd.train["path"])
    x = {role: model_inputs(fe, kind, load_waveforms(part["path"]), n_samples)
         for role, part in (("val", fd.val), ("test", fd.test))}
    x["train"] = model_inputs(fe, kind, train_wavs, n_samples)
    ytr, yva = fd.train["label_idx"].to_numpy(), fd.val["label_idx"].to_numpy()

    ckpt = resolve_checkpoint(mcfg.get("init_checkpoint"))
    model = FineTuneModel(backbone, k, mcfg.get("head") or "mean_pool", mcfg.get("peft"), ckpt,
                          context_dim=ctx["train"].shape[1]).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    bs, accum, epochs = int(tcfg["batch_size"]), int(tcfg["grad_accum"]), int(tcfg["epochs"])
    steps_per_epoch = math.ceil(math.ceil(len(ytr) / bs) / accum)
    total = max(1, steps_per_epoch * epochs)
    warm = int(float(tcfg.get("warmup_ratio", 0.0)) * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    loss_name = tcfg.get("loss", "weighted_ce")
    if loss_name not in ("weighted_ce", "ce", "focal"):
        raise ValueError(f"finetune supports loss weighted_ce / ce / focal, got {loss_name!r}")
    w_train = class_weights(ytr, k)
    if loss_name == "focal":   # alpha = the same class weights as weighted_ce
        criterion = FocalLoss(float(tcfg.get("focal_gamma", 2.0)), w_train.to(device))
    else:
        criterion = nn.CrossEntropyLoss(weight=w_train.to(device) if loss_name == "weighted_ce" else None)
    sampler = tcfg.get("sampler") or "shuffle"
    sel_metric = tcfg.get("early_stopping_metric") or "val_macro_f1"
    if sel_metric not in ("val_macro_f1", "val_loss_weighted"):
        raise ValueError(f"early_stopping_metric must be val_macro_f1 or val_loss_weighted, got {sel_metric!r}")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    patience = int(tcfg.get("patience", 5))

    best_score, best_state, best_epoch, bad, log, step = -math.inf, None, 0, 0, [], 0
    best_f1, sel_trace = -1.0, []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        order = epoch_order(ytr, k, sampler, rng)
        opt.zero_grad(set_to_none=True)
        batches = [order[i:i + bs] for i in range(0, len(order), bs)]
        for bi, b in enumerate(batches):
            if snr:   # noise is mixed into the waveform, so the model input is recomputed for this batch
                xnp = model_inputs(fe, kind, noisy_batch([train_wavs[i] for i in b], bank, tuple(snr),
                                                         noise_prob, rng), n_samples)
            else:
                xnp = x["train"][b]
            if spec:
                xnp = specaugment(xnp, spec, rng)
            cnp = ctx["train"][b]
            if mix_alpha:
                lam, perm = mixup_params(len(b), float(mix_alpha), rng)
                xnp = lam * xnp + (1 - lam) * xnp[perm]
                cnp = lam * cnp + (1 - lam) * cnp[perm]
            xb = torch.from_numpy(np.ascontiguousarray(xnp, dtype=np.float32)).to(device)
            cb = torch.from_numpy(np.ascontiguousarray(cnp, dtype=np.float32)).to(device) if cnp.shape[1] else None
            yb = torch.from_numpy(ytr[b]).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(xb, cb).float()
                if mix_alpha:
                    yp = torch.from_numpy(ytr[b][perm]).to(device)
                    loss = lam * criterion(logits, yb) + (1 - lam) * criterion(logits, yp)
                else:
                    loss = criterion(logits, yb)
            scaler.scale(loss / accum).backward()
            tot += float(loss.detach()) * len(b)
            n += len(b)
            if (bi + 1) % accum == 0 or bi == len(batches) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
        val_logits = _predict(model, x["val"], device, ctx=ctx["val"])
        val_pred = val_logits.argmax(1)
        val_f1 = float(f1_score(yva, val_pred, labels=range(k), average="macro", zero_division=0))
        val_loss = float(nn.functional.cross_entropy(torch.from_numpy(val_logits), torch.from_numpy(yva)))
        log.append({"epoch": epoch, "step": step, "train_loss": tot / max(n, 1), "val_loss": val_loss,
                    "val_macro_f1": val_f1, "val_acc": float((val_pred == yva).mean()),
                    "lr": sched.get_last_lr()[0], "seconds": round(time.time() - t0, 2)})
        print(f"    epoch {epoch}: train_loss {tot / max(n, 1):.3f} val_f1 {val_f1:.3f}", flush=True)
        if sel_metric == "val_macro_f1":
            score = val_f1
        else:  # class-weighted CE on validation (weights from TRAIN counts); lower is better
            score = -float(nn.functional.cross_entropy(torch.from_numpy(val_logits), torch.from_numpy(yva),
                                                       weight=w_train))
        sel_trace.append(round(score, 6))
        if score > best_score:
            best_score, best_f1, best_epoch, bad = score, val_f1, epoch, 0
            best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()

    cpu_model = None

    def predict_one(wav):
        nonlocal cpu_model
        if cpu_model is None:
            cpu_model = copy.deepcopy(model).float().cpu().eval()
        c1 = np.zeros((1, ctx["train"].shape[1]), np.float32)   # timing only: context values do not change cost
        return _predict(cpu_model, model_inputs(fe, kind, [wav], n_samples), torch.device("cpu"), bs=1, ctx=c1)

    return {
        "val_logits": _predict(model, x["val"], device, ctx=ctx["val"]),
        "test_logits": _predict(model, x["test"], device, ctx=ctx["test"]),
        "params_total": sum(p.numel() for p in model.parameters()),
        "params_trainable": sum(p.numel() for p in trainable),
        "train_log": log, "predict_one": predict_one,
        "chosen": {"best_epoch": best_epoch, "best_val_macro_f1": best_f1, "epochs_run": len(log),
                   "backbone": backbone, "revision": model.revision, "init_checkpoint": ckpt, "sampler": sampler, "loss": loss_name,
                   "selection_metric": sel_metric, "selection_trace": sel_trace,
                   "augment": {"noise_snr_db": snr, "noise_prob": noise_prob if snr else None,
                               "noise_bank_clips": len(bank), "specaugment": spec, "mixup_alpha": mix_alpha},
                   "context_features": ctx_feats},
    }
