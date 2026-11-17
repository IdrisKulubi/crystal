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
  grad_accum, loss ``weighted_ce`` (class weights n/(K*n_c) from the TRAIN split) or ``ce``.
* best epoch by validation macro-F1 (strictly greater), early stop after ``patience`` epochs
  without improvement; test is predicted once with the best weights.
* fp16 autocast on CUDA.
No augmentation here (that is Phase 5), so model inputs are computed once per clip.
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
    def __init__(self, backbone: str, n_classes: int, head: str = "mean_pool", peft: dict | None = None):
        super().__init__()
        from transformers import AutoModel

        spec = ENCODERS[backbone]
        self.kind, self.head_type = spec["kind"], head
        if self.kind not in ("ast", "w2v"):
            raise ValueError(f"fine-tuning supports ast / wav2vec2-style backbones, not {backbone!r}")
        self.backbone = AutoModel.from_pretrained(spec["hf_id"])
        self.revision = getattr(self.backbone.config, "_commit_hash", None)
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
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(dim, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x).last_hidden_state
        if self.head_type == "attention":
            z = self.pool(h)
        elif self.kind == "ast":
            z = h[:, :2].mean(1)
        else:
            z = h.mean(1)
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


@torch.no_grad()
def _predict(model, x: np.ndarray, device, bs: int = 16) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i:i + bs]).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model(xb).float().cpu().numpy())
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
    x = {role: model_inputs(fe, kind, load_waveforms(part["path"]), n_samples)
         for role, part in (("train", fd.train), ("val", fd.val), ("test", fd.test))}
    ytr, yva = fd.train["label_idx"].to_numpy(), fd.val["label_idx"].to_numpy()

    model = FineTuneModel(backbone, k, mcfg.get("head") or "mean_pool", mcfg.get("peft")).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    bs, accum, epochs = int(tcfg["batch_size"]), int(tcfg["grad_accum"]), int(tcfg["epochs"])
    steps_per_epoch = math.ceil(math.ceil(len(ytr) / bs) / accum)
    total = max(1, steps_per_epoch * epochs)
    warm = int(float(tcfg.get("warmup_ratio", 0.0)) * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    loss_name = tcfg.get("loss", "weighted_ce")
    if loss_name not in ("weighted_ce", "ce"):
        raise ValueError(f"finetune supports loss weighted_ce / ce, got {loss_name!r}")
    weight = class_weights(ytr, k).to(device) if loss_name == "weighted_ce" else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    patience = int(tcfg.get("patience", 5))

    best_f1, best_state, best_epoch, bad, log, step = -1.0, None, 0, 0, [], 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        order = rng.permutation(len(ytr))
        opt.zero_grad(set_to_none=True)
        batches = [order[i:i + bs] for i in range(0, len(order), bs)]
        for bi, b in enumerate(batches):
            xb = torch.from_numpy(x["train"][b]).to(device)
            yb = torch.from_numpy(ytr[b]).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = criterion(model(xb).float(), yb)
            scaler.scale(loss / accum).backward()
            tot += float(loss) * len(b)
            n += len(b)
            if (bi + 1) % accum == 0 or bi == len(batches) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
        val_logits = _predict(model, x["val"], device)
        val_pred = val_logits.argmax(1)
        val_f1 = float(f1_score(yva, val_pred, labels=range(k), average="macro", zero_division=0))
        val_loss = float(nn.functional.cross_entropy(torch.from_numpy(val_logits), torch.from_numpy(yva)))
        log.append({"epoch": epoch, "step": step, "train_loss": tot / max(n, 1), "val_loss": val_loss,
                    "val_macro_f1": val_f1, "val_acc": float((val_pred == yva).mean()),
                    "lr": sched.get_last_lr()[0], "seconds": round(time.time() - t0, 2)})
        print(f"    epoch {epoch}: train_loss {tot / max(n, 1):.3f} val_f1 {val_f1:.3f}", flush=True)
        if val_f1 > best_f1:
            best_f1, best_epoch, bad = val_f1, epoch, 0
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
        return _predict(cpu_model, model_inputs(fe, kind, [wav], n_samples), torch.device("cpu"), bs=1)

    return {
        "val_logits": _predict(model, x["val"], device),
        "test_logits": _predict(model, x["test"], device),
        "params_total": sum(p.numel() for p in model.parameters()),
        "params_trainable": sum(p.numel() for p in trainable),
        "train_log": log, "predict_one": predict_one,
        "chosen": {"best_epoch": best_epoch, "best_val_macro_f1": best_f1, "epochs_run": len(log),
                   "backbone": backbone, "revision": model.revision},
    }
