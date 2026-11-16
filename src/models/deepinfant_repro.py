"""E02 — DeepInfant reproduced on our grouped folds.

Port of ``DeepInfantModel`` / ``DeepInfantDataset`` / ``train_model`` from
skytells-research/DeepInfant ``train.py`` (commit 7526568, Apache-2.0; see NOTICE below).
Pre-processing, architecture sizes, optimiser, batch size, epochs and augmentation follow
that file. Deviations, each the minimum needed for the published code to run or to respect our
protocol (also listed in docs/DECISIONS.md):

D1  Squeeze-and-excitation: the published ``conv_layers`` ends with the SE gate itself
    (AdaptiveAvgPool -> 1x1 convs -> Sigmoid), which REPLACES the feature map with a
    [B, 256, 1, 1] gate; the following reshape to 256*10 then fails. We apply the gate
    multiplicatively to the [B, 256, 10, T'] feature map, as the comment "squeeze-and-excitation
    block" intends.
D2  Sequence layout: the published permute (0, 2, 1, 3) of [B, C, F, T] gives [B, F, C, T]
    (frequency-major) although the comment says "(batch, time, channels, freq)". We use
    (0, 3, 1, 2) -> [B, T', C, F] so the LSTM runs over time, matching the comment and
    ``input_size = 256 * 10``.
D3  ``train=True`` datasets call ``self.transform(mel_spec)`` with ``transform=True`` (a bool),
    which raises TypeError. We keep the waveform augmentation that flag enables (random shift
    ±100 ms, Gaussian noise σ=0.005 with p=0.3) and drop the invalid call.
D4  Data loading is replaced by our grouped folds; the best epoch is selected on our VALIDATION
    split by validation accuracy (the original selected on its test split), test is predicted once.
D5  Number of classes = our label set (5 Donate-a-Cry classes) instead of 9.

NOTICE: Portions of this file are derived from DeepInfant (Copyright Skytells AI Research),
licensed under the Apache License 2.0. http://www.apache.org/licenses/LICENSE-2.0
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn

from src.audio import TARGET_SR
from src.datasets import FoldData, load_waveforms, pad_or_crop

N_SAMPLES = 7 * TARGET_SR
MEL_KW = dict(sr=TARGET_SR, n_fft=1024, hop_length=256, n_mels=80, fmin=20, fmax=8000)


def deepinfant_mel(wav: np.ndarray) -> np.ndarray:
    """Exactly DeepInfant's `_process_audio` after augmentation: crop/pad 7 s, mel, dB(ref=max)."""
    import librosa

    wav = pad_or_crop(wav, N_SAMPLES)
    mel = librosa.feature.melspectrogram(y=wav, **MEL_KW)
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def deepinfant_augment(wav: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random time shift in [-1600, 1600) samples (±100 ms) + Gaussian noise (p=0.3, σ=0.005)."""
    shift = int(rng.integers(-1600, 1600))
    if shift > 0:
        wav = np.pad(wav, (shift, 0))[: len(wav)]
    else:
        wav = np.pad(wav, (0, -shift))[(-shift):]
    if rng.random() < 0.3:
        wav = wav + rng.normal(0, 0.005, len(wav)).astype(np.float32)
    return wav.astype(np.float32)


class DeepInfantRepro(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.se = nn.Sequential(  # D1: same layers as the original, applied as a gate
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(256, 16, 1), nn.ReLU(), nn.Conv2d(16, 256, 1), nn.Sigmoid(),
        )
        self.lstm = nn.LSTM(input_size=256 * 10, hidden_size=512, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)
        self.classifier = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
                                        nn.Linear(512, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 80, T]
        b = x.size(0)
        x = self.conv_layers(x)              # [B, 256, 10, T']
        x = x * self.se(x)                   # D1
        x = x.permute(0, 3, 1, 2)            # D2: [B, T', 256, 10]
        x = x.reshape(b, x.size(1), 256 * 10)
        x, _ = self.lstm(x)
        x = x[:, -1, :]                      # last time step (as original)
        return self.classifier(x)


def _batches(n: int, bs: int, rng: np.random.Generator | None):
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


@torch.no_grad()
def _predict(model: nn.Module, mels: np.ndarray, device, bs: int = 32) -> np.ndarray:
    model.eval()
    out = []
    for b in _batches(len(mels), bs, None):
        out.append(model(torch.from_numpy(mels[b]).unsqueeze(1).to(device)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.classifier[-1].out_features), np.float32)


def deepinfant_fit_predict(fd: FoldData, cfg: dict, seed: int) -> dict:
    tcfg = cfg["train"]
    epochs, bs, lr = int(tcfg["epochs"]), int(tcfg["batch_size"]), float(tcfg["lr"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    train_wavs = load_waveforms(fd.train["path"])
    ytr = torch.tensor(fd.train["label_idx"].to_numpy(), dtype=torch.long)
    yva = fd.val["label_idx"].to_numpy()
    val_mels = np.stack([deepinfant_mel(w) for w in load_waveforms(fd.val["path"])])
    test_mels = np.stack([deepinfant_mel(w) for w in load_waveforms(fd.test["path"])])

    model = DeepInfantRepro(len(fd.labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc, best_state, log = -1.0, None, []
    step = 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for b in _batches(len(train_wavs), bs, rng):
            mels = np.stack([deepinfant_mel(deepinfant_augment(train_wavs[i], rng)) for i in b])
            x = torch.from_numpy(mels).unsqueeze(1).to(device)
            y = ytr[b].to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            tot += loss.item() * len(b)
            n += len(b)
            step += 1
        val_logits = _predict(model, val_mels, device)
        val_loss = float(criterion(torch.from_numpy(val_logits), torch.from_numpy(yva)).item()) if len(yva) else None
        val_pred = val_logits.argmax(1)
        val_acc = float((val_pred == yva).mean()) if len(yva) else 0.0
        from sklearn.metrics import f1_score

        val_f1 = float(f1_score(yva, val_pred, labels=range(len(fd.labels)), average="macro", zero_division=0))
        log.append({"epoch": epoch, "step": step, "train_loss": tot / max(n, 1), "val_loss": val_loss,
                    "val_macro_f1": val_f1, "val_acc": val_acc, "lr": lr, "seconds": round(time.time() - t0, 2)})
        if val_acc > best_acc:  # original criterion: best validation accuracy (strictly greater)
            best_acc, best_state, best_epoch = val_acc, copy.deepcopy(model.state_dict()), epoch
    model.load_state_dict(best_state)
    model.eval()

    cpu_model = None

    def predict_one(wav: np.ndarray) -> np.ndarray:
        """CPU inference incl. feature extraction (used for inference_ms_per_clip_cpu)."""
        nonlocal cpu_model
        if cpu_model is None:
            cpu_model = copy.deepcopy(model).cpu().eval()
        return _predict(cpu_model, deepinfant_mel(wav)[None], torch.device("cpu"), bs=1)

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "val_logits": _predict(model, val_mels, device),
        "test_logits": _predict(model, test_mels, device),
        "params_total": n_params,
        "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_log": log,
        "predict_one": predict_one,
        "chosen": {"best_epoch": best_epoch, "best_val_acc": best_acc},
    }
