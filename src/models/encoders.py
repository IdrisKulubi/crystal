"""Pretrained audio encoders (Hugging Face) behind one small interface.

``Encoder(key).layer_embeddings(wav)`` -> float32 [n_layers, dim]: one clip-level vector per
hidden layer (index 0 = input embeddings, index -1 = final output incl. any final LayerNorm).
Clips are encoded one at a time at their true length, so there is no padding to pool over
(wav2vec2-base/HuBERT-base use no attention mask - HANDOFF §10).

Pooling per encoder (fixed, documented in docs/DECISIONS.md):
* wav2vec2 / HuBERT: mean over all frames.
* AST: mean of the [CLS] and [DIST] tokens (what ASTForAudioClassification pools); the input is
  padded to 1024 frames (10.24 s) by the official feature extractor, the position embeddings are
  fixed to that length, and the two summary tokens are used instead of mean-pooling patches so that
  padding patches are not averaged in.
* Whisper encoder: input is padded to 30 s by design; mean over the first ceil(n_samples/320)
  output frames (50 frames/s), i.e. only frames covering real audio.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from src.audio import TARGET_SR

ENCODERS = {
    "wav2vec2_base": {"hf_id": "facebook/wav2vec2-base", "kind": "w2v"},
    "hubert_base": {"hf_id": "facebook/hubert-base-ls960", "kind": "w2v"},
    "ast": {"hf_id": "MIT/ast-finetuned-audioset-10-10-0.4593", "kind": "ast"},
    "whisper_base": {"hf_id": "openai/whisper-base", "kind": "whisper"},
}


class Encoder:
    def __init__(self, key: str, device: str | torch.device = "cpu"):
        from transformers import AutoFeatureExtractor, AutoModel

        if key not in ENCODERS:
            raise ValueError(f"unknown encoder {key!r}; known: {sorted(ENCODERS)}")
        self.key, spec = key, ENCODERS[key]
        self.hf_id, self.kind = spec["hf_id"], spec["kind"]
        self.device = torch.device(device)
        self.fe = AutoFeatureExtractor.from_pretrained(self.hf_id)
        model = AutoModel.from_pretrained(self.hf_id)
        self.model = (model.encoder if self.kind == "whisper" else model).to(self.device).eval()
        self.revision = getattr(model.config, "_commit_hash", None)
        self.n_params = sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def layer_embeddings(self, wav: np.ndarray) -> np.ndarray:
        wav = np.asarray(wav, dtype=np.float32)
        if self.kind == "w2v":
            x = self.fe(wav, sampling_rate=TARGET_SR, return_tensors="pt").input_values.to(self.device)
            out = self.model(x, output_hidden_states=True)
            layers = [h[0].mean(0) for h in out.hidden_states[:-1]] + [out.last_hidden_state[0].mean(0)]
        elif self.kind == "ast":
            x = self.fe(wav, sampling_rate=TARGET_SR, return_tensors="pt").input_values.to(self.device)
            out = self.model(x, output_hidden_states=True)
            layers = [h[0, :2].mean(0) for h in out.hidden_states[:-1]] + [out.last_hidden_state[0, :2].mean(0)]
        elif self.kind == "whisper":
            x = self.fe(wav, sampling_rate=TARGET_SR, return_tensors="pt").input_features.to(self.device)
            out = self.model(x, output_hidden_states=True)
            n = max(1, min(out.last_hidden_state.shape[1], math.ceil(len(wav) / 320)))
            layers = [h[0, :n].mean(0) for h in out.hidden_states[:-1]] + [out.last_hidden_state[0, :n].mean(0)]
        else:  # pragma: no cover
            raise ValueError(self.kind)
        return torch.stack(layers).float().cpu().numpy()
