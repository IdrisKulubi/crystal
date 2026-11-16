# HANDOFF.md — Full project plan for Claude Code

Owner: **Idris Kulubi** (Nairobi, Kenya). AI engineer and software developer; this is his first
research paper and his first end-to-end pretraining/fine-tuning project. Explain what you are doing
in plain language as you go, and keep him in the loop on decisions.

Handoff written: 16 Nov 2026. Target first submission: **Interspeech 2027**, paper deadline reported as
**9 Feb 2027** (verify on the official site; double-blind, 4 pages + up to 2 pages for references,
acknowledgements and generative-AI disclosure). Fallbacks: MDPI *Sensors* (journal, no page squeeze),
IEEE EMBC 2027, ICASSP 2028 (~Sep 2027 deadline), Deep Learning Indaba / Black in AI workshops.

---------------------------------------------------------------------------------------------

## 1. What we are building and why

**Problem.** Infer from an infant's cry (a) whether the sound is a cry and (b) the likely cause
(hungry, pain/discomfort, tired, burping, other), with honest, calibrated confidence.

**Baseline we improve on — DeepInfant** (Skytells AI Research, Apache-2.0):
- Hybrid CNN–LSTM on log-mel spectrograms (16 kHz, 2048 FFT, 512 hop, 128 mels), clips ~2–7 s,
  augmentation by pitch shift, time stretch, noise. Earlier prototype: fastai ResNet34.
- Reports **89% accuracy** (V2), VGGish 75%, AFP 78%. Trained on a mix of private + public data
  (">10,000 labelled samples after augmentation").
- Weaknesses we exploit (state them neutrally and respectfully in the paper):
  no per-class / F1 / calibration results; partly private data so not reproducible; self-reported
  labels; inconsistent label sets (9 prediction classes vs 5 cry types); split method unspecified
  (risk of recording-level leakage).
- Repo files: `train.py`, `train_huggingface.py`, `prepare_dataset.py`, `predict.py`,
  `deepinfant_training.ipynb`, `TRAINING.md`, `Models/`, `Data/`, `iOS_iPadOS/`, `DeepInfant.pdf`.

**Why 99% "accuracy" is not the goal.** Cry-cause labels are noisy (trained humans ≈33% in one
study cited by Li et al. 2024), Donate-a-Cry is small and heavily "hungry"-dominated, and many
97–99% papers use leaky random splits. Our honest headline claims are instead:
1. ~99% **cry detection** (stage 1) — realistic.
2. **~99% precision on the cries the model chooses to answer** (selective prediction at stated coverage).
3. Clear macro-F1 gain over **reproduced** DeepInfant under a leak-free, baby-grouped protocol.
4. A quantified demonstration of how much random splitting inflates scores (leakage ablation).

**Paper contributions (must match Section 1 of the template):**
C1 rigorous reproducible benchmark (grouped CV, per-class + calibration metrics);
C2 domain-adaptive self-supervised pretraining (DAPT) on unlabelled infant cries;
C3 two-stage, context-aware system (detector → reason classifier with optional context features);
C4 calibrated selective prediction (temperature scaling + abstention, risk–coverage);
C5 (optional) new East African cry dataset — only if ethics approval is obtained in time.

## 2. Files in this kit
- `CLAUDE.md` — standing rules (auto-loaded).
- `docs/HANDOFF.md` — this plan.
- `docs/RESULTS_SPEC.md` — exactly what each run saves; how paper numbers are produced.
- `paper/Infant_Cry_Research_Paper_Template.docx` — full paper with yellow `[placeholders]`.
- `paper/Infant_Cry_Experiment_Guide.docx` — beginner-friendly explanation of every phase
  (includes reference code snippets for data prep, grouped splits, AST fine-tuning, LoRA,
  wav2vec2 DAPT, temperature scaling, ECE, risk–coverage).

## 3. Repository layout to create

```
infant-cry-research/
  CLAUDE.md
  README.md                     # how to reproduce everything, end to end
  pyproject.toml / requirements.txt   # pinned versions
  configs/
    project.yaml                # model_name, label set, paths, default seeds/folds
    defaults.yaml
    exp/E00_*.yaml ... E40_*.yaml   # one per experiment ID (§7)
  data/                         # git-ignored
    raw/ processed/ metadata.csv metadata_folds.csv dedup_report.csv
  baselines/DeepInfant/         # git submodule or clone (keep their LICENSE)
  src/
    __init__.py
    prepare_data.py             # per-dataset loaders -> unified metadata
    dedup.py                    # audio hashing (exact + near-duplicate via chromaprint or embedding cosine)
    make_splits.py              # StratifiedGroupKFold + assertions
    dataset_stats.py
    audio.py                    # load/resample/crop/pad, noise mixing, SpecAugment, mixup
    datasets.py                 # torch Dataset/collate for waveform & spectrogram models
    models/
      deepinfant_repro.py       # their architecture, unchanged
      hf_backbone.py            # AST / wav2vec2 / HuBERT / Whisper-encoder / BEATs wrappers
      heads.py                  # mean/attention pooling, context-fusion MLP
    train.py                    # generic trainer: config -> run folder per RESULTS_SPEC
    linear_probe.py             # frozen embeddings + logistic regression
    svm_baseline.py             # MFCC + SVM
    dapt.py                     # wav2vec2 continued pretraining wrapper
    calibrate.py                # temperature scaling, thresholds, ECE
    significance.py             # grouped paired bootstrap, McNemar
    detection.py                # stage-1 cry detector train/eval, false alarms/hour
    results/
      run.py                    # create run folder, env.json, status, registry append
      metrics.py                # ALL metric definitions (unit-tested)
      aggregate.py              # summary.json
    report.py                   # registry -> tables, paper_numbers.json, RESULTS.md
    figures.py
    fill_paper.py               # optional: paper_numbers.json -> filled copy of the .docx
  notebooks/
    kaggle_train.ipynb          # thin wrapper: clone repo, pip install, python -m src.train ...
    colab_dapt.ipynb
  tests/
    test_metrics.py test_splits.py test_dedup.py test_report.py
  results/                      # committed (small files only; no large logits if >50 MB -> git-lfs or HF)
  paper/
  docs/STATUS.md docs/LAB_NOTEBOOK.md docs/DECISIONS.md
```

## 4. Environment
- Python 3.10–3.12. Core: torch, torchaudio, transformers, datasets, accelerate, peft, evaluate,
  librosa, soundfile, scikit-learn, pandas, numpy, matplotlib, seaborn, pyyaml, wandb (optional),
  pytest. Pin exact versions in requirements.txt once working.
- `TrainingArguments` uses `eval_strategy` in recent transformers (older: `evaluation_strategy`).
  Prefer a small custom PyTorch loop in `src/train.py` over HF Trainer if it makes saving
  val/test logits and the RESULTS_SPEC files simpler — your call; be consistent.
- GPU: Kaggle (free weekly quota), Colab, or rented A100/L4 for DAPT. Ask before spending money.
- Make every script runnable on CPU with `--debug` (5 clips/class, 1 epoch) for fast checks.

## 5. Data

### 5.1 Sources
| ID | Dataset | Access | Use |
|---|---|---|---|
| donateacry | Donate-a-Cry corpus — `git clone https://github.com/gveres/donateacry-corpus` (use `donateacry_corpus_cleaned_and_updated_data/<label>/*.wav`) | public | fine-tune + eval |
| chillanto | Baby Chillanto (INAOE, Mexico) — Idris must request access by email | on request | fine-tune + eval (hunger, pain); normal/asphyxia/deaf for optional health experiment |
| cryceleb | Ubenwa CryCeleb2023 — `huggingface-cli download Ubenwa/CryCeleb2023 --repo-type dataset` (accept terms first) | public, CC | **unlabelled DAPT only** (~6.5 h, 786 infants) |
| audioset_cry / icsd | AudioSet infant-cry clips; ICSD dataset (arXiv 2408.10561) | public | detector positives; DAPT extra |
| esc50 | `git clone https://github.com/karolpiczak/ESC-50` | public | noise augmentation; detector negatives |
| deepinfant_data | whatever is in `baselines/DeepInfant/Data/` | check licence | **check overlap with donateacry** |
| household_noise | Idris records ~30 min of Kenyan home background (TV, cooking, talking, traffic, rain) — no infant, no identifiable speech | own | noise aug + detector negatives |
| own_kenya (optional) | only after ethics approval + NACOSTI licence | own | external test set, context labels |

### 5.2 Grouping (critical)
- Donate-a-Cry filenames encode a device/app UUID (first 5 dash-separated parts of the stem),
  timestamp, app version, gender, age bracket, reason code. **Print 20 filenames and confirm the
  format before relying on it.** Group = UUID. Parse gender and age bracket into metadata
  (age → context feature).
- Chillanto: segments of longer recordings — group by original recording/baby ID if available;
  if only segment files exist, reconstruct recording IDs from filenames and document the rule.
- Dedup: hash decoded 16 kHz PCM (exact) and flag near-duplicates (embedding cosine > 0.98 or
  chromaprint match). Cross-dataset duplicates (e.g., DeepInfant `Data/` vs Donate-a-Cry) must be
  merged into the same group or removed. Save `data/dedup_report.csv`; count goes in dataset_stats.

### 5.3 Unified label set `unified_v1` (confirm after inspecting data; record in DECISIONS.md)
| Source | Original | Unified |
|---|---|---|
| donateacry | hungry | hungry |
| donateacry | belly_pain, discomfort | pain_discomfort |
| donateacry | tired | tired |
| donateacry | burping | burping (merge to `other` if < 20 clips after dedup) |
| chillanto | hunger | hungry |
| chillanto | pain | pain_discomfort |
| chillanto | normal / asphyxia / deaf | excluded from reason task |
Also produce results on Donate-a-Cry alone (5 original classes) so the comparison with DeepInfant
is apples-to-apples.

### 5.4 Splits
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` per evaluation dataset;
within each training portion hold out 15% of groups (seeded) for validation. Assertions:
no group in >1 fold; no clip_id duplicated; every fold has ≥1 example per class (else warn and
document). Save `metadata_folds.csv` + its sha256 → every run's env.json.
Also create `metadata_randomsplit.csv` (non-grouped, clip-level split) used ONLY by E90 leakage demo.

## 6. Phased work plan (each phase: tasks → outputs → acceptance)

**Phase 0 — Bootstrap (day 1–2)**
Tasks: create repo layout, requirements, configs, `src/results/` per RESULTS_SPEC with unit
tests, `report.py` skeleton that runs on an empty registry, STATUS.md, LAB_NOTEBOOK.md.
Accept: `pytest -q` passes; `python -m src.report` produces an empty RESULTS.md listing all
paper placeholders as unfilled.

**Phase 1 — Data (week 1–2)**
Tasks: download public data; prepare_data, dedup, make_splits, dataset_stats; listen-check
helper that exports a random 5 clips/class to `data/listen/` for Idris to audit.
Outputs: metadata files, dedup_report, dataset_stats.json, fig5_class_distribution.
Accept: assertions pass; paper Table 1 numbers present in paper_numbers.json.

**Phase 2 — Baselines (week 2–3)** — E00–E05
- E00 majority-class baseline (always "hungry") — shows why accuracy misleads.
- E01 MFCC(40)+Δ mean/std + SVM (RBF, class_weight=balanced; C,γ tuned on val).
- E02 DeepInfant reproduced: their model + pre-processing unchanged; only data loading swapped
  to our folds; augmentation on train only. If their training code is fastai/old, port the
  architecture faithfully into `src/models/deepinfant_repro.py` and document any deviation.
- E03 DeepInfant pretrained CoreML/released weights evaluated zero-shot on our test folds,
  if the released model can be run (coremltools on macOS, or convert) — optional; label clearly.
- E04 "DeepInfant reported" row is a constant (89%, n/a) — store in configs, not a run.
Accept: T3 rows for E00–E02 filled with mean ± std over 5 folds × 3 seeds.

**Phase 3 — Pretrained encoders (week 3–5)** — E05–E14
- Linear probes (frozen, mean-pooled last layer, logistic regression C tuned on val):
  E05 wav2vec2-base, E06 HuBERT-base, E07 AST (`MIT/ast-finetuned-audioset-10-10-0.4593`),
  E08 Whisper-base encoder, E09 BEATs (Microsoft unilm repo; optional if time).
  Also try best-layer selection (probe each layer on val) — report as a note.
- Fine-tuning: E10 AST full FT, E11 AST + LoRA, E12 wav2vec2 full FT, E13 wav2vec2 + LoRA,
  E14 best-of-the-above with attention pooling head.
Accept: T3 rows filled; best backbone identified and recorded in DECISIONS.md.

**Phase 4 — Domain-adaptive pretraining (week 5–6)** — E20–E23
- Build unlabelled corpus: CryCeleb + AudioSet/ICSD infant cry + (Donate-a-Cry/Chillanto audio
  **excluding all evaluation clips** — safest: exclude labelled datasets entirely; optional variant
  E23 includes train-fold audio only, per fold — expensive, skip unless time).
  Record total hours → `dapt.hours_unlabelled`.
- DAPT wav2vec2-base using HF `examples/pytorch/speech-pretraining/run_wav2vec2_pretraining_no_trainer.py`
  modified to `Wav2Vec2ForPreTraining.from_pretrained(...)` (the stock script builds a fresh model).
  LR ~5e-5, warmup 10%, 10–20k steps; smoke-test 500 steps first. Monitor contrastive loss,
  diversity loss, codevector perplexity (collapse = failure). Save loss curve → fig6.
- E20 DAPT checkpoint + linear probe; E21 DAPT + best FT recipe (LoRA or full);
  E22 DAPT step-count sweep (e.g., 2k/10k/20k) if compute allows.
Accept: T4 row "− domain-adaptive pretraining" computable (E21 vs its non-DAPT parent).

**Phase 5 — Improvements, one at a time (week 6–7)** — E30–E36, each with explicit parent
- E30 + noise augmentation (ESC-50 + household noise, SNR 0–20 dB)
- E31 + SpecAugment
- E32 + mixup (α=0.4)
- E33 loss: focal (γ=2) vs weighted CE
- E34 + context fusion (age bracket from Donate-a-Cry filenames; more if own data)
- E35 ensemble of 3 seeds / backbones (report separately from single model)
- E36 **final model** = best cumulative config ("Ours" row in T3)
Ablations for T4 = remove each component from E36 one at a time (E36-noDAPT, -noContext,
-noNoise, -noFocal). Keep a component only if mean val macro-F1 improves; record decision.

**Phase 6 — Calibration & selective prediction (week 8)**
For every T3 model (not just ours): temperature scaling on val; ECE before/after;
thresholds for 90/70/50% coverage chosen on val; precision at coverage on test; AURC.
Outputs: T5, fig2, fig3, significance ours vs E02 (and vs best non-DAPT model).

**Phase 7 — Stage-1 cry detector (week 8)** — E40–E41
E40 AST fine-tuned binary (cry vs non-cry) on 1 s frames; E41 small CNN (mobile-friendly).
Evaluate frame F1/accuracy on held-out clips and **false alarms per hour** on a long household
recording (Idris records 1–2 h of normal home audio without infants, plus a few with cries if
possible). Median-filter smoothing (window tuned on val). Output T6.

**Phase 8 — Leakage demonstration** — E90
Re-run E02 (DeepInfant repro) and E36 (ours) on `metadata_randomsplit.csv`. Report the inflation.
Goes in T4 last row and Discussion.

**Phase 9 — Efficiency & deployment numbers**
Params total/trainable, CPU ms/clip, model size MB; optional ONNX/TFLite export and on-phone
timing. Goes into T2/T3 notes and Discussion.

**Phase 10 — Fill the paper (week 9–11)**
1. `python -m src.report` → verify RESULTS.md has no unfilled placeholders that experiments
   should have filled.
2. `src/fill_paper.py`: copy the template, replace placeholders whose text matches the map in
   §9 using paper_numbers.json, leave others highlighted. Never modify the original template.
   (Use python-docx; placeholders are yellow-highlighted runs.)
3. Draft Results/Discussion text **from the numbers** (state facts, no hype). Idris reviews.
4. Produce the Interspeech 4+2-page version (LaTeX paper kit from the official site / Overleaf):
   anonymise (no names, no GitHub URL — use "link withheld for review"), keep T3/T4/T5, move
   extra detail to the long journal version.
5. Verify every reference against its source (template marks uncertain ones "verify").

**Optional Phase 11 — Own Kenyan dataset** (only with ethics approval + NACOSTI licence +
Kenya Data Protection Act compliance). Context log per recording: age, minutes since last feed,
last nappy change, sleep, illness/vaccination today, what soothed the baby (= label evidence).
Use as external test set (T7: cross-dataset generalisation).

## 7. Experiment matrix (IDs are used in configs, run_ids and the registry)

| ID | Description | Parent | Paper |
|---|---|---|---|
| E00 | Majority class | — | T3 note |
| E01 | MFCC + SVM | — | T3 |
| E02 | DeepInfant reproduced (grouped CV) | — | T3 |
| E03 | DeepInfant released weights, zero-shot (optional) | — | T3 note |
| E05–E09 | Linear probes: w2v2, HuBERT, AST, Whisper-enc, BEATs | — | T3 |
| E10–E14 | Fine-tuning AST/w2v2, full vs LoRA, pooling | E05/E07 | T3 |
| E20–E22 | DAPT + probe / + FT / step sweep | E12 or E13 | T3, T4 |
| E30–E35 | Noise, SpecAugment, mixup, focal, context, ensemble | previous best | T4 |
| E36 | Final model ("Ours") | cumulative | T3, T4, T5 |
| E36-ablations | minus each component | E36 | T4 |
| E40–E41 | Cry detector AST / small CNN | — | T6 |
| E90 | Random-split leakage demo (E02 & E36) | E02, E36 | T4 |

## 8. Default hyperparameters & search ranges (tune on validation only)
| Param | Default | Range |
|---|---|---|
| clip length | 7 s (donateacry), pad/crop | 3–10 s |
| batch | 8 (grad-accum to 32 effective) | — |
| LR full FT | 3e-5 | 1e-5 – 5e-5 |
| LR LoRA | 2e-4 | 1e-4 – 5e-4 |
| LoRA r / α | 8 / 16 | r ∈ {4, 8, 16} |
| epochs | 15, early stop patience 5 on val macro-F1 | 10–30 |
| weight decay | 0.01 | 0–0.05 |
| focal γ | 2 | {1, 2, 3} |
| mixup α | 0.4 | {0.2, 0.4} |
| noise SNR | U(0, 20) dB, p=0.5 | — |
| DAPT LR / steps | 5e-5 / 20k | 1e-5 – 1e-4 / 2k–30k |
Log the search itself (every trial is a run with `paper.table: tuning`) so Appendix B can be written.

## 9. Paper placeholder → data key map (for fill_paper.py)
| Paper location | Placeholder text | Key |
|---|---|---|
| Title/abstract/body | [MODEL NAME] | model_name |
| Abstract | [datasets] | datasets.names_for_abstract |
| Abstract | cry-detection accuracy of [XX.X]% | abstract.detection_acc |
| Abstract | macro-F1 of [0.XX] (versus [0.XX] …) | abstract.reason_macro_f1_ours / _deepinfant |
| Abstract | [XX.X]% precision at [XX]% coverage | abstract.selective_precision / _coverage |
| Abstract | ECE of [0.0XX] | abstract.ece_ours |
| 1 Intro C2 | [N hours] | dapt.hours_unlabelled |
| 1 Intro C4 | [XX]% precision | abstract.selective_precision |
| 3 Table 1 | [N] clips etc. | datasets.<id>.n_clips … |
| 3.1 | [K] classes, class list, min count [N] | datasets.label_set.* |
| 4.2 | clip length [L] | config value from E36 |
| 4.3 | [N] steps, batch [B], LR [η] | T2.dapt.* |
| 4.4 | LoRA / freeze, SNR range | T2.finetune.* |
| 5.4 Table 2 | all | T2.* |
| 6 Tables 3–5 | all `[ ]` cells | T3.*, T4.*, T5.* |
| 6.4 | detection results | T6.* |
Any placeholder that is a writing task (e.g., "[Describe the main finding…]") is drafted as text
in `paper/drafts/<section>.md` for Idris to review — not auto-inserted.

## 10. Risks & gotchas
- **Tiny, imbalanced data** → high fold variance. Use 3 seeds × 5 folds; report std; consider
  repeated CV if variance is extreme. Consider merging rare classes (document it).
- **Chillanto may not arrive in time** → the whole pipeline must work on Donate-a-Cry alone; Chillanto
  results become an extra table if it arrives.
- **Donate-a-Cry group parsing may be wrong** → verify with Idris by printing examples; if groups
  are unreliable, fall back to grouping by (uuid OR timestamp-session) and document.
- **DAPT may not help** → that is a valid result; report it. Do not keep tuning until it "wins" on test.
- **AST expects 10.24 s @ 16 kHz (1024 frames)** — the feature extractor pads; check padding
  isn't dominating short clips (consider max_length matching clip length).
- **wav2vec2 on short clips**: min ~0.5 s; group-norm in base model is sensitive to padding — use
  attention masks where supported (wav2vec2-base does NOT use attention_mask by default — check
  `feature_extractor.return_attention_mask`).
- **Kaggle sessions time out** → checkpoint every epoch; resumable training.
- **Licences**: DeepInfant Apache-2.0 (keep notice for reused code); CryCeleb CC (check exact
  variant, attribute); Donate-a-Cry (check repo licence); ESC-50 CC BY-NC (non-commercial!).
- **Ethics**: infant data; no public raw audio; ask before any upload.
- **arXiv**: since Jan 2026 first-time submitters need a personal endorsement from an established
  author in the category — Idris needs an endorser/co-author. Check Interspeech's anonymity
  policy before posting a preprint.

## 11. Definition of done
- Every T1–T6 cell and every abstract number in paper_numbers.json, with source run IDs.
- All figures in results/figures (PNG + PDF).
- Significance test ours vs reproduced DeepInfant.
- README reproduces everything from a clean clone (data download steps documented).
- Filled long-form .docx draft + anonymised Interspeech-format draft ready for Idris's review.
- STATUS.md shows all phases complete; LAB_NOTEBOOK.md documents decisions and failures.

## 12. Templates

### docs/STATUS.md
```markdown
# STATUS
Last updated: <date> by Claude Code
## Current phase: <n>
## Next task:
## Blockers / needs Idris:
- [ ] Request Baby Chillanto access (email INAOE)
- [ ] Accept CryCeleb terms on Hugging Face
- [ ] Confirm model name (working: CRYSTAL)
- [ ] Record household noise + long household recording
- [ ] Find arXiv endorser / co-author
## Phase checklist
- [ ] P0 bootstrap  - [ ] P1 data  - [ ] P2 baselines  - [ ] P3 encoders
- [ ] P4 DAPT  - [ ] P5 improvements  - [ ] P6 calibration  - [ ] P7 detector
- [ ] P8 leakage  - [ ] P9 efficiency  - [ ] P10 paper
## Unfilled paper numbers (from RESULTS.md):
```

### docs/LAB_NOTEBOOK.md entry
```markdown
## 2026-11-20 — E10 AST fine-tune
Why: test whether full FT beats linear probe (E07).
What: lr 3e-5, 15 ep, weighted CE, folds 0–4 × seeds 0–2. run_id: E10__...
Result: val macro-F1 0.xx ± 0.xx (test not looked at for decisions).
Observations: tired vs pain confused; fold 3 unstable (few groups).
Next: try LoRA (E11).
```

### docs/DECISIONS.md entry
`<date> | decision | reason | evidence (run ids)` — e.g., label merges, chosen backbone, kept/dropped components.

## 13. First prompt Idris can give Claude Code
> Read CLAUDE.md and docs/HANDOFF.md and docs/RESULTS_SPEC.md. Then do Phase 0: create the repo
> structure, requirements, the results module with unit tests, report.py skeleton, STATUS.md and
> LAB_NOTEBOOK.md. Explain each step to me simply. Stop and show me before starting Phase 1.

## 14. Key references (full list in the paper template)
DeepInfant (github.com/skytells-research/DeepInfant); Donate-a-Cry (github.com/gveres/donateacry-corpus);
CryCeleb arXiv:2305.00969; ICSD arXiv:2408.10561; Bonafos et al. Interspeech 2025 arXiv:2509.02259;
Owino et al. arXiv:2512.16271 and arXiv:2512.13880 (Strathmore, Nairobi — potential collaborators);
Li et al. Sensors 2024 (SE-ResNet-Transformer); Onu et al. arXiv:1711.06405 (Ubenwa asphyxia);
Valizade Shayegh & Tadj, Electronics 2025 (sepsis/RDS from cries); wav2vec 2.0; HuBERT; AST; HTS-AT;
BEATs; Whisper; SpecAugment; mixup; LoRA; temperature scaling (Guo 2017); selective classification
(Geifman & El-Yaniv 2017); focal loss.
